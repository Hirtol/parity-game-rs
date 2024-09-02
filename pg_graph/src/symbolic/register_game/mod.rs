use std::{
    fmt::{Debug, Formatter},
    time::Duration,
};

use ecow::{eco_vec, EcoVec};
use itertools::Itertools;
use oxidd_cache::StatisticsGenerator;
use oxidd_core::{
    function::Function,
    HasApplyCache,
    Manager, ManagerRef, util::{AllocResult, OptBool, Subst}, WorkerManager,
};

use variable_order::VariableAllocatorInfo;

use crate::{
    explicit,
    explicit::{ParityGame, ParityGraph, register_game::Rank},
    Owner,
    Priority,
    symbolic, symbolic::{
        BddError,
        helpers::{CachedBinaryEncoder, MultiEncoder, SymbolicEncoder},
        oxidd_extensions::{BooleanFunctionExtensions, FunctionVarRef, GeneralBooleanFunction},
        sat::TruthAssignmentsIterator, SymbolicParityGame,
    },
};

pub mod helpers;
pub(crate) mod test_helpers;
mod variable_order;

pub struct SymbolicRegisterGame<F: Function> {
    pub k: Rank,
    pub controller: Owner,
    pub manager: F::ManagerRef,
    pub variables: RegisterVertexVars<F>,
    pub variables_edges: RegisterVertexVars<F>,
    pub conjugated_variables: F,
    pub conjugated_v_edges: F,

    pub vertices: F,
    pub v_even: F,
    pub v_odd: F,
    pub e_move: F,
    pub e_i_all: F,
    pub e_i: Vec<F>,
    pub priorities: ahash::HashMap<Priority, F>,

    pub base_true: F,
    pub base_false: F,
}

impl<F> SymbolicRegisterGame<F>
where
    F: GeneralBooleanFunction,
    for<'id> F::Manager<'id>: WorkerManager,
    for<'a, 'b> TruthAssignmentsIterator<'b, 'a, F>: Iterator<Item = Vec<OptBool>>,
{
    /// Construct a new symbolic register game straight from an explicit parity game.
    ///
    /// See [crate::register_game::RegisterGame::construct]
    #[tracing::instrument(name = "Build Symbolic Register Game", skip_all)]
    pub fn from_symbolic(explicit: &ParityGame, k: Rank, controller: Owner) -> symbolic::Result<Self> {
        check_rg_invariants(explicit)?;

        let manager = F::new_manager(
            explicit.vertex_count(),
            (explicit.vertex_count().ilog2() * 4096 * (k as u32 + 1)) as usize,
            12,
        );

        Self::from_manager(manager, explicit, k, controller, variable_order::default_alloc_vars)
    }

    pub fn from_manager<T>(
        manager: F::ManagerRef,
        explicit: &ParityGame,
        k: Rank,
        controller: Owner,
        variable_provider: T,
    ) -> symbolic::Result<Self>
    where
        T: for<'a> FnOnce(
            &mut <F as Function>::Manager<'a>,
            VariableAllocatorInfo,
        ) -> symbolic::Result<(RegisterVertexVars<F>, RegisterVertexVars<F>)>,
    {
        manager.with_manager_exclusive(|man| man.set_threading_enabled(false));

        let k = k as usize;
        let num_registers = k + 1;

        let extra_priorities = if controller == Owner::Even { 1 } else { 2 };
        let max_reg_priority = 2 * k + extra_priorities;
        let register_bits_needed = (explicit.priority_max() as f64 + 1.).log2().ceil() as usize;
        // Calculate the amount of variables we'll need for the binary encodings of the registers and vertices.
        let n_register_vars = register_bits_needed * num_registers;
        let n_vertex_vars = (explicit.vertex_count() as f64).log2().ceil() as usize;
        let n_prio_vars = (max_reg_priority as f64 + 1.).log2().ceil() as usize;

        // Construct base building blocks for the BDD
        let base_true = manager.with_manager_exclusive(|man| F::t(man));
        let base_false = manager.with_manager_exclusive(|man| F::f(man));

        let (variables, edge_variables) = manager.with_manager_exclusive(|man| {
            variable_provider(
                man,
                VariableAllocatorInfo {
                    n_register_vars,
                    n_vertex_vars,
                    n_priority_vars: n_prio_vars,
                    n_next_move_vars: 1,
                },
            )
        })?;

        tracing::debug!("Starting symbolic parity game construction");
        let sg = SymbolicParityGame::from_explicit_impl_vars(
            explicit,
            manager.clone(),
            variables.vertex_vars().iter().cloned().collect(),
            edge_variables.vertex_vars().iter().cloned().collect(),
        )?;

        tracing::debug!("Starting player vertex set construction");

        let (s_even, s_odd) = match controller {
            Owner::Even => {
                let odd = sg.vertices_odd.and(variables.next_move_var())?;
                (sg.vertices.diff(&odd)?, odd)
            }
            Owner::Odd => {
                let even = sg.vertices_even.and(variables.next_move_var())?;
                let odd = sg.vertices.diff(&even)?;
                (even, odd)
            }
        };

        tracing::debug!("Starting E_move construction");
        let mut prio_edge_encoder =
            CachedBinaryEncoder::new(&manager, edge_variables.priority_vars().iter().cloned().collect());

        // Ensure all registers remain the same past the E_move transition
        let iff_register_condition = variables
            .register_vars()
            .iter()
            .zip(edge_variables.register_vars())
            .try_fold(base_true.clone(), |acc, (r, r_next)| acc.and(&r.equiv(r_next)?))?;

        // t = 1 && t' = 0 && (r0 <-> r0') && (p' = 0)
        let edge_priority_zero = prio_edge_encoder.encode(0)?;
        let e_move = sg
            .edges
            .and(variables.next_move_var())?
            .diff(edge_variables.next_move_var())?
            .and(&iff_register_condition)?
            .and(edge_priority_zero)?;

        tracing::debug!("Starting E_i construction");

        let mut e_i_edges = vec![base_false.clone(); num_registers];
        let mut base_registers_encoder: MultiEncoder<Priority, F, CachedBinaryEncoder<Priority, F>> = MultiEncoder::new(
            &manager,
            &variables.register_vars().iter().cloned().chunks(register_bits_needed),
        );
        let mut next_encoder: MultiEncoder<Priority, _, _> = MultiEncoder::new(
            &manager,
            &edge_variables
                .register_vars()
                .iter()
                .cloned()
                .chunks(register_bits_needed),
        );

        // Calculate all possible next register sets based on a current set of registers and priorities.
        // This will be vastly more efficient than constructing the full register game following the explicit naive algorithm.
        // However, should a game with `n` vertices and `n` priorities be used, this approach will likely take... forever...
        let n_unique_priorities = sg.priorities.keys().len();
        let loop_count: usize = (0..=k).map(|i| n_unique_priorities * n_unique_priorities.pow((k - i) as u32)).sum();
        let mut logger = ProgressLogger::new(loop_count / 2);

        // We need to handle three cases to efficiently encode the whole permutation set:
        // 1. Efficient case where the contents in register `r_i` <= `priority`
        //    - We can exploit the fact that we know that the `next_register_state` and the priority will be the exact same for all such cases.
        // 2. Case where the contents in `i` > `priority` and the contents are even
        // 3. Case where the contents in `i` > `priority` and the contents are odd
        let mut small_ei_encode =
            |i: usize,
             priority: Priority,
             prio_bdd: &F,
             reg_vars: &[F],
             register_state: &[Priority],
             next_registers_state: &mut [Priority],
             permutation_encoding: F,
             permutation: Option<Vec<Priority>>,
             base_register_encoder: &mut MultiEncoder<Priority, F, CachedBinaryEncoder<Priority, F>>| {
                let base_starting_set = prio_bdd.and(&permutation_encoding)?;
                // We know that the resulting register states will _always_ be the same, no matter which of the cases above
                // that we have. So we can simply reset once for the current priority.
                explicit::register_game::next_registers_2021_out(
                    register_state,
                    next_registers_state,
                    priority,
                    num_registers,
                    i,
                );

                // ** First case **
                // First calculate how many bits are significant
                let significant_bits = (Priority::BITS - priority.leading_zeros()) as usize;
                // We can exclude all register states which are binary wise higher than our current `priority`
                // We still need to exclude special cases such as 3, when priority = 2.
                let exclude_bits = reg_vars[significant_bits..register_bits_needed]
                    .iter()
                    .try_fold(base_true.clone(), |acc, var| acc.diff(var))?;
                // We'll need to exclude values larger than `priority` up to the next power of two to ensure they don't get included
                let final_exclude = sg
                    .priorities
                    .keys()
                    .filter(|&&p| p > priority && p < (priority + 1).next_power_of_two())
                    .try_fold(exclude_bits, |acc, p| {
                        acc.diff(base_register_encoder.encode_single(i, *p).unwrap())
                    })?;
                // The general register game priority will always be the same for values <= priority.
                let rg_prio_general =
                    explicit::register_game::reset_to_priority_2021(i as Rank, priority, priority, controller);

                let starting_vertices = base_starting_set.and(&final_exclude)?;
                let next_registers = next_encoder.encode_many(next_registers_state)?;
                let next_with_priority = next_registers.and(prio_edge_encoder.encode(rg_prio_general)?)?;
                let all_vertices = starting_vertices.and(&next_with_priority)?;

                // tracing::debug!("Symbolic Reset of `r_{} <= {:?}`, prio: `{}`, from: `[X, {:?}]`, result: `{:?}`, rg_prio: `{}`", i, priority,  priority, permutation, next_registers_state, rg_prio_general);
                let e_i = &mut e_i_edges[i];
                *e_i = e_i.or(&all_vertices)?;

                // ** Cases 2. and 3.
                // We can simply invert the `final_exclude` to go from $r_i <= priority$ to $r_i > priority$
                let inverse = final_exclude.not_owned()?;
                let even_greater = inverse.and(&reg_vars[0].not()?)?;
                let odd_greater = inverse.and(&reg_vars[0])?;
                
                let rg_odd_priority = explicit::register_game::reset_to_priority_2021(i as Rank, 1, 0, controller);
                let rg_ev_priority = explicit::register_game::reset_to_priority_2021(i as Rank, 2, 0, controller);
                
                let ((rg_gt_eq_p, rg_gt_eq), (rg_gt_neq_p, rg_gt_neq)) = if rg_odd_priority == rg_prio_general {
                    ((rg_odd_priority, odd_greater), (rg_ev_priority, even_greater))
                } else {
                    ((rg_ev_priority, even_greater), (rg_odd_priority, odd_greater))
                };

                // We can re-use `next_with_priority` for `rq_gt_eq_p`
                let starting_vertices = base_starting_set.and(&rg_gt_eq)?;
                let all_vertices = starting_vertices.and(&next_with_priority)?;
                *e_i = e_i.or(&all_vertices)?;
                
                // Re-calculate the `next_with_priority` set for `rq_gt_neq_p`
                let next_with_priority = next_registers.and(prio_edge_encoder.encode(rg_gt_neq_p)?)?;
                let starting_vertices = base_starting_set.and(&rg_gt_neq)?;
                let all_vertices = starting_vertices.and(&next_with_priority)?;
                *e_i = e_i.or(&all_vertices)?;

                Ok::<_, BddError>(())
            };

        for i in 0..=k {
            let n_remaining_registers = k - i;
            let mut next_register_state = vec![0; num_registers];
            let reg_vars = variables.register_vars_i(i, register_bits_needed);

            // Simple case distinction:
            // 1. We have a permutation [.., r_i, .., r_k], where for all i < j <= k => r_j <= priority
            //    In this case we can merge them all into a single call, significantly reducing the number of permutations
            // 2. We have a  [.., r_i, .., r_k], where for some i < j <= k => r_j > priority
            //    For these permutations we need to manually add them all individually.
            for (&priority, prio_bdd) in sg.priorities.iter().sorted_by_key(|k| *k.0) {
                let mut register_state = vec![0; num_registers];
                // ** Handle case 1 **
                // First calculate how many bits are significant
                let significant_bits = (Priority::BITS - priority.leading_zeros()) as usize;
                // Exclude the states which would be greater than our allowed priority
                let mut permutation_exclude = base_true.clone();
                for j in (i + 1)..=k {
                    let reg_vars_j = variables.register_vars_i(j, register_bits_needed);
                    // We can exclude all register states which are binary wise higher than our current `priority`
                    // We still need to exclude special cases such as 3, when priority = 2.
                    let exclude_bits = reg_vars_j[significant_bits..register_bits_needed]
                        .iter()
                        .try_fold(base_true.clone(), |acc, var| acc.diff(var))?;
                    // We'll need to exclude values larger than priority up to the next power of two to ensure they don't get included
                    let remaining_exclude = sg
                        .priorities
                        .keys()
                        .filter(|&&p| p > priority && p < (priority + 1).next_power_of_two())
                        .try_fold(exclude_bits, |acc, p| {
                            acc.diff(base_registers_encoder.encode_single(j, *p).unwrap())
                        })?;

                    permutation_exclude = permutation_exclude.and(&remaining_exclude)?;
                }

                small_ei_encode(
                    i,
                    priority,
                    prio_bdd,
                    reg_vars,
                    &register_state,
                    &mut next_register_state,
                    permutation_exclude,
                    None,
                    &mut base_registers_encoder,
                )?;

                // ** Handle case 2 **
                let remaining_permutations = itertools::repeat_n(
                    sg.priorities.keys().copied().filter(|&p| p > priority),
                    n_remaining_registers,
                ).multi_cartesian_product();

                for partial_permutation in remaining_permutations {
                    logger.tick();
                    // For large (e.g., two_counters_14) cases intermediate GCs are required.
                    if logger.ticks > 0 && logger.ticks % 100_000 == 0 {
                        manager.with_manager_exclusive(|man| {
                            tracing::debug!(nodes = man.num_inner_nodes(), "Running GC");
                            man.gc();
                            tracing::debug!(nodes = man.num_inner_nodes(), "Post GC node count");
                        });
                    }

                    // Prepare our scratch register set for encodes.
                    register_state[i + 1..].copy_from_slice(&partial_permutation);
                    // Encode the register r_j, where j > i. These will always remain the same
                    let permutation_encoding = base_registers_encoder
                        .encode_many_range(i + 1..k + 1, &partial_permutation)
                        .unwrap_or_else(|_| base_true.clone());

                    small_ei_encode(
                        i,
                        priority,
                        prio_bdd,
                        reg_vars,
                        &register_state,
                        &mut next_register_state,
                        permutation_encoding,
                        Some(partial_permutation),
                        &mut base_registers_encoder,
                    )?;
                }
            }
        }

        // Add the additional conditions for each E_i relation.
        // From t=0 -> t=1
        let base_edge = edge_variables.next_move_var().diff(variables.next_move_var())?;
        // Any E_i transition will have the same underlying vertex, so we should assert it doesn't change.
        let base_vertex = variables
            .vertex_vars()
            .iter()
            .zip(edge_variables.vertex_vars())
            .try_fold(base_edge.clone(), |acc, (v, v_next)| acc.and(&v.equiv(v_next)?))?;

        e_i_edges = e_i_edges.into_iter().flat_map(|e_i| e_i.and(&base_vertex)).collect();
        let e_i_joined = e_i_edges.iter().try_fold(base_false.clone(), |acc, bdd| acc.or(bdd))?;

        let conj_v = variables.conjugated(&base_true)?;
        let conj_e = edge_variables.conjugated(&base_true)?;

        tracing::debug!("Creating priority BDDs");

        let extra_priorities = if controller == Owner::Even { 1 } else { 2 };
        let mut priorities: ahash::HashMap<_, _> = (0..=2 * k + extra_priorities)
            .map(|val| (val as Priority, base_false.clone()))
            .collect();
        let mut prio_encoder: CachedBinaryEncoder<Priority, F> =
            CachedBinaryEncoder::new(&manager, variables.priority_vars().iter().cloned().collect());

        // All E_move edges get a priority of `0` by default.
        let priority_zero = prio_encoder.encode(0)?;
        let _ = priorities.entry(0).and_modify(|pri| {
            *pri = sg
                .vertices
                .and(priority_zero)
                .unwrap()
                .and(&variables.next_move_var().not().unwrap())
                .unwrap()
        });

        // The E_i set contains our encoded priorities, we can create a quick look up set by substituting
        for (rg_priority, bdd) in priorities.iter_mut() {
            let prio_edge_encoding = prio_edge_encoder.encode(*rg_priority)?;

            let prio_vertices = e_i_joined.and(prio_edge_encoding)?.exist(&conj_v)?;
            let subs = Subst::new(&edge_variables.all_variables, &variables.all_variables);
            let reverse_substitute = prio_vertices.substitute(subs)?;

            *bdd = bdd.or(&reverse_substitute)?;
        }

        manager.with_manager_exclusive(|man| man.set_threading_enabled(true));

        Ok(Self {
            k: k as Rank,
            controller,
            manager,
            variables,
            variables_edges: edge_variables,
            conjugated_variables: conj_v,
            conjugated_v_edges: conj_e,
            vertices: s_even.or(&s_odd)?,
            v_even: s_even,
            v_odd: s_odd,
            e_move,
            e_i_all: e_i_joined,
            e_i: e_i_edges,
            priorities,
            base_true,
            base_false,
        })
    }

    pub fn to_symbolic_parity_game(&self) -> symbolic::Result<SymbolicParityGame<F>> {
        Ok(SymbolicParityGame {
            pg_vertex_count: self.vertices.sat_quick_count(self.variables.all_variables.len() as u32) as usize,
            manager: self.manager.clone(),
            variables: self.variables.all_variables.clone(),
            variables_edges: self.variables_edges.all_variables.clone(),
            conjugated_variables: self.conjugated_variables.clone(),
            conjugated_v_edges: self.conjugated_v_edges.clone(),
            vertices: self.vertices.clone(),
            vertices_even: self.v_even.clone(),
            vertices_odd: self.v_odd.clone(),
            priorities: self.priorities.clone(),
            edges: self.edges()?,
            base_true: self.base_true.clone(),
            base_false: self.base_false.clone(),
        })
    }

    #[cfg(feature = "statistics")]
    pub fn print_statistics<O: Copy>(&self)
    where
        for<'id> F::Manager<'id>: HasApplyCache<F::Manager<'id>, O, ApplyCache: StatisticsGenerator>,
    {
        self.manager.with_manager_shared(|man| {
            use oxidd_cache::StatisticsGenerator;
            use oxidd_core::HasApplyCache;
            oxidd::bdd::print_stats();
            oxidd::bcdd::print_stats();
            man.apply_cache().print_stats()
        });
    }

    /// Project the winning regions within the register game to the underlying symbolic parity game.
    ///
    /// The returned BDDs will reference _only_ the variables corresponding to the binary encoding of the original parity game's vertices.
    pub fn projected_winning_regions(&self, w_even: &F, w_odd: &F) -> symbolic::Result<(F, F)> {
        // The principle is simple:
        // Every vertex with zeroed registers, zero priority, and next move `reset`, is implicitly a 'starting' vertice in the register game.
        // Only one such vertex would exist for every vertex in the underlying parity game, so we can safely use their result
        // for the desired effect.
        let zero_registers = self
            .variables
            .register_vars()
            .iter()
            .try_fold(self.base_true.clone(), |acc, next| acc.and(&next.not()?))?;
        let zero_prio = CachedBinaryEncoder::encode_impl(self.variables.priority_vars(), 0u32)?;

        let projector_func = zero_registers
            .and(&zero_prio)?
            .and(&self.variables.next_move_var().not()?)?;

        Ok((w_even.and(&projector_func)?, w_odd.and(&projector_func)?))
    }

    /// Project the winning regions within the register game to the underlying vertices of the original parity game.
    pub fn project_winning_regions(&self, w_even: &F, w_odd: &F) -> symbolic::Result<(Vec<u32>, Vec<u32>)> {
        let (wp_even, wp_odd) = self.projected_winning_regions(w_even, w_odd)?;

        let result = self.manager.with_manager_shared(|man| {
            let (vals_even, vals_odd) = (wp_even.sat_assignments(man), wp_odd.sat_assignments(man));
            let variable_indices = [self.variables.var_indices(RegisterLayers::Vertex).as_slice()];

            (
                symbolic::sat::decode_split_assignments(vals_even, &variable_indices)
                    .pop()
                    .expect("No valid window was provided"),
                symbolic::sat::decode_split_assignments(vals_odd, &variable_indices)
                    .pop()
                    .expect("No valid window was provided"),
            )
        });

        Ok(result)
    }

    /// Return the number of registers encoded in this symbolic register game.
    fn n_registers(&self) -> usize {
        (self.k + 1) as usize
    }
}

#[derive(Clone)]
pub struct RegisterVertexVars<F: Function> {
    pub all_variables: EcoVec<F>,
    pub manager_layer_indices: ahash::HashMap<RegisterLayers, EcoVec<usize>>,
    n_register_vars: usize,
    n_priority_vars: usize,
}

impl<F: BooleanFunctionExtensions> Debug for RegisterVertexVars<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegisterVertexVars")
            .field("n_register_vars", &self.n_register_vars)
            .field("n_priority_vars", &self.n_priority_vars)
            .field("manager_layer_indices", &self.manager_layer_indices)
            .finish()
    }
}

#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Hash, Eq)]
pub enum RegisterLayers {
    /// The variable related to the next move of each register game vertex.
    NextMove,
    /// The variables related to the registers of the register game, containing the priorities of the underlying parity game.
    Registers,
    /// The variables related to the original vertex ids of the underlying parity game
    Vertex,
    /// The variables related to the register game priority of each register game vertex.
    Priority,
}

impl<F: BooleanFunctionExtensions> RegisterVertexVars<F> {
    pub fn new(
        next_move: FunctionVarRef<F>,
        register_vars: impl ExactSizeIterator<Item = FunctionVarRef<F>>,
        priority_vars: impl ExactSizeIterator<Item = FunctionVarRef<F>>,
        vertex_vars: impl Iterator<Item = FunctionVarRef<F>>,
    ) -> Self {
        let mut manager_indices =
            ahash::HashMap::from_iter([(RegisterLayers::NextMove, eco_vec![next_move.idx as usize])]);
        let mut all_variables = eco_vec![next_move.func];

        let n_register_vars = register_vars.len();
        let n_priority_vars = priority_vars.len();

        macro_rules! distribute {
            ($($itr:expr => $key:expr),*) => {
                $(
                    let entry = manager_indices.entry($key).or_insert_with(EcoVec::new);
                    for var_ref in $itr {
                        entry.push(var_ref.idx as usize);
                        all_variables.push(var_ref.func);
                    }
                )*
            };
        }

        distribute!(
            register_vars => RegisterLayers::Registers,
            vertex_vars => RegisterLayers::Vertex,
            priority_vars => RegisterLayers::Priority
        );

        Self {
            n_register_vars,
            n_priority_vars,
            all_variables,
            manager_layer_indices: manager_indices,
        }
    }

    #[inline(always)]
    pub fn next_move_var(&self) -> &F {
        &self.all_variables[0]
    }

    #[inline(always)]
    pub fn register_vars(&self) -> &[F] {
        &self.all_variables[1..self.n_register_vars + 1]
    }

    #[inline(always)]
    pub fn register_vars_i(&self, i: usize, num_register_bits: usize) -> &[F] {
        let start = i * num_register_bits;
        &self.register_vars()[start..start + num_register_bits]
    }

    #[inline(always)]
    pub fn vertex_vars(&self) -> &[F] {
        let start = self.n_register_vars + 1;
        let end = self.all_variables.len() - self.n_priority_vars;
        &self.all_variables[start..end]
    }

    #[inline(always)]
    pub fn priority_vars(&self) -> &[F] {
        &self.all_variables[self.all_variables.len() - self.n_priority_vars..]
    }

    pub fn var_indices(&self, group: RegisterLayers) -> &EcoVec<usize> {
        self.manager_layer_indices.get(&group).expect("Impossible")
    }

    fn conjugated(&self, base_true: &F) -> AllocResult<F> {
        self.all_variables
            .iter()
            .try_fold(base_true.clone(), |acc: F, next: &F| acc.and(next))
    }

    pub fn iter_names<'a>(&'a self, suffix: &'a str) -> impl Iterator<Item = (&F, String)> + 'a {
        [(self.next_move_var(), format!("t{suffix}"))]
            .into_iter()
            .chain(
                self.register_vars()
                    .iter()
                    .enumerate()
                    .map(move |(i, r)| (r, format!("r{i}{suffix}"))),
            )
            .chain(
                self.vertex_vars()
                    .iter()
                    .enumerate()
                    .map(move |(i, r)| (r, format!("x{i}{suffix}"))),
            )
            .chain(
                self.priority_vars()
                    .iter()
                    .enumerate()
                    .map(move |(i, r)| (r, format!("p{i}{suffix}"))),
            )
    }
}

struct ProgressLogger {
    total_len: usize,
    ticks: usize,
    start: std::time::Instant,
    now: std::time::Instant,
}

impl ProgressLogger {
    fn new(loop_count: usize) -> Self {
        Self {
            total_len: loop_count,
            ticks: 0,
            start: std::time::Instant::now(),
            now: std::time::Instant::now(),
        }
    }

    fn tick(&mut self) {
        self.ticks += 1;
        if self.now.elapsed() > Duration::from_secs(3) {
            let explore_rate = self.ticks as f64 / self.start.elapsed().as_secs_f64();
            let eta = (self.total_len - self.ticks) as f64 / explore_rate;
            tracing::debug!(
                "Current permutation: `{}/{}`, rate: `{:.2}/s`, ETA: `{:.2}`s",
                self.ticks,
                self.total_len,
                explore_rate,
                eta
            );
            self.now = std::time::Instant::now();
        }
    }
}

fn check_rg_invariants(pg: &ParityGame) -> symbolic::Result<()> {
    let priorities = pg.priorities_unique().collect_vec();

    // if priorities.len() < 2 {
    //     return Err(symbolic::BddError::InvariantViolated("A register game requires at least two distinct priorities".to_string()));
    // }
    // if !priorities.iter().any(|p| p % 2 == 0) || !priorities.iter().any(|p| p % 2 == 1) {
    //     return Err(symbolic::BddError::InvariantViolated("A register game requires at least one even and one odd priority".to_string()));
    // }

    Ok(())
}
