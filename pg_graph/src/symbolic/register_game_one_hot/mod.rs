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
    ManagerRef, util::{AllocResult, OptBool}, WorkerManager,
};

use variable_order::VariableAllocatorInfo;

use crate::{
    explicit,
    explicit::{ParityGame, ParityGraph, register_game::Rank},
    Owner,
    Priority,
    symbolic, symbolic::{
        helpers::{CachedBinaryEncoder, MultiEncoder},
        oxidd_extensions::{BooleanFunctionExtensions, FunctionVarRef, GeneralBooleanFunction},
        sat::TruthAssignmentsIterator,
        SymbolicParityGame,
    },
};
use crate::symbolic::helpers::SymbolicEncoder;
use crate::symbolic::register_game_one_hot::helpers::OneHotEncoder;

pub mod helpers;
pub(crate) mod test_helpers;
mod variable_order;

pub struct OneHotRegisterGame<F: Function> {
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

impl<F> OneHotRegisterGame<F>
where
    F: GeneralBooleanFunction,
    for<'id> F::Manager<'id>: WorkerManager,
    for<'a, 'b> TruthAssignmentsIterator<'b, 'a, F>: Iterator<Item = Vec<OptBool>>,
{
    /// Construct a new symbolic register game straight from an explicit parity game.
    ///
    /// See [crate::register_game::RegisterGame::construct]
    #[tracing::instrument(name = "Build Symbolic One Hot Register Game", skip_all)]
    pub fn from_symbolic(explicit: &ParityGame, k: Rank, controller: Owner) -> symbolic::Result<Self> {
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
        // We use a one-hot encoding scheme, so we simply need all unique priorities.
        let unique_priorities = explicit.priorities_unique().sorted().collect_vec();
        let register_bits_needed = unique_priorities.len();
        // Calculate the amount of variables we'll need for the one-hot encoding of the registers and binary encoding of the vertices.
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

        tracing::debug!("Creating priority BDDs");
        let extra_priorities = if controller == Owner::Even { 1 } else { 2 };
        let mut priorities: ahash::HashMap<_, _> = (0..=2 * k + extra_priorities)
            .map(|val| (val as Priority, base_false.clone()))
            .collect();
        let mut prio_encoder =
            CachedBinaryEncoder::new(&manager, variables.priority_vars().iter().cloned().collect());
        let mut prio_edge_encoder =
            CachedBinaryEncoder::new(&manager, edge_variables.priority_vars().iter().cloned().collect());

        tracing::debug!("Starting E_move construction");

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

        tracing::debug!("Starting E_i construction");

        let mut e_i_edges = vec![base_false.clone(); num_registers];
        let mut base_permutation_encoder: MultiEncoder<Priority, _, _> = MultiEncoder::new_collection(
            (&variables.register_vars().iter().cloned().chunks(register_bits_needed))
                .into_iter()
                .flat_map(|reg_vars| OneHotEncoder::new(unique_priorities.iter().copied(), reg_vars.collect())),
        );
        let mut next_permutation_encoder: MultiEncoder<Priority, _,_> = MultiEncoder::new_collection(
            (&edge_variables.register_vars().iter().cloned().chunks(register_bits_needed))
                .into_iter()
                .flat_map(|reg_vars| OneHotEncoder::new(unique_priorities.iter().copied(), reg_vars.collect())),
        );

        // Calculate all possible next register sets based on a current set of registers and priorities.
        // This will be vastly more efficient than constructing the full register game following the explicit naive algorithm.
        // However, should a game with `n` vertices and `n` priorities be used, this approach will likely take... forever...
        let n_unique_priorities = sg.priorities.keys().len();
        let unique_register_contents =
            itertools::repeat_n(sg.priorities.keys().copied(), num_registers).multi_cartesian_product();
        let mut logger = ProgressLogger::new(&unique_register_contents);

        for (n_th_permutation, permutation) in unique_register_contents.enumerate() {
            logger.tick(n_th_permutation);
            let permutation_encoding = base_permutation_encoder.encode_many_partial_rev(&permutation)?;

            // Calculate the `e_i` reset
            for i in 0..=k {
                // As we iterate permutations backwards to forwards (aka, [0, 0, _], where `_` will have all values changed before [0, _, 0])
                // we can stop iteration once we pass the point where we _know_ that the valuable BDDs have been constructed for the state-space.
                let multiplier = (k - i) + 1;
                let skip_threshold = n_unique_priorities.pow(multiplier as u32);
                if n_th_permutation >= skip_threshold {
                    break;
                }

                let index_to_use = k - i;
                let used_permutation = &permutation_encoding[index_to_use];

                for (&priority, prio_bdd) in &sg.priorities {
                    let starting_vertices = prio_bdd.and(used_permutation)?;

                    let mut next_registers = permutation.clone();
                    let rg_priority = explicit::register_game::reset_to_priority_2021(
                        i as Rank,
                        permutation[i],
                        priority,
                        controller,
                    );
                    explicit::register_game::next_registers_2021(&mut next_registers, priority, num_registers, i);

                    let next_encoding = next_permutation_encoder.encode_many(&next_registers)?;
                    let next_with_priority = next_encoding.and(prio_edge_encoder.encode(rg_priority)?)?;
                    let all_vertices = starting_vertices.and(&next_with_priority)?;

                    let e_i = &mut e_i_edges[i];
                    *e_i = e_i.or(&all_vertices)?;

                    // Update the priority set as well
                    let next_base_encoding = base_permutation_encoder.encode_many(&next_registers)?;
                    let next_base_with_priority = next_base_encoding.and(prio_encoder.encode(rg_priority)?)?;
                    let vertices_with_priority =
                        prio_bdd.and(&next_base_with_priority)?.and(variables.next_move_var())?;
                    priorities
                        .entry(rg_priority)
                        .and_modify(|bdd| *bdd = bdd.or(&vertices_with_priority).unwrap());
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
        let zero_registers = self.variables.chunked_register_vars(self.n_registers())
            .flat_map(|reg_vars| {
                let first = reg_vars[0].clone();
                reg_vars.iter().skip(1).flat_map(|f| f.not()).try_fold(first, |acc, next| acc.and(&next))
            })
            .try_fold(self.base_true.clone(), |acc, next| acc.and(&next))?;
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
            let (vals_even, vals_odd) = (vals_even.collect_vec(), vals_odd.collect_vec());
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

#[derive(Clone)]
pub struct RegisterVertexVars<F: Function> {
    pub all_variables: EcoVec<F>,
    pub manager_layer_indices: ahash::HashMap<RegisterLayers, EcoVec<usize>>,
    n_register_vars: usize,
    n_priority_vars: usize,
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

    pub fn chunked_register_vars(&self, chunks: usize) -> impl Iterator<Item=&[F]> {
        let registers = self.register_vars();
        let chunk_size = registers.len() / chunks;

        (0..chunks)
            .map(move |i| {
            let start = chunk_size * i;
            &registers[start..start + chunk_size]
        })
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

impl<F: BooleanFunctionExtensions> Debug for RegisterVertexVars<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegisterVertexVars")
            .field("n_register_vars", &self.n_register_vars)
            .field("n_priority_vars", &self.n_priority_vars)
            .field("manager_layer_indices", &self.manager_layer_indices)
            .finish()
    }
}

struct ProgressLogger {
    total_len: usize,
    start: std::time::Instant,
    now: std::time::Instant,
}

impl ProgressLogger {
    fn new<T>(permutations: &impl Iterator<Item = T>) -> Self {
        let total_len = permutations.size_hint().0;
        Self {
            total_len,
            start: std::time::Instant::now(),
            now: std::time::Instant::now(),
        }
    }

    fn tick(&mut self, progress: usize) {
        if self.now.elapsed() > Duration::from_secs(3) {
            let explore_rate = progress as f64 / self.start.elapsed().as_secs_f64();
            let eta = (self.total_len - progress) as f64 / explore_rate;
            tracing::debug!(
                "Current permutation: `{}/{}`, rate: `{:.2}/s`, ETA: `{:.2}`s",
                progress,
                self.total_len,
                explore_rate,
                eta
            );
            self.now = std::time::Instant::now();
        }
    }
}
