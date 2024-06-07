use std::{collections::HashMap, marker::PhantomData};

use ecow::{eco_vec, EcoVec};
use itertools::Itertools;
use oxidd::bdd::BDDManagerRef;
use oxidd_core::{
    function::{BooleanFunction, BooleanFunctionQuant, Function, FunctionSubst},
    Manager,
    ManagerRef, util::{AllocResult, Subst},
};
use petgraph::prelude::EdgeRef;

use crate::{
    explicit,
    explicit::{ParityGame, ParityGraph, register_game::Rank},
    Owner,
    Priority,
    symbolic, symbolic::{
        BDD,
        helpers::{CachedSymbolicEncoder, MultiEncoder},
        oxidd_extensions::{BddExtensions, BooleanFunctionExtensions, FunctionManagerExtension, FunctionVarRef}, SymbolicParityGame,
    },
};

mod test_helpers;

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
    pub edges: F,
    pub e_move: F,
    pub e_i: Vec<F>,
    pub priorities: ahash::HashMap<Priority, F>,

    pub base_true: F,
    pub base_false: F,
}

type F = BDD;
impl SymbolicRegisterGame<BDD>
// where
// F: Function + BooleanFunctionExtensions + BooleanFunctionQuant + FunctionManagerExtension + FunctionSubst,
{
    /// Construct a new symbolic register game straight from an explicit parity game.
    ///
    /// See [crate::register_game::RegisterGame::construct]
    #[tracing::instrument(name = "Build Symbolic Register Game", skip_all)]
    pub fn from_symbolic(explicit: &ParityGame, k: Rank, controller: Owner) -> symbolic::Result<Self> {
        let k = k as usize;
        let num_registers = k + 1;

        let extra_priorities = if controller == Owner::Even { 1 } else { 2 };
        let max_reg_priority = 2 * k + extra_priorities;
        let register_bits_needed = (explicit.priority_max() as f64 + 1.).log2().ceil() as usize;
        // Calculate the amount of variables we'll need for the binary encodings of the registers and vertices.
        let n_register_vars = register_bits_needed * num_registers;
        let n_variables = (explicit.vertex_count() as f64).log2().ceil() as usize;
        let n_prio_vars = (max_reg_priority as f64 + 1.).log2().ceil() as usize;

        let manager = F::new_manager(explicit.vertex_count(), explicit.vertex_count(), 12);

        // Construct base building blocks for the BDD
        let base_true = manager.with_manager_exclusive(|man| F::t(man));
        let base_false = manager.with_manager_exclusive(|man| F::f(man));

        let (variables, edge_variables) = manager.with_manager_exclusive(|man| {
            let next_move = F::new_var_layer_idx(man)?;
            let next_move_edge = F::new_var_layer_idx(man)?;

            let registers = (0..n_register_vars)
                .flat_map(|_| F::new_var_layer_idx(man))
                .collect_vec();
            let registers_edge = (0..n_register_vars)
                .flat_map(|_| F::new_var_layer_idx(man))
                .collect_vec();

            let vertex_vars = (0..n_variables).flat_map(|_| F::new_var_layer_idx(man)).collect_vec();
            let vertex_vars_edge = (0..n_variables).flat_map(|_| F::new_var_layer_idx(man)).collect_vec();

            let prio_vars = (0..n_prio_vars).flat_map(|_| F::new_var_layer_idx(man)).collect_vec();
            let prio_vars_edge = (0..n_prio_vars).flat_map(|_| F::new_var_layer_idx(man)).collect_vec();

            Ok::<_, symbolic::BddError>((
                RegisterVertexVars::new(
                    next_move,
                    registers.into_iter(),
                    prio_vars.into_iter(),
                    vertex_vars.into_iter(),
                ),
                RegisterVertexVars::new(
                    next_move_edge,
                    registers_edge.into_iter(),
                    prio_vars_edge.into_iter(),
                    vertex_vars_edge.into_iter(),
                ),
            ))
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
        let extr_priorities = if controller == Owner::Even { 1 } else { 2 };
        let mut priorities: ahash::HashMap<_, _> = (0..=2 * k + extr_priorities)
            .into_iter()
            .map(|val| (val as Priority, base_false.clone()))
            .collect();

        tracing::debug!("Starting E_move construction");

        // Ensure all registers remain the same past the E_move transition
        let iff_register_condition = variables
            .register_vars()
            .iter()
            .zip(edge_variables.register_vars())
            .fold(base_true.clone(), |acc, (r, r_next)| {
                acc.and(&r.equiv(r_next).unwrap()).unwrap()
            });

        // t = 1 && t' = 0 && (r0 <-> r0')
        let mut e_move = sg
            .edges
            .and(variables.next_move_var())?
            .diff(edge_variables.next_move_var())?
            .and(&iff_register_condition)?;

        // All E_move edges get a priority of `0` by default.
        let _ = priorities
            .entry(0)
            .and_modify(|pri| *pri = variables.next_move_var().not().unwrap());

        tracing::debug!("Starting E_i construction");

        let mut e_i_edges = vec![base_false.clone(); num_registers];
        let mut perm_encoder: MultiEncoder<Priority, _> = MultiEncoder::new(
            &manager,
            &variables
                .register_vars()
                .into_iter()
                .cloned()
                .chunks(register_bits_needed),
        );
        let mut next_encoder: MultiEncoder<Priority, _> = MultiEncoder::new(
            &manager,
            &edge_variables
                .register_vars()
                .into_iter()
                .cloned()
                .chunks(register_bits_needed),
        );

        let mut prio_encoder =
            CachedSymbolicEncoder::new(&manager, variables.priority_vars().into_iter().cloned().collect());
        let mut prio_edge_encoder =
            CachedSymbolicEncoder::new(&manager, variables.priority_vars().into_iter().cloned().collect());

        // Calculate all possible next register sets based on a current set of registers and priorities.
        // This will be vastly more efficient than constructing the full register game following the explicit naive algorithm.
        // However, should a game with `n` vertices and `n` priorities be used, this approach will likely take... forever...
        let unique_register_contents =
            itertools::repeat_n(sg.priorities.keys().copied(), num_registers).multi_cartesian_product();

        for (i, permutation) in unique_register_contents.enumerate() {
            tracing::trace!("Starting E_i permutation number: `{i}`");
            let permutation_encoding = perm_encoder.encode_many(&permutation)?;

            for (&priority, prio_bdd) in &sg.priorities {
                let starting_vertices = prio_bdd.and(&permutation_encoding)?;
                // Calculate the `e_i` reset
                for i in 0..=k {
                    let mut next_registers = permutation.clone();
                    let rg_priority = explicit::register_game::reset_to_priority_2021(
                        i as Rank,
                        permutation[i],
                        priority,
                        controller,
                    );
                    explicit::register_game::next_registers_2021(&mut next_registers, priority, num_registers, i);

                    let next_encoding = next_encoder.encode_many(&next_registers)?;
                    let next_with_priority = next_encoding.and(prio_edge_encoder.encode(rg_priority)?)?;
                    let all_vertices = starting_vertices.and(&next_with_priority)?;

                    let e_i = &mut e_i_edges[i];
                    *e_i = e_i.or(&all_vertices)?;

                    // Update the priority set as well
                    let next_base_encoding = perm_encoder.encode_many(&next_registers)?;
                    let next_base_with_priority = next_base_encoding.and(prio_encoder.encode(rg_priority)?)?;
                    let vertices_with_priority =
                        prio_bdd.and(&next_base_with_priority)?.and(variables.next_move_var())?;
                    priorities
                        .entry(rg_priority)
                        .and_modify(|bdd| *bdd = bdd.or(&vertices_with_priority).unwrap());

                    tracing::debug!(
                        "Debug: {:?}/{:?} - Prio: {} - i: {} - next_prio: {}",
                        permutation,
                        next_registers,
                        priority,
                        i,
                        rg_priority
                    );
                }
            }
        }

        // Add the additional conditions for each E_i relation.
        // From t=0 -> t=1
        let base_edge = edge_variables.next_move_var().diff(&variables.next_move_var())?;
        // Any E_i transition will have the same underlying vertex, so we should assert it doesn't change.
        let base_vertex = variables
            .vertex_vars()
            .iter()
            .zip(edge_variables.vertex_vars())
            .fold(base_edge.clone(), |acc, (v, v_next)| {
                acc.and(&v.equiv(v_next).unwrap()).unwrap()
            });

        e_i_edges = e_i_edges.into_iter().flat_map(|e_i| e_i.and(&base_vertex)).collect();

        let conj_v = variables.conjugated(&base_true)?;
        let conj_e = edge_variables.conjugated(&base_true)?;

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
            edges: e_move.or(&e_i_edges
                .iter()
                .fold(base_false.clone(), |acc, bdd| acc.or(bdd).unwrap()))?,
            e_move,
            e_i: e_i_edges,
            priorities,
            base_true,
            base_false,
        })
    }

    pub fn to_symbolic_parity_game(&self) -> SymbolicParityGame {
        SymbolicParityGame {
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
            edges: self
                .e_move
                .or(&self
                    .e_i
                    .iter()
                    .fold(self.base_false.clone(), |acc, bdd| acc.or(bdd).unwrap()))
                .unwrap(),
            base_true: self.base_true.clone(),
            base_false: self.base_false.clone(),
        }
    }

    pub fn gc(&self) -> usize {
        self.manager.with_manager_exclusive(|m| m.gc())
    }

    pub fn bdd_node_count(&self) -> usize {
        self.manager.with_manager_exclusive(|m| m.num_inner_nodes())
    }

    #[inline]
    fn edge_substitute(&self, bdd: &F) -> symbolic::Result<F> {
        let subs = Subst::new(&self.variables.all_variables, &self.variables_edges.all_variables);
        Ok(bdd.substitute(subs)?)
    }

    #[inline]
    fn rev_edge_substitute(&self, bdd: &F) -> symbolic::Result<F> {
        let subs = Subst::new(&self.variables_edges.all_variables, &self.variables.all_variables);
        Ok(bdd.substitute(subs)?)
    }
}

pub struct RegisterVertexVars<F: Function> {
    pub all_variables: EcoVec<F>,
    pub manager_layer_indices: ahash::HashMap<RegisterLayers, EcoVec<usize>>,
    n_register_vars: usize,
    n_priority_vars: usize,
}

#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Hash, Eq)]
pub enum RegisterLayers {
    NextMove,
    Registers,
    Vertex,
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
                    let mut entry = manager_indices.entry($key).or_insert_with(EcoVec::new);
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

        println!("Priority Indices: {:?}", manager_indices);

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
        Ok(self
            .all_variables
            .iter()
            .fold(base_true.clone(), |acc: F, next: &F| acc.and(next).unwrap()))
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
