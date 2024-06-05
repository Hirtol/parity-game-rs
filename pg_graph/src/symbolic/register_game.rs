use std::{collections::HashMap, marker::PhantomData};

use ecow::EcoVec;
use itertools::Itertools;
use oxidd::bdd::BDDManagerRef;
use oxidd_core::{
    function::{BooleanFunction, BooleanFunctionQuant, Function, FunctionSubst},
    ManagerRef,
    util::{AllocResult, Subst},
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
        oxidd_extensions::{BooleanFunctionExtensions, FunctionManagerExtension}, SymbolicParityGame,
    },
};

pub struct SymbolicRegisterGame<F: Function> {
    pub manager: F::ManagerRef,
    pub variables: RegisterVertexVars<F>,
    pub variables_edges: RegisterVertexVars<F>,
    pub conjugated_variables: F,
    pub conjugated_v_edges: F,

    pub v_even: F,
    pub v_odd: F,
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
    // /// Construct a new symbolic register game straight from an explicit parity game.
    // ///
    // /// See [crate::register_game::RegisterGame::construct]
    // #[tracing::instrument(name = "Build Symbolic Register Game", skip_all)]
    // pub fn from_explicit(explicit: &ParityGame, k: Rank, controller: Owner) -> symbolic::Result<Self> {
    //     let k = k as usize;
    //     let register_bits_needed = (explicit.priority_max() as f64).log2().ceil() as usize;
    //     // Calculate the amount of variables we'll need for the binary encodings of the registers and vertices.
    //     let n_regis_vars = register_bits_needed * (k + 1);
    //     let n_variables = (explicit.vertex_count() as f64).log2().ceil() as usize;
    //
    //     let manager = F::new_manager(explicit.vertex_count(), explicit.vertex_count(), 12);
    //
    //     // Construct base building blocks for the BDD
    //     let base_true = manager.with_manager_exclusive(|man| F::t(man));
    //     let base_false = manager.with_manager_exclusive(|man| F::f(man));
    //
    //     let (variables, edge_variables) = manager.with_manager_exclusive(|man| {
    //         let next_move = F::new_var(man)?;
    //         let next_move_edge = F::new_var(man)?;
    //         Ok::<_, symbolic::BddError>((
    //             RegisterVertexVars::new(
    //                 next_move,
    //                 (0..n_regis_vars)
    //                     .flat_map(|_| F::new_var(man))
    //                     .collect_vec()
    //                     .into_iter(),
    //                 (0..n_variables).flat_map(|_| F::new_var(man)),
    //             ),
    //             RegisterVertexVars::new(
    //                 next_move_edge,
    //                 (0..n_regis_vars)
    //                     .flat_map(|_| F::new_var(man))
    //                     .collect_vec()
    //                     .into_iter(),
    //                 (0..n_variables).flat_map(|_| F::new_var(man)),
    //             ),
    //         ))
    //     })?;
    //
    //     let mut var_encoder = CachedSymbolicEncoder::new(&manager, variables.vertex_vars().into());
    //     let mut e_var_encoder = CachedSymbolicEncoder::new(&manager, edge_variables.vertex_vars().into());
    //
    //     tracing::debug!("Starting player vertex set construction");
    //
    //     let mut s_even = base_false.clone();
    //     let mut s_odd = base_false.clone();
    //     for v_idx in explicit.vertices_index() {
    //         let vertex = explicit.get(v_idx).expect("Impossible");
    //         let mut v_expr = var_encoder.encode(v_idx.index())?.clone();
    //
    //         // The opposing player can only play with E_move
    //         // Thus, `next_move_var` should be `1/true`
    //         if controller != vertex.owner {
    //             v_expr = v_expr.and(variables.next_move_var())?;
    //         }
    //
    //         // Set owners
    //         // Note that possible register contents are not constrained here, this will be done in the edge relation.
    //         match vertex.owner {
    //             Owner::Even => s_even = s_even.or(&v_expr)?,
    //             Owner::Odd => s_odd = s_odd.or(&v_expr)?,
    //         }
    //     }
    //
    //     tracing::debug!("Starting E_move construction");
    //
    //     let mut e_move = base_false.clone();
    //     let iff_register_condition = variables
    //         .register_vars()
    //         .iter()
    //         .zip(edge_variables.register_vars())
    //         .fold(base_true.clone(), |acc, (r, r_next)| {
    //             acc.and(&r.equiv(r_next).unwrap()).unwrap()
    //         });
    //
    //     for edge in explicit.graph_edges() {
    //         let (source, target) = (
    //             var_encoder
    //                 .encode(edge.source().index())?
    //                 .and(&variables.next_move_var())?,
    //             e_var_encoder
    //                 .encode(edge.target().index())?
    //                 .diff(&edge_variables.next_move_var())?,
    //         );
    //         e_move = e_move.or(&source.and(&target)?.and(&iff_register_condition)?)?;
    //     }
    //
    //     tracing::debug!("Starting E_i construction");
    //     let n_priorities = if controller == Owner::Even { k + 1 } else { k + 2 };
    //     let mut priorities = (0..=2 * n_priorities)
    //         .into_iter()
    //         .map(|val| (val, base_false.clone()))
    //         .collect::<ahash::HashMap<_, _>>();
    //     // All E_move edges get a priority of `0` by default.
    //     let _ = priorities
    //         .entry(0)
    //         .and_modify(|pri| *pri = variables.next_move_var().not().unwrap());
    //
    //     let mut e_i_edges = vec![base_false.clone(); k + 1];
    //
    //     for i in 0..=k {
    //         let e_i = &mut e_i_edges[i];
    //         // From t=0 -> t=1
    //         let base_edge = edge_variables.next_move_var().diff(&variables.next_move_var())?;
    //
    //         for (v_idx, vertex) in explicit.vertices_and_index() {}
    //     }
    //
    //     let conj_v = variables.conjugated(&base_true)?;
    //     let conj_e = edge_variables.conjugated(&base_true)?;
    //
    //     Ok(Self {
    //         manager,
    //         variables,
    //         variables_edges: edge_variables,
    //         conjugated_variables: conj_v,
    //         conjugated_v_edges: conj_e,
    //         v_even: s_even,
    //         v_odd: s_odd,
    //         e_move,
    //         base_true,
    //         base_false,
    //     })
    // }

    /// Construct a new symbolic register game straight from an explicit parity game.
    ///
    /// See [crate::register_game::RegisterGame::construct]
    #[tracing::instrument(name = "Build Symbolic Register Game", skip_all)]
    pub fn from_symbolic(explicit: &ParityGame, k: Rank, controller: Owner) -> symbolic::Result<Self> {
        let k = k as usize;
        let num_registers = k + 1;
        let register_bits_needed = (explicit.priority_max() as f64 + 1.).log2().ceil() as usize;
        // Calculate the amount of variables we'll need for the binary encodings of the registers and vertices.
        let n_register_vars = register_bits_needed * num_registers;
        let n_variables = (explicit.vertex_count() as f64).log2().ceil() as usize;

        let manager = F::new_manager(explicit.vertex_count(), explicit.vertex_count(), 12);

        // Construct base building blocks for the BDD
        let base_true = manager.with_manager_exclusive(|man| F::t(man));
        let base_false = manager.with_manager_exclusive(|man| F::f(man));

        let (variables, edge_variables) = manager.with_manager_exclusive(|man| {
            let next_move = F::new_var(man)?;
            let next_move_edge = F::new_var(man)?;

            let registers = (0..n_register_vars).flat_map(|_| F::new_var(man)).collect_vec();
            let next_registers = (0..n_register_vars).flat_map(|_| F::new_var(man)).collect_vec();

            Ok::<_, symbolic::BddError>((
                RegisterVertexVars::new(
                    next_move,
                    registers.into_iter(),
                    (0..n_variables).flat_map(|_| F::new_var(man)),
                ),
                RegisterVertexVars::new(
                    next_move_edge,
                    next_registers.into_iter(),
                    (0..n_variables).flat_map(|_| F::new_var(man)),
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
            Owner::Even => (
                sg.vertices_even.clone(),
                sg.vertices_odd.and(variables.next_move_var())?,
            ),
            Owner::Odd => (
                sg.vertices_even.and(variables.next_move_var())?,
                sg.vertices_odd.clone(),
            ),
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

        // Calculate all possible next register sets based on a current set of registers and priorities.
        // This will be vastly more efficient than constructing the full register game following the explicit naive algorithm.
        // However, should a game with `n` vertices and `n` priorities be used, this approach will likely take... forever...
        // let unique_register_contents = sg.priorities.keys().copied().permutations(num_registers);
        let unique_register_contents =
            itertools::repeat_n(sg.priorities.keys().copied(), num_registers).multi_cartesian_product();

        for permutation in unique_register_contents {
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
                    let all_vertices = starting_vertices.and(&next_encoding)?;

                    let e_i = &mut e_i_edges[i];
                    *e_i = e_i.or(&all_vertices)?;

                    // Update the priority set as well
                    let next_base_encoding = perm_encoder.encode_many(&next_registers)?;
                    let vertices_with_priority = prio_bdd.and(&next_base_encoding)?.and(variables.next_move_var())?;
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
            manager,
            variables,
            variables_edges: edge_variables,
            conjugated_variables: conj_v,
            conjugated_v_edges: conj_e,
            v_even: s_even,
            v_odd: s_odd,
            e_move,
            e_i: e_i_edges,
            priorities,
            base_true,
            base_false,
        })
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
    n_register_vars: usize,
}

impl<F: BooleanFunctionExtensions> RegisterVertexVars<F> {
    pub fn new(
        next_move: F,
        register_vars: impl ExactSizeIterator<Item = F>,
        vertex_vars: impl Iterator<Item = F>,
    ) -> Self {
        Self {
            n_register_vars: register_vars.len(),
            all_variables: [next_move]
                .into_iter()
                .chain(register_vars)
                .chain(vertex_vars)
                .collect(),
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
        &self.all_variables[self.n_register_vars + 1..]
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
    }
}

#[cfg(test)]
mod tests {
    use oxidd_core::{
        function::{BooleanFunction, BooleanFunctionQuant},
        Manager, ManagerRef,
    };

    use pg_parser::parse_pg;

    use crate::{
        explicit::ParityGame,
        Owner,
        symbolic::{BDD, helpers::CachedSymbolicEncoder, register_game::SymbolicRegisterGame},
        tests::example_dir,
        visualize::DotWriter,
    };

    #[test]
    pub fn test_tue() {
        let input = std::fs::read_to_string(example_dir().join("tue_example.pg")).unwrap();
        let pg = parse_pg(&mut input.as_str()).unwrap();
        let game = ParityGame::new(pg).unwrap();
        let s_pg: SymbolicRegisterGame<BDD> = SymbolicRegisterGame::from_symbolic(&game, 1, Owner::Even).unwrap();
        s_pg.manager.with_manager_exclusive(|man| man.gc());

        std::fs::write("out.dot", DotWriter::write_dot_symbolic_register(&s_pg, []).unwrap()).unwrap();
    }

    #[tracing_test::traced_test]
    #[test]
    pub fn test_small() {
        let game = small_pg().unwrap();
        let s_pg: SymbolicRegisterGame<BDD> = SymbolicRegisterGame::from_symbolic(&game, 1, Owner::Even).unwrap();
        s_pg.manager.with_manager_exclusive(|man| man.gc());
        // let vert = CachedSymbolicEncoder::encode_impl(&s_pg.variables.vertex_vars(), 0usize).unwrap();
        // let next_move = vert.and(&s_pg.variables.next_move_var()).unwrap();
        // let register_contents = CachedSymbolicEncoder::encode_impl(&s_pg.variables.register_vars(), 3usize).unwrap();
        // let full_vert = next_move.and(&register_contents).unwrap();
        //
        // let total_next_v = s_pg
        //     .e_move
        //     .and(&full_vert)
        //     .unwrap()
        //     .exist(&s_pg.conjugated_variables)
        //     .unwrap();
        // let real = s_pg.rev_edge_substitute(&total_next_v).unwrap();
        // s_pg.manager.with_manager_exclusive(|man| man.gc());

        // std::fs::write(
        //     "out.dot",
        //     DotWriter::write_dot_symbolic_register(&s_pg, [(&real, "Next V".to_string())]).unwrap(),
        // )
        // .unwrap();
        std::fs::write("out.dot", DotWriter::write_dot_symbolic_register(&s_pg, []).unwrap()).unwrap();
    }

    fn small_pg() -> eyre::Result<ParityGame> {
        let mut pg = r#"parity 3;
0 1 1 0,1 "0";
1 1 0 2 "1";
2 2 0 2 "2";"#;
        let pg = pg_parser::parse_pg(&mut pg).unwrap();
        ParityGame::new(pg)
    }
}
