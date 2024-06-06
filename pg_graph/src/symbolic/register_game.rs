use std::{collections::HashMap, marker::PhantomData};

use ecow::EcoVec;
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
        oxidd_extensions::{BddExtensions, BooleanFunctionExtensions, FunctionManagerExtension}, SymbolicParityGame,
    },
};

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

        // Calculate all possible next register sets based on a current set of registers and priorities.
        // This will be vastly more efficient than constructing the full register game following the explicit naive algorithm.
        // However, should a game with `n` vertices and `n` priorities be used, this approach will likely take... forever...
        // let unique_register_contents = sg.priorities.keys().copied().permutations(num_registers);
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

mod experiments {
    use itertools::Itertools;
    use oxidd_core::{function::BooleanFunction, ManagerRef};

    use crate::{
        explicit::ParityGame,
        Owner,
        symbolic,
        symbolic::{
            BDD, helpers::CachedSymbolicEncoder, oxidd_extensions::BddExtensions,
            register_game::SymbolicRegisterGame, sat::decode_split_assignments,
        }, Vertex,
    };

// fn symbolic_to_explicit<'a>(empty_pg: &'a ParityGame, symb: &SymbolicRegisterGame<BDD>) -> RegisterGame<'a> {
    //     let rg = RegisterGame::construct(&empty_pg, symb.k, symb.controller);
    //     let num_registers = symb.k + 1;
    //     let n_reg_vars = symb.variables.n_register_vars;
    //     let mut reg_v_index = ahash::HashMap::default();
    //
    //     let (r_vertex_window, r_next_vertex_window) = {
    //         // t
    //         let mut r_vertex_window = vec![0];
    //         // r
    //         let mut n_reg_range = 2..2 + n_reg_vars;
    //         r_vertex_window.extend(n_reg_range);
    //         // x
    //         let n_vertex_vars = symb.variables.vertex_vars().len();
    //         let max_reg_var_index = 2 + 2 * n_reg_vars;
    //         let n_vertex_range = max_reg_var_index..max_reg_var_index + n_vertex_vars;
    //         r_vertex_window.extend(n_vertex_range);
    //         // t'
    //         let mut r_vertex_next_window = vec![1];
    //         // r'
    //         let mut n_next_reg_range = 2 + n_reg_vars..2 + 2 * n_reg_vars;
    //         r_vertex_next_window.extend(n_next_reg_range);
    //         // x'
    //         let n_next_vertex_range = max_reg_var_index + n_vertex_vars..max_reg_var_index + n_vertex_vars * 2;
    //         r_vertex_next_window.extend(n_next_vertex_range);
    //         (r_vertex_window, r_vertex_next_window)
    //     };
    //
    //     symb.manager.with_manager_shared(|man| {
    //         let mut vertex_window = vec![&r_vertex_window[0..1]];
    //         // The set of vertice variables from the original game
    //         vertex_window.push(&r_vertex_window[1 + n_reg_vars..]);
    //         // Split all the registers into their own set
    //         vertex_window.extend(r_vertex_window[1..].chunks(n_reg_vars / num_registers as usize));
    //
    //         for (priority, bdd) in &symb.priorities {
    //             // First, all even
    //             let split_prios = [
    //                 (Owner::Even, bdd.and(&symb.v_even)?),
    //                 (Owner::Odd, bdd.and(&symb.v_odd)?),
    //             ];
    //             for (owner, prio) in split_prios {
    //                 let vertex_contents = prio.sat_assignments(man).collect_vec();
    //
    //                 let register_vertices = decode_split_assignments(&vertex_contents, &vertex_window);
    //
    //                 for next_move in register_vertices[0] {
    //                     for &original_pg_id in &register_vertices[1] {
    //                         let new_v = RegisterVertex {
    //                             original_graph_id: original_pg_id.into(),
    //                             priority: *priority,
    //                             owner,
    //                             register_state: Default::default(),
    //                             next_action: ChosenAction::from_num(next_move),
    //                         };
    //                     }
    //                 }
    //             }
    //             // Then all odd
    //         }
    //     });
    //
    //     rg
    // }

    pub fn symbolic_to_explicit_alt<'a>(symb: &SymbolicRegisterGame<BDD>) -> ParityGame {
        let mut pg = ParityGame::empty();
        let n_reg_vars = symb.variables.n_register_vars;
        let mut reg_v_index = ahash::HashMap::default();
        let mut edges = ahash::HashMap::default();

        let (r_vertex_window, r_next_vertex_window) = {
            // t
            let mut r_vertex_window = vec![0];
            // r
            let mut n_reg_range = (2..2 + n_reg_vars);
            r_vertex_window.extend(n_reg_range);
            // x
            let n_vertex_vars = symb.variables.vertex_vars().len();
            let max_reg_var_index = 2 + 2 * n_reg_vars;
            let n_vertex_range = (max_reg_var_index..max_reg_var_index + n_vertex_vars);
            r_vertex_window.extend(n_vertex_range);
            // t'
            let mut r_vertex_next_window = vec![1];
            // r'
            let mut n_next_reg_range = (2 + n_reg_vars..2 + 2 * n_reg_vars);
            r_vertex_next_window.extend(n_next_reg_range);
            // x'
            let n_next_vertex_range = (max_reg_var_index + n_vertex_vars..max_reg_var_index + n_vertex_vars * 2);
            r_vertex_next_window.extend(n_next_vertex_range);
            (r_vertex_window, r_vertex_next_window)
        };

        symb.manager
            .with_manager_shared(|man| {
                for (priority, bdd) in &symb.priorities {
                    let split_prios = [
                        (Owner::Even, bdd.and(&symb.v_even)?),
                        (Owner::Odd, bdd.and(&symb.v_odd)?),
                    ];
                    for (owner, prio) in split_prios {
                        let vertex_contents = prio.sat_assignments(man).collect_vec();

                        let register_vertices = decode_split_assignments(&vertex_contents, &[&r_vertex_window])
                            .pop()
                            .unwrap();
                        let mut cached_encoder =
                            CachedSymbolicEncoder::new(&symb.manager, symb.variables.all_variables.clone());

                        for vertex in register_vertices {
                            let idx = reg_v_index.entry(vertex).or_insert_with(|| {
                                pg.graph.add_node(Vertex {
                                    priority: *priority,
                                    owner,
                                })
                            });

                            let encoded = cached_encoder.encode(vertex)?;
                            let edges_from = encoded.and(&symb.edges)?;
                            let targets =
                                decode_split_assignments(edges_from.sat_assignments(man), &[&r_next_vertex_window])
                                    .pop()
                                    .unwrap();

                            tracing::debug!("Vertex: {vertex} has edges: {targets:?}");

                            edges
                                .entry(vertex)
                                .and_modify(|v: &mut Vec<u32>| v.extend(targets))
                                .or_insert_with(Vec::new);
                        }
                    }
                }

                Ok::<_, symbolic::BddError>(())
            })
            .unwrap();

        for (v_idx, edges) in edges {
            let start_idx = reg_v_index.get(&v_idx).unwrap();
            for target in edges {
                let target_idx = reg_v_index.get(&target).unwrap();
                pg.graph.add_edge(*start_idx, *target_idx, ());
            }
        }

        pg
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use oxidd_core::{
        function::{BooleanFunction, BooleanFunctionQuant},
        Manager, ManagerRef,
    };

    use pg_parser::parse_pg;

    use crate::{
        explicit,
        explicit::ParityGame,
        Owner,
        Priority,
        symbolic::{
            BDD,
            helpers::{CachedSymbolicEncoder, MultiEncoder},
            oxidd_extensions::BddExtensions,
            register_game::SymbolicRegisterGame,
            solvers::symbolic_zielonka::SymbolicZielonkaSolver,
        },
        tests::example_dir, visualize::DotWriter,
    };

    #[tracing_test::traced_test]
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
        let game = trivial_pg_2().unwrap();
        let normal_sol = explicit::solvers::zielonka::ZielonkaSolver::new(&game).run();
        println!("Expected: {normal_sol:#?}");

        let s_pg: SymbolicRegisterGame<BDD> = SymbolicRegisterGame::from_symbolic(&game, 0, Owner::Even).unwrap();
        s_pg.manager.with_manager_exclusive(|man| man.gc());

        let spg = s_pg.to_symbolic_parity_game();
        spg.gc();

        let mut solver = SymbolicZielonkaSolver::new(&spg);

        let (w_even, w_odd) = solver.run_symbolic();
        let mut perm_encoder: MultiEncoder<Priority, _> = MultiEncoder::new(
            &s_pg.manager,
            &s_pg.variables.register_vars().into_iter().cloned().chunks(2),
        );
        let zero_registers = perm_encoder.encode_many([0]).unwrap();
        let won_projected = w_even
            .and(&zero_registers)
            .unwrap()
            .and(&s_pg.variables.next_move_var().not().unwrap())
            .unwrap();
        // println!("WON: {:#?}", spg.vertices_of_bdd(&won_projected));
        s_pg.manager.with_manager_shared(|man| {
            let valuations = won_projected.sat_assignments(man).collect_vec();
            println!("VALUES: {valuations:#?}");
            let l = crate::symbolic::sat::decode_split_assignments(valuations, &[&[4]]);
            println!("VALUES: {l:#?}");
            // crate::symbolic::helpers::decode_assignments(valuations, self.variables.len())
        });
        s_pg.gc();

        std::fs::write(
            "out.dot",
            DotWriter::write_dot_symbolic_register(&s_pg, [(&won_projected, "won_projected_even".to_string())])
                .unwrap(),
        )
        .unwrap();

        let converted_to_pg = super::experiments::symbolic_to_explicit_alt(&s_pg);
        std::fs::write("converted.dot", DotWriter::write_dot(&converted_to_pg).unwrap()).unwrap();
        // std::fs::write(
        //     "out.dot",
        //     DotWriter::write_dot_symbolic(&spg, []).unwrap(),
        // )
        // .unwrap();
    }

    fn small_pg() -> eyre::Result<ParityGame> {
        let mut pg = r#"parity 3;
0 1 1 0,1 "0";
1 1 0 2 "1";
2 2 0 2 "2";"#;
        let pg = pg_parser::parse_pg(&mut pg).unwrap();
        ParityGame::new(pg)
    }

    fn other_pg() -> eyre::Result<ParityGame> {
        let mut pg = r#"parity 1;
0 1 1 0 "0";"#;
        let pg = pg_parser::parse_pg(&mut pg).unwrap();
        ParityGame::new(pg)
    }

    fn trivial_pg_2() -> eyre::Result<ParityGame> {
        let mut pg = r#"parity 2;
0 0 0 0,1 "0";
1 1 1 1 "1";"#;
        let pg = pg_parser::parse_pg(&mut pg).unwrap();
        ParityGame::new(pg)
    }
}
