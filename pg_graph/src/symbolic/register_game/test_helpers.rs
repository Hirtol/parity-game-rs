use ahash::{HashSet, HashSetExt};
use itertools::Itertools;
use oxidd_core::{function::BooleanFunction, Manager, ManagerRef};

use crate::{
    explicit::ParityGame,
    Owner,
    symbolic,
    symbolic::{
        BDD, helpers::CachedSymbolicEncoder, oxidd_extensions::BddExtensions,
        register_game::SymbolicRegisterGame, sat::decode_split_assignments,
    }, Vertex,
};

pub fn symbolic_to_explicit_alt<'a>(symb: &SymbolicRegisterGame<BDD>) -> ParityGame {
    let mut pg = ParityGame::empty();
    let n_reg_vars = symb.variables.n_register_vars;
    let mut reg_v_index = ahash::HashMap::default();
    let mut edges = ahash::HashMap::default();

    let (r_vertex_window, r_next_vertex_window) = (
        symb.variables
            .manager_layer_indices
            .values()
            .flatten()
            .copied()
            .collect_vec(),
        symb.variables_edges
            .manager_layer_indices
            .values()
            .flatten()
            .copied()
            .collect_vec(),
    );

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

                        tracing::debug!("Vertex: ({vertex}, {owner:?}, {priority}) has edges: {targets:?}");

                        let edge = edges.entry(vertex).or_insert_with(HashSet::new);
                        edge.extend(targets);
                    }
                }
            }

            Ok::<_, symbolic::BddError>(())
        })
        .unwrap();

    pg.labels = vec![None; reg_v_index.keys().len()];
    for (v_idx, edges) in edges {
        let start_idx = reg_v_index.get(&v_idx).unwrap();
        pg.labels[start_idx.index()] = v_idx.to_string().into();
        for target in edges {
            let Some(target_idx) = reg_v_index.get(&target) else {
                tracing::warn!("Missing edge target in reg_v_index: {target}");
                continue;
            };
            pg.graph.add_edge(*start_idx, *target_idx, ());
        }
    }

    pg
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
            register_game::{RegisterLayers, SymbolicRegisterGame, test_helpers},
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

        let mut solver = SymbolicZielonkaSolver::new(&spg);

        let (w_even, w_odd) = solver.run_symbolic();
        let mut perm_encoder: MultiEncoder<Priority, _> = MultiEncoder::new(
            &s_pg.manager,
            &s_pg.variables.register_vars().into_iter().cloned().chunks(2),
        );

        let zero_registers = perm_encoder.encode_many([0]).unwrap();
        let zero_prio = CachedSymbolicEncoder::encode_impl(s_pg.variables.priority_vars(), 0u32).unwrap();

        let won_projected = w_odd
            .and(&zero_registers)
            .unwrap()
            .and(&zero_prio)
            .unwrap()
            .and(&s_pg.variables.next_move_var().not().unwrap())
            .unwrap();
        // println!("WON: {:#?}", spg.vertices_of_bdd(&won_projected));
        s_pg.manager.with_manager_shared(|man| {
            let valuations = won_projected.sat_assignments(man).collect_vec();
            println!("VALUES: {valuations:#?}");
            
            let l = crate::symbolic::sat::decode_split_assignments(
                valuations,
                &[s_pg.variables.var_indices(RegisterLayers::Vertex).as_slice()],
            );
            println!("VALUES: {l:#?}");
            // crate::symbolic::helpers::decode_assignments(valuations, self.variables.len())
        });
        s_pg.gc();

        std::fs::write(
            "out.dot",
            DotWriter::write_dot_symbolic_register(&s_pg, [(&won_projected, "won_projected_odd".to_string())]).unwrap(),
        )
        .unwrap();

        let converted_to_pg = test_helpers::symbolic_to_explicit_alt(&s_pg);
        std::fs::write("converted.dot", DotWriter::write_dot(&converted_to_pg).unwrap()).unwrap();

        // spg.gc();
        // std::fs::write("out.dot", DotWriter::write_dot_symbolic(&spg, []).unwrap()).unwrap();
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
