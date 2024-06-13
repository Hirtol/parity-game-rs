use ahash::{HashSet, HashSetExt};
use itertools::Itertools;
use oxidd_core::{function::BooleanFunction, Manager, ManagerRef};

use crate::{
    explicit::ParityGame,
    symbolic,
    symbolic::{
        helpers::CachedSymbolicEncoder, oxidd_extensions::BddExtensions, register_game::SymbolicRegisterGame,
        sat::decode_split_assignments, BDD,
    },
    Owner, Vertex,
};

pub fn symbolic_to_explicit_alt<'a>(symb: &SymbolicRegisterGame<BDD>) -> ParityGame {
    let mut pg = ParityGame::empty();
    let mut reg_v_index = ahash::HashMap::default();
    let mut edges = ahash::HashMap::default();

    let (r_vertex_window, r_next_vertex_window) = (
        symb.variables
            .manager_layer_indices
            .values()
            .flatten()
            .copied()
            .sorted()
            .collect_vec(),
        symb.variables_edges
            .manager_layer_indices
            .values()
            .flatten()
            .copied()
            .sorted()
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
        symbolic,
        symbolic::{
            helpers::{CachedSymbolicEncoder, MultiEncoder},
            oxidd_extensions::BddExtensions,
            register_game::{test_helpers, RegisterLayers, SymbolicRegisterGame},
            solvers::symbolic_zielonka::SymbolicZielonkaSolver,
            BDD,
        },
        tests::example_dir,
        visualize::{DotWriter, VisualRegisterGame},
        Owner, Priority,
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
    pub fn test_convert_symb_to_pg() -> symbolic::Result<()> {
        // let input = std::fs::read_to_string(example_dir().join("amba_decomposed_decode.tlsf.ehoa.pg")).unwrap();
        // let pg = parse_pg(&mut input.as_str()).unwrap();
        // let game = ParityGame::new(pg).unwrap();

        let game = trivial_pg_2().unwrap();
        let register_index = 0;
        let controller = Owner::Even;

        let rg = explicit::register_game::RegisterGame::construct_2021(&game, register_index, controller);
        let explicit_rg = DotWriter::write_dot(&VisualRegisterGame(&rg)).unwrap();
        std::fs::write("explicit_rg.dot", &explicit_rg).unwrap();

        let s_pg: SymbolicRegisterGame<BDD> = SymbolicRegisterGame::from_symbolic(&game, register_index, controller)?;
        s_pg.gc();

        let converted_to_pg = test_helpers::symbolic_to_explicit_alt(&s_pg);
        std::fs::write("converted_rg.dot", DotWriter::write_dot(&converted_to_pg).unwrap()).unwrap();
        Ok(())
    }

    #[tracing_test::traced_test]
    #[test]
    pub fn test_small() -> symbolic::Result<()> {
        // amba_decomposed_decode.tlsf.ehoa.pg
        let input = std::fs::read_to_string(example_dir().join("two_counters_4.pg")).unwrap();
        let pg = parse_pg(&mut input.as_str()).unwrap();
        let game = ParityGame::new(pg).unwrap();

        // let game = small_pg().unwrap();
        let normal_sol = explicit::solvers::zielonka::ZielonkaSolver::new(&game).run();
        println!("Expected: {normal_sol:#?}");

        let k = 2;
        let s_pg: SymbolicRegisterGame<BDD> = SymbolicRegisterGame::from_symbolic(&game, k, Owner::Even)?;
        s_pg.manager.with_manager_exclusive(|man| man.gc());

        let spg = s_pg.to_symbolic_parity_game();

        let mut solver = SymbolicZielonkaSolver::new(&spg);
        println!("RVars: {:#?}", s_pg.variables.register_vars().len());
        let (w_even, w_odd) = solver.run_symbolic();

        let chunk_size = s_pg.variables.register_vars().len() / (k + 1) as usize;
        let mut perm_encoder: MultiEncoder<Priority, _> = MultiEncoder::new(
            &s_pg.manager,
            &s_pg.variables.register_vars().into_iter().cloned().chunks(chunk_size),
        );

        let zero_registers = perm_encoder.encode_many(vec![0; (k + 1) as usize]).unwrap();
        let zero_prio = CachedSymbolicEncoder::encode_impl(s_pg.variables.priority_vars(), 0u32).unwrap();

        let projector_func = zero_registers
            .and(&zero_prio)?
            .and(&s_pg.variables.next_move_var().not()?)?;

        let (wp_even, wp_odd) = (w_even.and(&projector_func)?, w_odd.and(&projector_func)?);

        s_pg.manager.with_manager_shared(|man| {
            for (winner, winning_fn) in [("even", &wp_even), ("odd", &wp_odd)] {
                let valuations = winning_fn.sat_assignments(man).collect_vec();
                tracing::debug!(winner, "Valuations: {valuations:?}");
                let projected_win = crate::symbolic::sat::decode_split_assignments(
                    valuations,
                    &[s_pg.variables.var_indices(RegisterLayers::Vertex).as_slice()],
                )
                .pop()
                .unwrap();

                tracing::info!(winner, "Winning: {:?}", projected_win);
            }
        });
        s_pg.gc();

        std::fs::write(
            "out.dot",
            DotWriter::write_dot_symbolic_register(
                &s_pg,
                [
                    (&wp_odd, "won_projected_odd".to_string()),
                    (&wp_even, "won_projected_even".to_string()),
                ],
            )
            .unwrap(),
        )
        .unwrap();

        // let converted_to_pg = test_helpers::symbolic_to_explicit_alt(&s_pg);
        // std::fs::write("converted.dot", DotWriter::write_dot(&converted_to_pg).unwrap()).unwrap();

        // spg.gc();
        // std::fs::write("out.dot", DotWriter::write_dot_symbolic(&spg, []).unwrap()).unwrap();
        Ok(())
    }

    fn small_pg() -> eyre::Result<ParityGame> {
        //TODO: The below game doesn't get a correct solution, even though it's very similar (discontinuous priority problem?)
        //        let mut pg = r#"parity 3;
        // 0 1 1 0,1 "0";
        // 1 1 0 2 "1";
        // 2 2 0 2 "2";"#;
        // This one has the problem that the solution should be [Odd, Even, Even], but it thinks [Odd, Odd, Even]
        //         let mut pg = r#"parity 3;
        // 0 1 1 0,1 "0";
        // 1 1 0 2 "1";
        // 2 0 0 2 "2";"#;
        let mut pg = r#"parity 3;
0 0 0 0,1 "0";
1 0 1 2 "1";
2 1 1 2 "2";"#;
        let pg = pg_parser::parse_pg(&mut pg).unwrap();
        ParityGame::new(pg)
    }

    fn trivial_pg() -> eyre::Result<ParityGame> {
        let mut pg = r#"parity 1;
0 1 1 0 "0";"#;
        let pg = pg_parser::parse_pg(&mut pg).unwrap();
        ParityGame::new(pg)
    }

    fn trivial_pg_2() -> eyre::Result<ParityGame> {
        let mut pg = r#"parity 2;
0 0 0 0,1 "0";
1 1 1 1,0 "1";"#;
        let pg = pg_parser::parse_pg(&mut pg).unwrap();
        ParityGame::new(pg)
    }
}
