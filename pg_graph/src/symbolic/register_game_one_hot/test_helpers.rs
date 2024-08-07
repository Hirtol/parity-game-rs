use ahash::{HashSet, HashSetExt};
use itertools::Itertools;
use oxidd_core::{ManagerRef, WorkerManager};
use oxidd_core::util::OptBool;

use crate::{
    explicit::ParityGame,
    Owner,
    symbolic,
    symbolic::{
        helpers::CachedBinaryEncoder, oxidd_extensions::BddExtensions,
        register_game_one_hot::OneHotRegisterGame, sat::decode_split_assignments,
    }, Vertex,
};
use crate::symbolic::helpers::SymbolicEncoder;
use crate::symbolic::oxidd_extensions::GeneralBooleanFunction;
use crate::symbolic::sat::TruthAssignmentsIterator;

pub fn symbolic_to_explicit_alt<F: GeneralBooleanFunction>(symb: &OneHotRegisterGame<F>) -> ParityGame
    where for<'id> F::Manager<'id>: WorkerManager,
          for<'a, 'b> TruthAssignmentsIterator<'b, 'a, F>: Iterator<Item=Vec<OptBool>> {
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
                        CachedBinaryEncoder::new(&symb.manager, symb.variables.all_variables.clone());

                    for vertex in register_vertices {
                        let _ = reg_v_index.entry(vertex).or_insert_with(|| {
                            pg.graph.add_node(Vertex {
                                priority: *priority,
                                owner,
                            })
                        });

                        let encoded = cached_encoder.encode(vertex)?;
                        let edges_from = encoded.and(&symb.edges()?)?;
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
    use ahash::HashSet;
    use itertools::Itertools;
    use oxidd_core::ManagerRef;

    use crate::{
        explicit,
        explicit::ParityGame,
        Owner,
        symbolic,
        symbolic::{
            BDD,
            oxidd_extensions::BddExtensions,
            register_game_one_hot::{OneHotRegisterGame, RegisterLayers},
            solvers::symbolic_zielonka::SymbolicZielonkaSolver,
        },
        visualize::DotWriter,
    };
    use crate::explicit::ParityGraph;
    use crate::symbolic::register_game_one_hot::helpers::OneHotEncoder;

    #[test]
    pub fn test_trivial_2() -> eyre::Result<()> {
        let game = crate::tests::trivial_pg_2()?;
        let srg: OneHotRegisterGame<BDD> = OneHotRegisterGame::from_symbolic(&game, 0, Owner::Even).unwrap();
        let encod = OneHotEncoder::new(game.priorities_unique().sorted(), srg.variables.register_vars().into_iter().cloned().collect())?;

        let spg = srg.to_symbolic_parity_game()?;
        let mut solver = SymbolicZielonkaSolver::new(&spg);
        let (w_even, w_odd) = solver.run_symbolic();
        let (wp_even, wp_odd) = srg.projected_winning_regions(&w_even, &w_odd)?;

        spg.gc();
        std::fs::write(
            "out.dot",
            DotWriter::write_dot_symbolic_register_hot(
                &srg,
                [
                    (encod.mapping.get(&1).unwrap(), "prio 1 register".to_string()),
                    (encod.mapping.get(&0).unwrap(), "prio 0 register".to_string()),
                    (&w_odd, "won_odd".to_string()),
                    (&w_even, "won_even".to_string()),
                    (&wp_odd, "won_projected_odd".to_string()),
                    (&wp_even, "won_projected_even".to_string()),
                ],
            )
                .unwrap(),
        )
            .unwrap();
        
        let (wp_even, wp_odd) = srg.project_winning_regions(&w_even, &w_odd)?;
        assert_eq!(wp_even, vec![0]);
        assert_eq!(wp_odd, vec![1]);

        Ok(())
    }

    #[tracing_test::traced_test]
    #[test]
    pub fn test_amba_decomposed() -> eyre::Result<()> {
        // Register-index=1, but also gets correct results for Owner::Even as the controller and k=0.
        let (game, compare) = crate::tests::load_and_compare_example("amba_decomposed_decode.tlsf.ehoa.pg");
        let srg: OneHotRegisterGame<BDD> = OneHotRegisterGame::from_symbolic(&game, 0, Owner::Even).unwrap();

        srg.gc();
        println!(
            "RVars: {:#?} - BDD Size: {:#?}",
            srg.variables.register_vars().len(),
            srg.bdd_node_count()
        );
        
        let spg = srg.to_symbolic_parity_game()?;
        let mut solver = SymbolicZielonkaSolver::new(&spg);
        let (w_even, w_odd) = solver.run_symbolic();
        let (wp_even, wp_odd) = srg.project_winning_regions(&w_even, &w_odd)?;

        let _ = explicit::solvers::zielonka::ZielonkaSolver::new(&game).run();
        
        assert_eq!(wp_even.into_iter().collect::<HashSet<_>>(), vec![2, 0, 5, 3].into_iter().collect::<HashSet<_>>());
        assert_eq!(wp_odd.into_iter().collect::<HashSet<_>>(), vec![6, 4, 1].into_iter().collect::<HashSet<_>>());

        Ok(())
    }

    #[tracing_test::traced_test]
    #[test]
    pub fn test_small() -> symbolic::Result<()> {
        let (game, compare) = crate::tests::load_and_compare_example("amba_decomposed_arbiter_6.tlsf.ehoa.pg");

        let normal_sol = explicit::solvers::zielonka::ZielonkaSolver::new(&game).run();
        println!("Expected: {normal_sol:#?}");

        let k = 2;
        let s_pg: OneHotRegisterGame<BDD> = OneHotRegisterGame::from_symbolic(&game, k, Owner::Even)?;
        s_pg.gc();

        println!(
            "RVars: {:#?} - BDD Size: {:#?}",
            s_pg.variables.register_vars().len(),
            s_pg.bdd_node_count()
        );

        let spg = s_pg.to_symbolic_parity_game()?;
        let mut solver = SymbolicZielonkaSolver::new(&spg);
        let (w_even, w_odd) = solver.run_symbolic();

        let (wp_even, wp_odd) = s_pg.projected_winning_regions(&w_even, &w_odd)?;

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
            DotWriter::write_dot_symbolic_register_hot(
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
        //TODO: The below game doesn't get a correct solution, even though it's very similar.
        // This seems to be caused by the lack of a zero priority in the underlying game?
        let mut pg = r#"parity 3;
0 1 1 0,1 "0";
1 1 0 2 "1";
2 2 0 2 "2";"#;
        // This one has the problem that the solution should be [Odd, Even, Even], but it thinks [Odd, Odd, Even]
        //         let mut pg = r#"parity 3;
        // 0 1 1 0,1 "0";
        // 1 1 0 2 "1";
        // 2 0 0 2 "2";"#;
        //         let mut pg = r#"parity 3;
        // 0 0 0 0,1 "0";
        // 1 0 1 2 "1";
        // 2 1 1 2 "2";"#;
        let pg = pg_parser::parse_pg(&mut pg).unwrap();
        ParityGame::new(pg)
    }
}
