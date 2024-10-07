use ahash::HashSetExt;
use itertools::Itertools;
use oxidd_core::{ManagerRef, WorkerManager};

use crate::symbolic::helpers::SymbolicEncoder;
use crate::symbolic::oxidd_extensions::GeneralBooleanFunction;

#[cfg(test)]
mod tests {
    use ahash::HashSet;
    use itertools::Itertools;
    use oxidd_core::ManagerRef;

    use crate::{
        explicit,
        explicit::ParityGame,
        symbolic,
        symbolic::{
            oxidd_extensions::BddExtensions,
            register_game::{RegisterLayers, SymbolicRegisterGame},
            solvers::symbolic_zielonka::SymbolicZielonkaSolver,
            BDD,
        },
        visualize::DotWriter,
        Owner,
    };

    #[test]
    pub fn test_trivial_2() -> eyre::Result<()> {
        let game = crate::tests::trivial_pg_2();
        let srg: SymbolicRegisterGame<BDD> = SymbolicRegisterGame::from_symbolic(&game, 0, Owner::Even).unwrap();

        let spg = srg.to_symbolic_parity_game()?;
        let mut solver = SymbolicZielonkaSolver::new(&spg);
        let (w_even, w_odd) = solver.run_symbolic();
        let (wp_even, wp_odd) = srg.project_winning_regions(&w_even, &w_odd)?;

        assert_eq!(wp_even, vec![0]);
        assert_eq!(wp_odd, vec![1]);

        Ok(())
    }

    #[test]
    pub fn test_amba_decomposed() -> eyre::Result<()> {
        // Register-index=1, but also gets correct results for Owner::Even as the controller and k=0.
        let (game, compare) = crate::tests::load_and_compare_example("amba_decomposed_decode.tlsf.ehoa.pg");
        let srg: SymbolicRegisterGame<BDD> = SymbolicRegisterGame::from_symbolic(&game, 1, Owner::Even).unwrap();

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
        let s_pg: SymbolicRegisterGame<BDD> = SymbolicRegisterGame::from_symbolic(&game, k, Owner::Even)?;
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
        Ok(crate::tests::parse_pg_from_str(pg))
    }
}
