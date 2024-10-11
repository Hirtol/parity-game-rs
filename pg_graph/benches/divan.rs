use std::path::PathBuf;

use pg_graph::explicit::{ParityGame, ParityGameBuilder};

fn main() {
    divan::main();
}

static GAMES: [&str; 2] = ["ActionConverter.tlsf.ehoa.pg", "amba_decomposed_arbiter_6.tlsf.ehoa.pg"];

#[divan::bench_group(max_time = 5)]
mod solver_benches {
    use pg_graph::explicit::solvers::priority_promotion::PPSolver;
    use pg_graph::load_example;
    use pg_graph::symbolic::register_game::SymbolicRegisterGame;
    use pg_graph::symbolic::solvers::symbolic_register_zielonka::SymbolicRegisterZielonkaSolver;
    use pg_graph::symbolic::BDD;
    use pg_graph::{explicit::solvers::{small_progress::SmallProgressSolver, zielonka::ZielonkaSolver}, symbolic::{parity_game::SymbolicParityGame, solvers::symbolic_zielonka::SymbolicZielonkaSolver}, Owner};

    #[divan::bench(args = super::GAMES)]
    fn bench_symbolic_zielonka_bdd(bencher: divan::Bencher, game: &str) {
        let pg = load_example(game);
        let s_pg = SymbolicParityGame::from_explicit_bdd(&pg).unwrap();
        bencher.bench(|| {
            let mut game = SymbolicZielonkaSolver::new(&s_pg);

            game.run();
        });
    }

    #[divan::bench(args = super::GAMES)]
    fn bench_symbolic_zielonka_bcdd(bencher: divan::Bencher, game: &str) {
        let pg = load_example(game);
        let s_pg = SymbolicParityGame::from_explicit_bcdd(&pg).unwrap();
        bencher.bench(|| {
            let mut game = SymbolicZielonkaSolver::new(&s_pg);

            game.run();
        });
    }

    #[divan::bench(args = super::GAMES)]
    fn bench_register_symbolic_register_zielonka(bencher: divan::Bencher, game: &str) {
        let pg = load_example(game);
        let rg = SymbolicRegisterGame::<BDD>::from_symbolic(&pg, 1, Owner::Even).unwrap();
        bencher.bench(|| {
            let mut game = SymbolicRegisterZielonkaSolver::new(&rg);

            game.run_symbolic();
        });
    }

    #[divan::bench(args = super::GAMES)]
    fn bench_register_symbolic_zielonka(bencher: divan::Bencher, game: &str) {
        let pg = load_example(game);
        let rg = SymbolicRegisterGame::<BDD>::from_symbolic(&pg, 1, Owner::Even).unwrap();
        let s_rg = rg.to_symbolic_parity_game().unwrap();
        bencher.bench(|| {
            let mut game = SymbolicZielonkaSolver::new(&s_rg);

            game.run_symbolic();
        });
    }

    #[divan::bench(args = super::GAMES)]
    fn bench_small_progress(bencher: divan::Bencher, game: &str) {
        bencher
            .with_inputs(|| load_example(game))
            .bench_values(|parity_game| {
                let mut game = SmallProgressSolver::new(&parity_game);

                game.run();
            });
    }

    #[divan::bench(args = super::GAMES)]
    fn bench_zielonka(bencher: divan::Bencher, game: &str) {
        bencher
            .with_inputs(|| load_example(game))
            .bench_values(|parity_game| {
                let mut game = ZielonkaSolver::new(&parity_game);

                game.run();
            });
    }

    #[divan::bench(args = super::GAMES)]
    fn bench_pp(bencher: divan::Bencher, game: &str) {
        bencher
            .with_inputs(|| load_example(game))
            .bench_values(|parity_game| {
                let mut game = PPSolver::new(&parity_game);

                game.run();
            });
    }

    #[divan::bench(args = super::GAMES)]
    fn bench_symbolic_construction(bencher: divan::Bencher, game: &str) {
        bencher
            .with_inputs(|| load_example(game))
            .bench_values(|parity_game| {
                let _ = SymbolicParityGame::from_explicit_bdd(&parity_game);
            });
    }
}
