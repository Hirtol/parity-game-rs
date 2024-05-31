use pg_graph::ParityGame;
use std::path::PathBuf;

fn main() {
    divan::main();
}

static GAMES: [&str; 2] = ["ActionConverter.tlsf.ehoa.pg", "amba_decomposed_arbiter_6.tlsf.ehoa.pg"];

#[divan::bench_group(max_time = 5)]
mod solver_benches {

    use pg_graph::{
        explicit::solvers::{small_progress::SmallProgressSolver, zielonka::ZielonkaSolver},
        symbolic::{solvers::symbolic_zielonka::SymbolicZielonkaSolver, SymbolicParityGame},
    };

    #[divan::bench(args = super::GAMES)]
    fn bench_symbolic_zielonka(bencher: divan::Bencher, game: &str) {
        bencher
            .with_inputs(|| {
                let pg = super::load_pg(game);
                SymbolicParityGame::from_explicit(&pg).unwrap()
            })
            .bench_values(|s_pg| {
                let mut game = SymbolicZielonkaSolver::new(&s_pg);

                game.run();
            });
    }

    #[divan::bench(args = super::GAMES)]
    fn bench_small_progress(bencher: divan::Bencher, game: &str) {
        bencher
            .with_inputs(|| super::load_pg(game))
            .bench_values(|parity_game| {
                let mut game = SmallProgressSolver::new(&parity_game);

                game.run();
            });
    }

    #[divan::bench(args = super::GAMES)]
    fn bench_zielonka(bencher: divan::Bencher, game: &str) {
        bencher
            .with_inputs(|| super::load_pg(game))
            .bench_values(|parity_game| {
                let mut game = ZielonkaSolver::new(&parity_game);

                game.run();
            });
    }

    #[divan::bench(args = super::GAMES)]
    fn bench_symbolic_construction(bencher: divan::Bencher, game: &str) {
        bencher
            .with_inputs(|| super::load_pg(game))
            .bench_values(|parity_game| {
                let s_pg = SymbolicParityGame::from_explicit(&parity_game);
            });
    }
}

pub fn load_pg(game: &str) -> ParityGame {
    let path = crate::example_dir().join(game);
    let data = std::fs::read_to_string(&path).unwrap();
    let graph = pg_parser::parse_pg(&mut data.as_str()).unwrap();
    let parity_game = pg_graph::ParityGame::new(graph).unwrap();
    parity_game
}

pub fn example_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("game_examples")
}
