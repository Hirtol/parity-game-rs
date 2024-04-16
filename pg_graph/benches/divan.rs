use std::path::PathBuf;

fn main() {
    divan::main();
}

static GAMES: [&str; 2] = ["ActionConverter.tlsf.ehoa.pg", "amba_decomposed_arbiter_6.tlsf.ehoa.pg"];

#[divan::bench_group(max_time = 5)]
mod solver_benches {
    use pg_graph::solvers;

    #[divan::bench(args = super::GAMES)]
    fn bench_small_progress(bencher: divan::Bencher, game: &str) {
        bencher.with_inputs(|| {
            let path = super::example_dir().join(game);
            let data = std::fs::read_to_string(&path).unwrap();
            let graph = pg_parser::parse_pg(&mut data.as_str()).unwrap();
            let parity_game = pg_graph::ParityGame::new(graph).unwrap();
            parity_game
        }).bench_values(|parity_game| {
            let mut game = solvers::small_progress::SmallProgressSolver::new(&parity_game);

            game.run();
        });
    }
}

pub fn example_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .join("game_examples")
}