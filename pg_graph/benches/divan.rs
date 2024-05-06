use std::path::PathBuf;
use pg_graph::ParityGame;

fn main() {
    divan::main();
}

static GAMES: [&str; 2] = ["ActionConverter.tlsf.ehoa.pg", "amba_decomposed_arbiter_6.tlsf.ehoa.pg"];

#[divan::bench_group(max_time = 5)]
mod solver_benches {
    use oxidd::bdd::BDDFunction;
    use pg_graph::solvers;
    use pg_graph::symbolic::SymbolicParityGame;
    use oxidd_core::function::BooleanFunction;
    use oxidd_core::ManagerRef;
    use pg_graph::{ParityGame};
    use pg_graph::symbolic::helpers::BddExtensions;

    #[divan::bench]
    fn bench_substitution(bencher: divan::Bencher) {
        let pg = super::load_pg("amba_decomposed_arbiter_6.tlsf.ehoa.pg");
        let mut s_pg = SymbolicParityGame::from_explicit(&pg).unwrap();
        let mut true_base = s_pg.manager.with_manager_exclusive(|man| BDDFunction::t(man));
        let start_var = s_pg.variables.iter().fold(true_base.clone(), |acc, v_2| acc.and(v_2).unwrap());
        let other_vars = s_pg.variables_edges.iter().fold(true_base, |acc, v_2| acc.and(v_2).unwrap());

        bencher.bench(|| {
            let subs = s_pg.vertices.substitute(&start_var, &other_vars).unwrap();
            drop(subs);
            // Otherwise we're just benchmarking how quickly it can re-discover nodes.
            s_pg.gc();
        });
    }

    #[divan::bench(args = super::GAMES)]
    fn bench_small_progress(bencher: divan::Bencher, game: &str) {
        bencher.with_inputs(|| super::load_pg(game)).bench_values(|parity_game| {
            let mut game = solvers::small_progress::SmallProgressSolver::new(&parity_game);

            game.run();
        });
    }

    #[divan::bench(args = super::GAMES)]
    fn bench_zielonka(bencher: divan::Bencher, game: &str) {
        bencher.with_inputs(|| super::load_pg(game)).bench_values(|parity_game| {
            let mut game = solvers::zielonka::ZielonkaSolver::new(&parity_game);

            game.run();
        });
    }

    #[divan::bench(args = super::GAMES)]
    fn bench_symbolic_construction(bencher: divan::Bencher, game: &str) {
        bencher.with_inputs(|| super::load_pg(game)).bench_values(|parity_game| {
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
        .parent().unwrap()
        .join("game_examples")
}