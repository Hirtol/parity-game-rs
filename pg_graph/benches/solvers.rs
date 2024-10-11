use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use pg_graph::explicit::solvers::small_progress::SmallProgressSolver;
use pg_graph::load_example;

fn bench_solvers(c: &mut Criterion) -> eyre::Result<()> {
    let mut group = c.benchmark_group("Solvers");
    let benchmark_games = ["ActionConverter.tlsf.ehoa.pg", "amba_decomposed_arbiter_6.tlsf.ehoa.pg"];

    for i in benchmark_games {
        let example = get_example_dir().join(i);
        let parity_game = load_example(example);
        if i == "amba_decomposed_arbiter_6.tlsf.ehoa.pg" {
            group.sample_size(30);
        }
        group.bench_with_input(BenchmarkId::new("SmallProgress", i), i, |b, i| {
            b.iter(|| {
                let mut game = SmallProgressSolver::new(&parity_game);

                game.run();
            });
        });
    }

    group.finish();

    Ok(())
}

fn get_example_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("game_examples")
}

criterion_group!(benches, bench_solvers);
criterion_main!(benches);
