use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use eyre::eyre;
use itertools::Itertools;
use serde_with::{serde_as, DurationSecondsWithFrac};
use pg_graph::{Owner, ParityGame, ParityGraph};
use pg_graph::register_game::{Rank, RegisterGame};
use pg_graph::solvers::zielonka::ZielonkaSolver;

#[derive(clap::Args, Debug)]
pub struct BenchCommand {
    /// Where to deposit the csv file
    out_path: PathBuf,
    /// The path to the directory with games to benchmark.
    benchmark_games: PathBuf,
}

#[serde_as]
#[derive(serde::Serialize, Debug)]
struct CsvOutput {
    name: String,
    reg_index: Rank,
    /// Total time to construct all register games (including all `k` up to reg_index)
    #[serde_as(as="DurationSecondsWithFrac<f64>")]
    time_to_construct: Duration,
    /// Time to construct just R_even for the register index.
    #[serde_as(as="DurationSecondsWithFrac<f64>")]
    final_construct_time: Duration,
    /// Time to solve the parity game with just Zielonka
    #[serde_as(as="DurationSecondsWithFrac<f64>")]
    solve_time_zielonka: Duration,
    /// Time to solve the register game with Zielonka.
    #[serde_as(as="DurationSecondsWithFrac<f64>")]
    solve_time_rg_zielonka: Duration,
    register_game_states: usize,
    parity_game_states: usize,
    rg_pg_ratio: f64,
}

macro_rules! timed_solve {
    ($to_run:expr) => {
        timed_solve!($to_run, "Solving done")
    };
    ($to_run:expr, $text:expr) => {
        {
            let now = std::time::Instant::now();
            let out = $to_run;
            tracing::info!(elapsed=?now.elapsed(), $text);
            (out, now.elapsed())
        }
    };
}

impl BenchCommand {
    #[tracing::instrument(name="Benchmark Register Games", skip(self), fields(path=?self.benchmark_games))]
    pub fn run(self) -> eyre::Result<()> {
        let games = std::fs::read_dir(&self.benchmark_games)?
            .flatten()
            .filter(|f| f.file_name().to_string_lossy().ends_with("pg") || f.file_name().to_string_lossy().ends_with("gm"))
            .collect_vec();

        let mut output = Vec::with_capacity(games.len());

        #[tracing::instrument(name="Run benchmark")]
        fn run_game(game: &Path) -> eyre::Result<CsvOutput> {
            let text = std::fs::read_to_string(game)?;
            let pg = pg_parser::parse_pg(&mut text.as_str()).map_err(|e| eyre!(e))?;
            let parity_game = ParityGame::new(pg)?;
            let start_time = Instant::now();

            let mut final_k = 0;
            let parity_game_states = parity_game.vertex_count();
            let mut even_register_game_states = 0;
            let mut final_solve_output = Vec::new();

            let mut final_even_construct_duration = None;
            let mut final_rg_solve_event_duration = None;

            for k in 0..3 {
                let (odd_game, time_odd) = timed_solve!(RegisterGame::construct_2021(&parity_game, k, Owner::Odd), "Constructed Odd register game");
                let rg_pg = odd_game.to_game()?;
                let mut solver = ZielonkaSolver::new(&rg_pg);
                let (output_odd, time_solve_odd) = timed_solve!(solver.run(), "Solved Zielonka on Odd register game");
                let solution_rg_odd = odd_game.project_winners_original(&output_odd.winners);

                let (even_game, time_even) = timed_solve!(RegisterGame::construct_2021(&parity_game, k, Owner::Even), "Constructed Even register game");
                let rg_pg = even_game.to_game()?;
                let mut solver = ZielonkaSolver::new(&rg_pg);
                let (output_even, time_solve_even) = timed_solve!(solver.run(), "Solved Zielonka on Even register game");
                let solution_rg_even = even_game.project_winners_original(&output_even.winners);

                if solution_rg_even == solution_rg_odd {
                    tracing::debug!(k, "Found register-index for game");
                    final_k = k;
                    even_register_game_states = even_game.vertices.len();
                    final_even_construct_duration = Some(time_even);
                    final_rg_solve_event_duration = Some(time_solve_even);
                    final_solve_output = solution_rg_even;
                    break;
                } else {
                    tracing::debug!(k, "Mismatched solutions, skipping to next k")
                }
            };

            let construct_time = start_time.elapsed();

            let mut solver = ZielonkaSolver::new(&parity_game);
            let (raw_solution, raw_solve_time) = timed_solve!(solver.run());

            if raw_solution.winners != final_solve_output {
                tracing::error!(final_k, "Mismatch between Zielonka and Register Game solution!");
                return Err(eyre!("Mismatched solutions"));
            }

            Ok(CsvOutput {
                name: game.file_stem().unwrap_or_default().to_string_lossy().into_owned(),
                reg_index: final_k,
                time_to_construct: construct_time,
                final_construct_time: final_even_construct_duration.unwrap(),
                solve_time_zielonka: raw_solve_time,
                solve_time_rg_zielonka: final_rg_solve_event_duration.unwrap(),
                register_game_states: even_register_game_states,
                parity_game_states,
                rg_pg_ratio: even_register_game_states as f64 / parity_game_states as f64,
            })
        }

        let mut writer = csv::WriterBuilder::default().has_headers(true).from_path(&self.out_path)?;

        for game in games {
            let (output_run, _) = timed_solve!(run_game(&game.path()));

            match output_run {
                Ok(out) => {
                    writer.serialize(&out)?;
                    output.push(out);
                    writer.flush()?;
                }
                Err(e) => {
                    tracing::error!(?game, ?e, "Failed to run benchmark")
                }
            }
        }

        Ok(())
    }
}