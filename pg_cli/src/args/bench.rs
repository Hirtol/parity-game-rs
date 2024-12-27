use ahash::HashMap;
use eyre::{eyre, ContextCompat};
use itertools::Itertools;
use pg_graph::{
    explicit::{
        register_game::{Rank, RegisterGame},
        solvers::{zielonka::ZielonkaSolver, SolverOutput},
        ParityGame, ParityGraph,
    },
    symbolic::{
        register_game::SymbolicRegisterGame, solvers::symbolic_register_zielonka::SymbolicRegisterZielonkaSolver, BDD,
    },
    Owner,
};
use regex::Regex;
use serde_with::{serde_as, DurationSecondsWithFrac};
use std::fmt::{Formatter, Write};
use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use sysinfo::ProcessesToUpdate;

#[derive(clap::Args, Debug)]
pub struct BenchCommand {
    /// Where to deposit the csv file
    out_path: PathBuf,
    /// The path to the directory with games to benchmark.
    benchmark_games: PathBuf,
    /// The objective of the benchmark
    #[clap(subcommand)]
    objective: BenchmarkObjective,
}

#[derive(clap::Subcommand, Debug)]
pub enum BenchmarkObjective {
    /// Register game benchmark
    Rg,
    /// Symbolic register game benchmark
    Srg,
    /// Normal algorithm benchmarks
    Algo,
    /// External algorithm benchmarks
    AlgoExt {
        #[clap(default_value = "temp_dir")]
        temp_dir: PathBuf,
        #[clap(default_value = "logs")]
        logs_dir: PathBuf,
        #[clap(long = "exe")]
        cli_path: PathBuf,
        /// Timeout in seconds
        #[clap(long, default_value = "50.0")]
        timeout: f64,
        /// Register Game Benches
        #[clap(long)]
        rg: bool,
    },
    RgConstruct {
        #[clap(default_value = "temp_dir")]
        temp_dir: PathBuf,
        #[clap(default_value = "logs")]
        logs_dir: PathBuf,
        #[clap(long = "exe")]
        cli_path: PathBuf,
        /// Timeout in seconds
        #[clap(long, default_value = "50.0")]
        timeout: f64,
        /// CSV file which contains the register indexes.
        #[clap(default_value = "r_index.csv")]
        index_map: PathBuf,
    }
}

#[serde_as]
#[derive(serde::Serialize, Debug)]
struct RgCsvOutput {
    name: String,
    reg_index: Rank,
    /// Total time to construct all register games (including all `k` up to reg_index)
    #[serde_as(as = "DurationSecondsWithFrac<f64>")]
    time_to_construct: Duration,
    /// Time to construct just R_even for the register index.
    #[serde_as(as = "DurationSecondsWithFrac<f64>")]
    final_construct_time: Duration,
    /// Time to solve the parity game with just Zielonka
    #[serde_as(as = "DurationSecondsWithFrac<f64>")]
    solve_time_zielonka: Duration,
    /// Time to solve the register game with Zielonka.
    #[serde_as(as = "DurationSecondsWithFrac<f64>")]
    solve_time_rg_zielonka: Duration,
    register_game_states: usize,
    parity_game_states: usize,
    rg_pg_ratio: f64,
}

#[serde_as]
#[derive(serde::Serialize, Debug)]
struct AlgoRgCsvOutput {
    name: String,
    reg_index: Option<Rank>,
    /// Total time to run symbolic parametrised Lehtinen
    #[serde_as(as = "DurationSecondsWithFrac<f64>")]
    srg_lehtinen_solve: Duration,
    /// Total time to run explicit parametrised Lehtinen
    #[serde_as(as = "DurationSecondsWithFrac<f64>")]
    erg_lehtinen_solve: Duration,
    #[serde_as(as = "DurationSecondsWithFrac<f64>")]
    erg_jit_lehtinen_solve: Duration,
    #[serde_as(as = "DurationSecondsWithFrac<f64>")]
    erg_reduced_lehtinen_solve: Duration,
    parity_game_states: usize,
}

#[serde_as]
#[derive(serde::Serialize, Debug)]
struct AlgoCsvOutput {
    name: String,
    #[serde_as(as = "DurationSecondsWithFrac<f64>")]
    zlk_solve: Duration,
    #[serde_as(as = "DurationSecondsWithFrac<f64>")]
    pp_solve: Duration,
    #[serde_as(as = "DurationSecondsWithFrac<f64>")]
    tl_solve: Duration,
    #[serde_as(as = "DurationSecondsWithFrac<f64>")]
    qlz_solve: Duration,
    #[serde_as(as = "DurationSecondsWithFrac<f64>")]
    qlz_tangle_solve: Duration,
    parity_game_states: usize,
}

macro_rules! timed_solve {
    ($to_run:expr) => {
        timed_solve!($to_run, "Solving done")
    };
    ($to_run:expr, $text:expr) => {
        {
            let now = std::time::Instant::now();
            let out = $to_run;
            let elapsed = now.elapsed();
            tracing::info!(elapsed=?elapsed, $text);
            (out, elapsed)
        }
    };
}

impl BenchCommand {
    #[tracing::instrument(name="Benchmark Games", skip(self))]
    pub fn run(self) -> eyre::Result<()> {
        let games = std::fs::read_dir(&self.benchmark_games)?
            .flatten()
            .filter(|f| {
                f.file_name().to_string_lossy().ends_with("pg")
                    || f.file_name().to_string_lossy().ends_with("gm")
                    || f.file_name().to_string_lossy().ends_with("bz2")
            })
            .collect_vec();

        let mut writer = csv::WriterBuilder::default()
            .has_headers(true)
            .from_path(&self.out_path)?;

        for (i, game) in games.into_iter().enumerate() {
            let output_run = match &self.objective {
                BenchmarkObjective::Rg => {
                    let out = Self::run_rg_bench(&game.path());
                    if let Ok(out) = &out {
                        writer.serialize(out)?;
                    }
                    out.map(|v| ())
                }
                BenchmarkObjective::Srg => {
                    let out = Self::run_game_symbolic(&game.path());
                    if let Ok(out) = &out {
                        writer.serialize(out)?;
                    }
                    out.map(|v| ())
                }
                BenchmarkObjective::Algo => {
                    let out = Self::run_algo_bench(&game.path());
                    if let Ok(out) = &out {
                        tracing::info!("Completed game, results: {out:#?}");
                        writer.serialize(out)?;
                    }
                    out.map(|v| ())
                }
                BenchmarkObjective::AlgoExt { temp_dir, logs_dir, cli_path, timeout, rg } => {
                    std::fs::create_dir_all(temp_dir)?;
                    std::fs::create_dir_all(logs_dir)?;

                    if *rg {
                        let out = Self::run_rg_bench_external(cli_path, temp_dir, logs_dir, &game.path(), Duration::from_secs_f64(*timeout));
                        if let Ok(out) = &out {
                            tracing::info!("Completed game, results: {out:#?}");
                            writer.serialize(out)?;
                        }
                        out.map(|v| ())
                    } else {
                        let out = Self::run_algo_bench_external(cli_path, temp_dir, logs_dir, &game.path(), Duration::from_secs_f64(*timeout));
                        if let Ok(out) = &out {
                            tracing::info!("Completed game, results: {out:#?}");
                            writer.serialize(out)?;
                        }
                        out.map(|v| ())
                    }
                }
                BenchmarkObjective::RgConstruct { temp_dir, logs_dir, cli_path, timeout, index_map } => {
                    let mut reader = csv::ReaderBuilder::default()
                        .from_path(index_map)?;

                    std::fs::create_dir_all(temp_dir)?;
                    std::fs::create_dir_all(logs_dir)?;
                    let Some(r_index) = reader.records().flatten().find(|n| game.file_name().to_string_lossy().contains(n.get(0).unwrap())) else {
                        tracing::warn!("Skipping: {:?} due to missing register index", game);
                        continue;
                    };
                    let k: f32 = r_index.get(1).context("no r-index")?.parse()?;
                    let k = k as u8;

                    Self::run_rg_construct_bench_external(cli_path, temp_dir, logs_dir, &game.path(), Duration::from_secs_f64(*timeout), k)?;
                    Ok(())
                }
            };

            match output_run {
                Ok(_) => {
                    writer.flush()?;
                }
                Err(e) => {
                    std::fs::write(
                        format!("error_{:?}.txt", i),
                        format!("{}\n{e:#?}", game.file_name().to_string_lossy()),
                    )?;
                    tracing::error!(?game, ?e, "Failed to run benchmark")
                }
            }
        }

        Ok(())
    }

    #[tracing::instrument(skip_all, name = "Run benchmark")]
    fn run_algo_bench_external(cli_path: &Path, tmp_dir: &Path, logs_dir: &Path, game: &Path, timeout: Duration) -> eyre::Result<AlgoCsvOutput> {
        use crate::args::solve::ExplicitSolvers::*;
        let algos = [Zielonka, PP, TL, Qlz { tangles: true }, Qlz { tangles: false }];

        tracing::info!(?game, "Loading next parity game");
        let parity_game = pg_graph::load_parity_game(game)?;
        // Serialize the parity game for faster loading
        let temp_pg = tmp_dir.join(game.with_extension("pgrs").file_name().unwrap());
        std::fs::write(&temp_pg, bitcode::serialize(&parity_game)?)?;

        let parity_game_states = parity_game.vertex_count();

        let mut results = ahash::HashMap::default();
        let make_cmd = || {
            let mut cmd = std::process::Command::new(cli_path);
            cmd.arg("s").arg(&temp_pg).arg("explicit").arg("--simple-trace");
            cmd
        };

        // 3 runs for each for consistency
        const RUNS: u32 = 3;
        let mut algos_to_skip = Vec::new();
        
        for i in 0..RUNS {
            // For verification whilst running the benchmarks, just in case.
            let mut previous_solution: Option<(usize, usize)> = None;

            'outer: for algo in &algos {
                if algos_to_skip.contains(algo) {
                    tracing::trace!(?algo, "Skipping due to previous timeout");
                    continue;
                }
                tracing::debug!(?i, ?algo, "Running next algorithm");
                let mut final_cmd = make_cmd();
                match algo {
                    Zielonka => {
                        final_cmd.arg("zielonka");
                    }
                    Qlz { tangles } => {
                        final_cmd.arg("qlz");
                        if *tangles {
                            final_cmd.arg("--tangles");
                        }
                    }
                    PP => {
                        final_cmd.arg("pp");
                    }
                    TL => {
                        final_cmd.arg("tl");
                    }
                    _ => unreachable!(),
                };
                
                let mut child = final_cmd
                    .stderr(std::process::Stdio::piped())
                    .stdout(std::process::Stdio::piped())
                    .spawn()?;

                let now = std::time::Instant::now();

                loop {
                    match child.try_wait() {
                        Ok(Some(status)) => {
                            if !status.success() {
                                let output = child.wait_with_output()?;
                                
                                tracing::error!("Failed to execute {algo:?}, status code: {status:?}, out: {}, err: {}", String::from_utf8(output.stdout)?, String::from_utf8(output.stderr)?);
                                algos_to_skip.push(*algo);
                                continue 'outer;
                            };
                            // Exited successfully
                            break;
                        }
                        Ok(None) => {
                            if now.elapsed() > timeout {
                                tracing::warn!(?timeout, ?algo, ?game, "Timeout reached, skipping and assuming 2 * timeout");
                                let _ = child.kill();
                                algos_to_skip.push(*algo);
                                let to_mut = results.entry(*algo).or_insert_with(Vec::new);
                                // Assume it would've cost twice the time-out to run
                                to_mut.push(RUNS * timeout * 2);
                                continue 'outer;
                            }
                        }
                        Err(err) => {
                            tracing::error!(?err, "Underlying process panicked");
                            eyre::bail!(err);
                        }
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
                let output = child.wait_with_output()?;
                let std_out = String::from_utf8(output.stdout)?;

                let run_log = logs_dir.join(format!("{}_{algo}_n_{i}.txt", game.file_stem().unwrap().to_string_lossy()));
                std::fs::write(run_log, &std_out)?;

                static RESULTS_RX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"Results even_wins=(\d+) odd_wins=(\d+)").unwrap());
                static SOLVING_RX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"Solving done elapsed=(.*)").unwrap());

                let solution = RESULTS_RX.captures(&std_out).context("No results")?;
                let solving = SOLVING_RX.captures(&std_out).context("No solve time")?;

                let output: (usize, usize) = (solution.get(1).context("No even")?.as_str().parse()?, solution.get(2).context("No even")?.as_str().parse()?);

                let took = solving.get(1).context("No solving duration")?;
                // let time_taken = Duration::from_secs_f64(f64::from_str(took.as_str())?);
                let time_taken = parse_duration(took.as_str()).ok_or_else(|| eyre::eyre!("Failed to parse duration: {}", took.as_str()))?;
                
                // Verify solution
                if let Some(previous) = &previous_solution {
                    assert_eq!(previous, &output, "Two solver solutions don't match");
                } else {
                    previous_solution = Some(output);
                }

                let to_mut = results.entry(*algo).or_insert_with(Vec::new);

                to_mut.push(time_taken)
            }

            tracing::debug!("Completed run: {i}")
        }

        std::fs::remove_file(temp_pg)?;

        Ok(AlgoCsvOutput {
            name: game.file_stem().unwrap_or_default().to_string_lossy().into_owned(),
            zlk_solve: results.entry(Zielonka).or_default().iter().sum::<Duration>() / RUNS,
            pp_solve: results.entry(PP).or_default().iter().sum::<Duration>() / RUNS,
            tl_solve: results.entry(TL).or_default().iter().sum::<Duration>() / RUNS,
            qlz_solve: results
                .entry(Qlz { tangles: false })
                .or_default()
                .iter()
                .sum::<Duration>()
                / RUNS,
            qlz_tangle_solve: results
                .entry(Qlz { tangles: true })
                .or_default()
                .iter()
                .sum::<Duration>()
                / RUNS,
            parity_game_states,
        })
    }

    #[tracing::instrument(skip_all, name = "Run benchmark")]
    fn run_rg_bench_external(cli_path: &Path, tmp_dir: &Path, logs_dir: &Path, game: &Path, timeout: Duration) -> eyre::Result<AlgoRgCsvOutput> {
        use crate::args::solve::ExplicitSolvers::*;
        #[derive(PartialOrd, PartialEq, Debug, Copy, Clone, Hash, Ord, Eq)]
        enum Algos {
            Srg,
            Erg,
            ErgReduced,
            ErgJit,
        }
        impl std::fmt::Display for Algos {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                match self {
                    Algos::Srg => f.write_str("srg"),
                    Algos::Erg => f.write_str("erg"),
                    Algos::ErgReduced => f.write_str("ergreduced"),
                    Algos::ErgJit => f.write_str("ergjit"),
                }
            }
        }
        let algos = [Algos::Srg, Algos::ErgReduced];

        tracing::info!(?game, "Loading next parity game");
        let parity_game = pg_graph::load_parity_game(game)?;
        // Serialize the parity game for faster loading
        let temp_pg = tmp_dir.join(game.with_extension("pgrs").file_name().unwrap());
        std::fs::write(&temp_pg, bitcode::serialize(&parity_game)?)?;

        let parity_game_states = parity_game.vertex_count();

        let mut results = ahash::HashMap::default();
        let mut found_index = None;
        let make_cmd = || {
            let mut cmd = std::process::Command::new(cli_path);
            cmd.arg("s").arg(&temp_pg).arg("--simple-trace");
            cmd
        };

        // 3 runs for each for consistency
        const RUNS: u32 = 3;
        let mut algos_to_skip = Vec::new();
        let mut sys = sysinfo::System::new_all();

        for i in 0..RUNS {
            // For verification whilst running the benchmarks, just in case.
            let mut previous_solution: Option<(usize, usize)> = None;

            'outer: for algo in &algos {
                if algos_to_skip.contains(algo) {
                    tracing::trace!(?algo, "Skipping due to previous timeout");
                    continue;
                }
                tracing::debug!(?i, ?algo, "Running next algorithm");
                let mut final_cmd = make_cmd();
                match algo {
                    Algos::Srg => {
                        final_cmd.arg("symbolic")
                            .arg("-r")
                            .arg("param")
                    }
                    Algos::Erg => {
                        final_cmd.arg("explicit")
                            .arg("-r")
                            .arg("param")
                    },
                    Algos::ErgReduced => {
                        final_cmd.arg("explicit")
                            .arg("-r")
                            .arg("param")
                            .arg("-s")
                            .arg("partial-reduced")
                    },
                    Algos::ErgJit => {
                        final_cmd.arg("explicit")
                            .arg("-r")
                            .arg("param")
                            .arg("-s")
                            .arg("reduced")
                    }
                    _ => unreachable!(),
                };

                let mut child = final_cmd
                    .stderr(std::process::Stdio::piped())
                    .stdout(std::process::Stdio::piped())
                    .spawn()?;
                let child_pid = sysinfo::Pid::from_u32(child.id());
                let now = std::time::Instant::now();
                let stdoutput = Arc::new(Mutex::new(Vec::new()));

                let cancel = Arc::new(AtomicBool::new(false));
                let clone = stdoutput.clone();
                let cancel_clone = cancel.clone();
                let mut pipe = child.stdout.take();
                std::thread::spawn(move || {
                    let mut buf = vec![0; 1024];

                    loop {
                        if cancel_clone.load(Ordering::SeqCst) {
                            if let Some(std) = &mut pipe {
                                std.read_to_end(&mut clone.lock().unwrap()).unwrap();
                            }
                            cancel_clone.store(false, Ordering::SeqCst);
                            break;
                        } else {
                            if let Some(std) = &mut pipe {
                                let bytes_read = std.read(&mut buf).unwrap();
                                clone.lock().unwrap().extend_from_slice(&buf[..bytes_read])
                            }
                            std::thread::sleep(Duration::from_millis(10))
                        }
                    }
                });

                loop {
                    match child.try_wait() {
                        Ok(Some(status)) => {
                            if !status.success() {
                                let output = child.wait_with_output()?;

                                tracing::error!("Failed to execute {algo:?}, status code: {status:?}, out: {}, err: {}", String::from_utf8(output.stdout)?, String::from_utf8(output.stderr)?);
                                algos_to_skip.push(*algo);
                                let mut to_mut = results.entry(*algo).insert_entry(Vec::new());
                                // Assume it would've cost twice the time-out to run
                                to_mut.get_mut().push(RUNS * timeout * 2);
                                // Persist the log, wait for our reading thread to finish
                                cancel.store(true, Ordering::SeqCst);
                                while cancel.load(Ordering::SeqCst) {}

                                let mut std_out = String::from_utf8(stdoutput.lock().unwrap().to_vec())?;
                                std_out.push_str("\nPREEMPTIVE EXIT: CRASHED");

                                let run_log = logs_dir.join(format!("{}_{algo}_n_{i}.txt", game.file_stem().unwrap().to_string_lossy()));
                                std::fs::write(run_log, &std_out)?;
                                continue 'outer;
                            };
                            // Exited successfully
                            break;
                        }
                        Ok(None) => {
                            sys.refresh_processes(ProcessesToUpdate::Some(&[child_pid]), true);
                            let oom = if let Some(proc) = sys.process(child_pid) {
                                const MAX_MEM: u64 = 8e9 as u64;
                                let mem_usage = proc.memory();
                                mem_usage >= MAX_MEM
                            } else {
                                false
                            };

                            let timeout_reached = {
                                static SYMBOLIC_RX: LazyLock<regex::bytes::Regex> = LazyLock::new(|| regex::bytes::Regex::new(r"Finished symbolic parity game construction took=(.*)").unwrap());
                                static GC_RX: LazyLock<regex::bytes::Regex> = LazyLock::new(|| regex::bytes::Regex::new(r"Garbage collection elapsed=(.*)").unwrap());
                                static SOLVED_CNT_RX: LazyLock<regex::bytes::Regex> = LazyLock::new(|| regex::bytes::Regex::new(r"Solving done elapsed").unwrap());
                                let stdoutput = stdoutput.lock().unwrap();
                                match algo {
                                    Algos::Srg => if SYMBOLIC_RX.is_match(&stdoutput) && GC_RX.is_match(&stdoutput) {
                                        let mut found_construction_times = SYMBOLIC_RX.captures_iter(&stdoutput)
                                            .flat_map(|m| m.get(1))
                                            .map(|found_constr_time| {
                                                let match_str = std::str::from_utf8(found_constr_time.as_bytes()).unwrap();
                                                parse_duration(match_str).expect("Failed to parse duration")
                                            })
                                            .collect_vec();
                                        let mut garbage_collection_time = GC_RX.captures_iter(&stdoutput)
                                            .flat_map(|m| m.get(1))
                                            .map(|found_gc_time| {
                                                let match_str = std::str::from_utf8(found_gc_time.as_bytes()).unwrap();
                                                parse_duration(match_str).expect("Failed to parse duration")
                                            })
                                            .collect_vec();
                                        let solved_cnt = SOLVED_CNT_RX.find_iter(&stdoutput).count();

                                        // This holds if we're busy creating a new game, then add 1 to match count
                                        // Otherwise just subtract the construction time.
                                        if solved_cnt == found_construction_times.len() {
                                            // Pretend that the current encoding will take a similar amount of time
                                            found_construction_times.push(found_construction_times[0]);
                                            garbage_collection_time.push(garbage_collection_time[0]);
                                        }

                                        // GC of the symbolic tree takes time which we account for here.
                                        let total_time_taken = found_construction_times.iter().sum::<Duration>() + garbage_collection_time.iter().sum();
                                        (now.elapsed().saturating_sub(total_time_taken)) > timeout
                                    } else {
                                        // Can't time out while constructing
                                        false
                                    }
                                    _ => {
                                        now.elapsed() > timeout
                                    }
                                }
                            };

                            if timeout_reached || oom {
                                tracing::warn!(?timeout, ?algo, ?game, "Timeout or OOM reached, skipping and assuming 2 * timeout");
                                let _ = child.kill();
                                algos_to_skip.push(*algo);
                                // Discard whatever used to be in there
                                let mut to_mut = results.entry(*algo).insert_entry(Vec::new());
                                // Assume it would've cost twice the time-out to run
                                to_mut.get_mut().push(RUNS * timeout * 2);
                                // Persist the log, wait for our reading thread to finish
                                cancel.store(true, Ordering::SeqCst);
                                while cancel.load(Ordering::SeqCst) {}

                                let mut std_out = String::from_utf8(stdoutput.lock().unwrap().to_vec())?;
                                if oom {
                                    std_out.push_str("\nPREEMPTIVE EXIT: OOM REACHED");
                                } else if timeout_reached {
                                    std_out.push_str("\nPREEMPTIVE EXIT: TIMEOUT REACHED");
                                }

                                let run_log = logs_dir.join(format!("{}_{algo}_n_{i}.txt", game.file_stem().unwrap().to_string_lossy()));
                                std::fs::write(run_log, &std_out)?;
                                continue 'outer;
                            }
                        }
                        Err(err) => {
                            tracing::error!(?err, "Underlying process panicked");
                            eyre::bail!(err);
                        }
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
                // Persist the log, wait for our reading thread to finish
                cancel.store(true, Ordering::SeqCst);
                while cancel.load(Ordering::SeqCst) {}
                let std_out = String::from_utf8(stdoutput.lock().unwrap().to_vec())?;

                let run_log = logs_dir.join(format!("{}_{algo}_n_{i}.txt", game.file_stem().unwrap().to_string_lossy()));
                std::fs::write(run_log, &std_out)?;

                static RESULTS_RX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"Results even_wins=(\d+) odd_wins=(\d+)").unwrap());
                static SOLVING_RX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"Solving done elapsed=(.*)").unwrap());
                static RG_INDEX_RX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"Discovered register index k=(\d+)").unwrap());

                let solving = SOLVING_RX.captures_iter(&std_out);
                let rg_construct = match algo {
                    Algos::Srg => {
                        static RG_CONSTR_RX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"Finished symbolic register game construction took=(.*)").unwrap());
                        RG_CONSTR_RX.captures_iter(&std_out)
                    }
                    _ => {
                        static RG_CONSTR_RX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"Constructed.*?Register Game elapsed=(.*)").unwrap());
                        RG_CONSTR_RX.captures_iter(&std_out)
                    }
                };
                let solution = RESULTS_RX.captures(&std_out).context("No results")?;
                let rg_index = RG_INDEX_RX.captures(&std_out).context("No solve time")?;

                let output: (usize, usize) = (solution.get(1).context("No even")?.as_str().parse()?, solution.get(2).context("No even")?.as_str().parse()?);

                let mut total_solve_time = Duration::from_secs(0);
                for (constr_time, solve_time) in solving.zip(rg_construct) {
                    let constr_time = constr_time.get(1).expect("No construct time");
                    let solve_time = solve_time.get(1).expect("No solve time");
                    total_solve_time += parse_duration(constr_time.as_str()).ok_or_else(|| eyre::eyre!("Failed to parse duration: {}", constr_time.as_str()))?;
                    total_solve_time += parse_duration(solve_time.as_str()).ok_or_else(|| eyre::eyre!("Failed to parse duration: {}", solve_time.as_str()))?;
                }
                let reg_index = rg_index.get(1).context("No register index found")?.as_str().parse()?;
                found_index = Some(reg_index);

                // Verify solution
                if let Some(previous) = &previous_solution {
                    assert_eq!(previous, &output, "Two solver solutions don't match");
                } else {
                    previous_solution = Some(output);
                }

                let to_mut = results.entry(*algo).or_insert_with(Vec::new);

                to_mut.push(total_solve_time)
            }

            tracing::debug!("Completed run: {i}")
        }

        std::fs::remove_file(temp_pg)?;

        Ok(AlgoRgCsvOutput {
            name: game.file_stem().unwrap_or_default().to_string_lossy().into_owned(),
            reg_index: found_index,
            srg_lehtinen_solve: results.entry(Algos::Srg).or_default().iter().sum::<Duration>() / RUNS,
            erg_lehtinen_solve: results.entry(Algos::Erg).or_default().iter().sum::<Duration>() / RUNS,
            erg_jit_lehtinen_solve: results.entry(Algos::ErgJit).or_default().iter().sum::<Duration>() / RUNS,
            erg_reduced_lehtinen_solve: results.entry(Algos::ErgReduced).or_default().iter().sum::<Duration>() / RUNS,
            parity_game_states,
        })
    }

    /// # Arguments
    /// * `indexes` - Game name -> Register-index map
    #[tracing::instrument(skip_all, name = "Run benchmark")]
    fn run_rg_construct_bench_external(cli_path: &Path, tmp_dir: &Path, logs_dir: &Path, game: &Path, timeout: Duration, k: u8) -> eyre::Result<()> {
        use crate::args::solve::ExplicitSolvers::*;
        #[derive(PartialOrd, PartialEq, Debug, Copy, Clone, Hash, Ord, Eq)]
        enum Algos {
            Srg,
            Erg,
            ErgReduced,
            ErgJit,
        }
        impl std::fmt::Display for Algos {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                match self {
                    Algos::Srg => f.write_str("srg"),
                    Algos::Erg => f.write_str("erg"),
                    Algos::ErgReduced => f.write_str("ergreduced"),
                    Algos::ErgJit => f.write_str("ergjit"),
                }
            }
        }
        let algos = [Algos::Srg, Algos::ErgReduced, Algos::Erg, Algos::ErgJit];

        tracing::info!(?game, "Loading next parity game");
        let parity_game = pg_graph::load_parity_game(game)?;
        // Serialize the parity game for faster loading
        let temp_pg = tmp_dir.join(game.with_extension("pgrs").file_name().unwrap());
        std::fs::write(&temp_pg, bitcode::serialize(&parity_game)?)?;

        let make_cmd = || {
            let mut cmd = std::process::Command::new(cli_path);
            cmd.arg("s").arg(&temp_pg).arg("--simple-trace").arg("-n");
            cmd
        };

        // 3 runs for each for consistency
        const RUNS: u32 = 3;
        let mut algos_to_skip = Vec::new();
        let mut sys = sysinfo::System::new_all();

        for i in 0..RUNS {
            // For verification whilst running the benchmarks, just in case.
            let mut previous_solution: Option<(usize, usize)> = None;

            'outer: for algo in &algos {
                if algos_to_skip.contains(algo) {
                    tracing::trace!(?algo, "Skipping due to previous timeout");
                    continue;
                }
                tracing::debug!(?i, ?algo, "Running next algorithm");
                let mut final_cmd = make_cmd();
                match algo {
                    Algos::Srg => {
                        final_cmd.arg("symbolic")
                            .arg("-r")
                            .arg(format!("{k}"))
                    }
                    Algos::Erg => {
                        final_cmd.arg("explicit")
                            .arg("-r")
                            .arg(format!("{k}"))
                    },
                    Algos::ErgReduced => {
                        final_cmd.arg("explicit")
                            .arg("-r")
                            .arg(format!("{k}"))
                            .arg("-s")
                            .arg("partial-reduced")
                    },
                    Algos::ErgJit => {
                        final_cmd.arg("explicit")
                            .arg("-r")
                            .arg(format!("{k}"))
                            .arg("-s")
                            .arg("reduced")
                    }
                    _ => unreachable!(),
                };

                let mut child = final_cmd
                    .stderr(std::process::Stdio::piped())
                    .stdout(std::process::Stdio::piped())
                    .spawn()?;
                let child_pid = sysinfo::Pid::from_u32(child.id());
                let now = std::time::Instant::now();
                let stdoutput = Arc::new(Mutex::new(Vec::new()));

                let cancel = Arc::new(AtomicBool::new(false));
                let clone = stdoutput.clone();
                let cancel_clone = cancel.clone();
                let mut pipe = child.stdout.take();
                std::thread::spawn(move || {
                    let mut buf = vec![0; 1024];

                    loop {
                        if cancel_clone.load(Ordering::SeqCst) {
                            if let Some(std) = &mut pipe {
                                std.read_to_end(&mut clone.lock().unwrap()).unwrap();
                            }
                            cancel_clone.store(false, Ordering::SeqCst);
                            break;
                        } else {
                            if let Some(std) = &mut pipe {
                                let bytes_read = std.read(&mut buf).unwrap();
                                clone.lock().unwrap().extend_from_slice(&buf[..bytes_read])
                            }
                            std::thread::sleep(Duration::from_millis(10))
                        }
                    }
                });

                loop {
                    match child.try_wait() {
                        Ok(Some(status)) => {
                            if !status.success() {
                                let output = child.wait_with_output()?;

                                tracing::error!("Failed to execute {algo:?}, status code: {status:?}, out: {}, err: {}", String::from_utf8(output.stdout)?, String::from_utf8(output.stderr)?);
                                algos_to_skip.push(*algo);
                                // Persist the log, wait for our reading thread to finish
                                cancel.store(true, Ordering::SeqCst);
                                while cancel.load(Ordering::SeqCst) {}

                                let mut std_out = String::from_utf8(stdoutput.lock().unwrap().to_vec())?;
                                std_out.push_str("\nPREEMPTIVE EXIT: CRASHED");

                                let run_log = logs_dir.join(format!("{}_{algo}_n_{i}.txt", game.file_stem().unwrap().to_string_lossy()));
                                std::fs::write(run_log, &std_out)?;
                                continue 'outer;
                            };
                            // Exited successfully
                            break;
                        }
                        Ok(None) => {
                            sys.refresh_processes(ProcessesToUpdate::Some(&[child_pid]), true);
                            let oom = if let Some(proc) = sys.process(child_pid) {
                                const MAX_MEM: u64 = 8e9 as u64;
                                let mem_usage = proc.memory();
                                mem_usage >= MAX_MEM
                            } else {
                                false
                            };

                            let timeout_reached = {
                                static SYMBOLIC_RX: LazyLock<regex::bytes::Regex> = LazyLock::new(|| regex::bytes::Regex::new(r"Finished symbolic parity game construction took=(.*)").unwrap());
                                static GC_RX: LazyLock<regex::bytes::Regex> = LazyLock::new(|| regex::bytes::Regex::new(r"Garbage collection elapsed=(.*)").unwrap());
                                static SOLVED_CNT_RX: LazyLock<regex::bytes::Regex> = LazyLock::new(|| regex::bytes::Regex::new(r"Solving done elapsed").unwrap());
                                let stdoutput = stdoutput.lock().unwrap();
                                match algo {
                                    Algos::Srg => if SYMBOLIC_RX.is_match(&stdoutput) && GC_RX.is_match(&stdoutput) {
                                        let mut found_construction_times = SYMBOLIC_RX.captures_iter(&stdoutput)
                                            .flat_map(|m| m.get(1))
                                            .map(|found_constr_time| {
                                                let match_str = std::str::from_utf8(found_constr_time.as_bytes()).unwrap();
                                                parse_duration(match_str).expect("Failed to parse duration")
                                            })
                                            .collect_vec();
                                        let mut garbage_collection_time = GC_RX.captures_iter(&stdoutput)
                                            .flat_map(|m| m.get(1))
                                            .map(|found_gc_time| {
                                                let match_str = std::str::from_utf8(found_gc_time.as_bytes()).unwrap();
                                                parse_duration(match_str).expect("Failed to parse duration")
                                            })
                                            .collect_vec();
                                        let solved_cnt = SOLVED_CNT_RX.find_iter(&stdoutput).count();

                                        // This holds if we're busy creating a new game, then add 1 to match count
                                        // Otherwise just subtract the construction time.
                                        if solved_cnt == found_construction_times.len() {
                                            // Pretend that the current encoding will take a similar amount of time
                                            found_construction_times.push(found_construction_times[0]);
                                            garbage_collection_time.push(garbage_collection_time[0]);
                                        }

                                        // GC of the symbolic tree takes time which we account for here.
                                        let total_time_taken = found_construction_times.iter().sum::<Duration>() + garbage_collection_time.iter().sum();
                                        (now.elapsed().saturating_sub(total_time_taken)) > timeout
                                    } else {
                                        // Can't time out while constructing
                                        false
                                    }
                                    _ => {
                                        now.elapsed() > timeout
                                    }
                                }
                            };

                            if timeout_reached || oom {
                                tracing::warn!(?timeout, ?algo, ?game, "Timeout or OOM reached, skipping and assuming 2 * timeout");
                                let _ = child.kill();
                                algos_to_skip.push(*algo);
                                // Persist the log, wait for our reading thread to finish
                                cancel.store(true, Ordering::SeqCst);
                                while cancel.load(Ordering::SeqCst) {}

                                let mut std_out = String::from_utf8(stdoutput.lock().unwrap().to_vec())?;
                                if oom {
                                    std_out.push_str("\nPREEMPTIVE EXIT: OOM REACHED");
                                } else if timeout_reached {
                                    std_out.push_str("\nPREEMPTIVE EXIT: TIMEOUT REACHED");
                                }

                                let run_log = logs_dir.join(format!("{}_{algo}_n_{i}.txt", game.file_stem().unwrap().to_string_lossy()));
                                std::fs::write(run_log, &std_out)?;
                                continue 'outer;
                            }
                        }
                        Err(err) => {
                            tracing::error!(?err, "Underlying process panicked");
                            eyre::bail!(err);
                        }
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
                // Persist the log, wait for our reading thread to finish
                cancel.store(true, Ordering::SeqCst);
                while cancel.load(Ordering::SeqCst) {}
                let std_out = String::from_utf8(stdoutput.lock().unwrap().to_vec())?;

                let run_log = logs_dir.join(format!("{}_{algo}_n_{i}.txt", game.file_stem().unwrap().to_string_lossy()));
                std::fs::write(run_log, &std_out)?;
            }

            tracing::debug!("Completed run: {i}")
        }

        std::fs::remove_file(temp_pg)?;

        Ok(())
    }

    #[tracing::instrument(name = "Run benchmark")]
    fn run_algo_bench(game: &Path) -> eyre::Result<AlgoCsvOutput> {
        use crate::args::solve::ExplicitSolvers::*;
        let algos = [Zielonka, PP, TL, Qlz { tangles: true }, Qlz { tangles: false }];
        let mut timeouts = ahash::HashMap::default();

        timeouts.entry(TL).or_insert_with(Vec::new).push("two_counters_14");
        timeouts.entry(TL).or_insert_with(Vec::new).push("two_counters_14p");

        let parity_game = pg_graph::load_parity_game(game)?;

        let parity_game_states = parity_game.vertex_count();

        let mut results = ahash::HashMap::default();

        // 3 runs for each for consistency
        const RUNS: u32 = 3;
        for i in 0..RUNS {
            // For verification whilst running the benchmarks, just in case.
            let mut previous_solution: Option<SolverOutput> = None;

            for algo in &algos {
                if timeouts
                    .entry(*algo)
                    .or_default()
                    .iter()
                    .any(|excl| game.to_string_lossy().contains(excl))
                {
                    tracing::info!("Skipping {algo:?} due to know timeout");
                    continue;
                }

                let (output, took) = match algo {
                    Zielonka => {
                        let mut solver = pg_graph::explicit::solvers::zielonka::ZielonkaSolver::new(&parity_game);
                        timed_solve!(solver.run(), "Solved Zielonka")
                    }
                    Qlz { tangles } => {
                        if *tangles {
                            let mut solver =
                                pg_graph::explicit::solvers::qpt_tangle_liverpool::TLZSolver::new(&parity_game);
                            timed_solve!(solver.run(), "Solved QLZ-tangle")
                        } else {
                            let mut solver =
                                pg_graph::explicit::solvers::qpt_liverpool::LiverpoolSolver::new(&parity_game);
                            timed_solve!(solver.run(), "Solved QLZ")
                        }
                    }
                    PP => {
                        let mut solver = pg_graph::explicit::solvers::priority_promotion::PPSolver::new(&parity_game);
                        timed_solve!(solver.run(), "Solved Priority Promotion")
                    }
                    TL => {
                        let mut solver = pg_graph::explicit::solvers::tangle_learning::TangleSolver::new(&parity_game);
                        timed_solve!(solver.run(), "Solved Tangle Learning")
                    }
                    _ => unreachable!(),
                };

                // Verify solution
                if let Some(previous) = &previous_solution {
                    assert_eq!(previous.winners, output.winners, "Two solver solutions don't match");
                } else {
                    previous_solution = Some(output);
                }

                let to_mut = results.entry(*algo).or_insert_with(Vec::new);

                to_mut.push(took)
            }

            tracing::debug!("Completed run: {i}")
        }

        Ok(AlgoCsvOutput {
            name: game.file_stem().unwrap_or_default().to_string_lossy().into_owned(),
            zlk_solve: results.entry(Zielonka).or_default().iter().sum::<Duration>() / RUNS,
            pp_solve: results.entry(PP).or_default().iter().sum::<Duration>() / RUNS,
            tl_solve: results.entry(TL).or_default().iter().sum::<Duration>() / RUNS,
            qlz_solve: results
                .entry(Qlz { tangles: false })
                .or_default()
                .iter()
                .sum::<Duration>()
                / RUNS,
            qlz_tangle_solve: results
                .entry(Qlz { tangles: true })
                .or_default()
                .iter()
                .sum::<Duration>()
                / RUNS,
            parity_game_states,
        })
    }

    #[tracing::instrument(name = "Run benchmark")]
    fn run_rg_bench(game: &Path) -> eyre::Result<RgCsvOutput> {
        let parity_game = pg_graph::load_parity_game(game)?;

        let start_time = Instant::now();

        let mut final_k = 0;
        let parity_game_states = parity_game.vertex_count();
        let mut even_register_game_states = 0;
        let mut final_solve_output = Vec::new();

        let mut final_even_construct_duration = None;
        let mut final_rg_solve_event_duration = None;

        for k in 0..3 {
            let (odd_game, time_odd) = timed_solve!(
                RegisterGame::construct_2021(&parity_game, k, Owner::Odd),
                "Constructed Odd register game"
            );

            let rg_pg = odd_game.to_normal_game()?;
            let mut solver = pg_graph::explicit::solvers::qpt_zielonka::ZielonkaSolver::new(&rg_pg);
            let (output_odd, time_solve_odd) = timed_solve!(solver.run(), "Solved Zielonka on Odd register game");
            let solution_rg_odd = odd_game.project_winners_original(&output_odd.winners);
            let (even_wins, odd_wins) = solution_rg_odd.iter().fold((0, 0), |acc, win| match win {
                Owner::Even => (acc.0 + 1, acc.1),
                Owner::Odd => (acc.0, acc.1 + 1),
            });
            tracing::info!(even_wins, odd_wins, "Results Odd");

            let (even_game, time_even) = timed_solve!(
                RegisterGame::construct_2021(&parity_game, k, Owner::Even),
                "Constructed Even register game"
            );
            let rg_pg = even_game.to_normal_game()?;
            let mut solver = pg_graph::explicit::solvers::qpt_zielonka::ZielonkaSolver::new(&rg_pg);
            let (output_even, time_solve_even) = timed_solve!(solver.run(), "Solved Zielonka on Even register game");
            let solution_rg_even = even_game.project_winners_original(&output_even.winners);
            let (even_wins, odd_wins) = solution_rg_even.iter().fold((0, 0), |acc, win| match win {
                Owner::Even => (acc.0 + 1, acc.1),
                Owner::Odd => (acc.0, acc.1 + 1),
            });
            tracing::info!(even_wins, odd_wins, "Results Even");

            if solution_rg_even == solution_rg_odd {
                tracing::debug!(k, "Found register-index for game");
                final_k = k;
                even_register_game_states = rg_pg.vertex_count();
                final_even_construct_duration = Some(time_even);
                final_rg_solve_event_duration = Some(time_solve_even);
                final_solve_output = solution_rg_even;
                break;
            } else {
                tracing::debug!(k, "Mismatched solutions, skipping to next k")
            }
        }

        let construct_time = start_time.elapsed();

        let mut solver = ZielonkaSolver::new(&parity_game);
        let (raw_solution, raw_solve_time) = timed_solve!(solver.run());

        if raw_solution.winners != final_solve_output {
            tracing::error!(final_k, raw_winners=?raw_solution, ?final_solve_output, "Mismatch between Zielonka and Register Game solution!");
            return Err(eyre!("Mismatched solutions"));
        }

        Ok(RgCsvOutput {
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

    #[tracing::instrument(name = "Run benchmark")]
    fn run_game_symbolic(game: &Path) -> eyre::Result<RgCsvOutput> {
        let parity_game = pg_graph::load_parity_game(game)?;

        let start_time = Instant::now();

        let mut final_k = 0;
        let parity_game_states = parity_game.vertex_count();
        let mut even_register_game_states = 0;
        let mut final_solve_output = Vec::new();

        let mut final_even_construct_duration = None;
        let mut final_rg_solve_event_duration = None;

        for k in 0..3 {
            let (odd_game, time_odd) = timed_solve!(
                SymbolicRegisterGame::<BDD>::from_symbolic(&parity_game, k, Owner::Odd)?,
                "Constructed Odd register game"
            );
            let mut solver = SymbolicRegisterZielonkaSolver::new(&odd_game);

            // let rg_pg = odd_game.to_game()?;
            // let mut solver = ZielonkaSolver::new(&rg_pg);
            // let (output_odd, time_solve_odd) = timed_solve!(solver.run(), "Solved Zielonka on Odd register game");
            let (solution_rg_odd, time_solve_odd) = timed_solve!(solver.run(), "Solved Zielonka on Odd register game");
            // let solution_rg_odd = odd_game.project_winners_original(&output_odd.winners);
            let (even_wins, odd_wins) = solution_rg_odd.winners.iter().fold((0, 0), |acc, win| match win {
                Owner::Even => (acc.0 + 1, acc.1),
                Owner::Odd => (acc.0, acc.1 + 1),
            });
            tracing::info!(even_wins, odd_wins, "Results Odd");

            // let (even_game, time_even) = timed_solve!(
            //     RegisterGame::construct_2021(&parity_game, k, Owner::Even),
            //     "Constructed Even register game"
            // );
            let (even_game, time_even) = timed_solve!(
                SymbolicRegisterGame::<BDD>::from_symbolic(&parity_game, k, Owner::Even)?,
                "Constructed Even register game"
            );
            // let rg_pg = even_game.to_game()?;
            // let mut solver = ZielonkaSolver::new(&rg_pg);
            // let (output_even, time_solve_even) =
            //     timed_solve!(solver.run(), "Solved Zielonka on Even register game");
            // let solution_rg_even = even_game.project_winners_original(&output_even.winners);
            let mut solver = SymbolicRegisterZielonkaSolver::new(&even_game);
            let (solution_rg_even, time_solve_even) =
                timed_solve!(solver.run(), "Solved Zielonka on Even register game");
            let (even_wins, odd_wins) = solution_rg_even.winners.iter().fold((0, 0), |acc, win| match win {
                Owner::Even => (acc.0 + 1, acc.1),
                Owner::Odd => (acc.0, acc.1 + 1),
            });
            tracing::info!(even_wins, odd_wins, "Results Even");

            if solution_rg_even.winners == solution_rg_odd.winners {
                tracing::debug!(k, "Found register-index for game");
                final_k = k;
                even_register_game_states = even_game.to_symbolic_parity_game()?.vertex_count();
                final_even_construct_duration = Some(time_even);
                final_rg_solve_event_duration = Some(time_solve_even);
                final_solve_output = solution_rg_even.winners;
                break;
            } else {
                tracing::debug!(k, "Mismatched solutions, skipping to next k")
            }
        }

        let construct_time = start_time.elapsed();

        let mut solver = ZielonkaSolver::new(&parity_game);
        let (raw_solution, raw_solve_time) = timed_solve!(solver.run());

        if raw_solution.winners != final_solve_output {
            tracing::error!(final_k, raw_winners=?raw_solution, ?final_solve_output, "Mismatch between Zielonka and Register Game solution!");
            return Err(eyre!("Mismatched solutions"));
        }

        Ok(RgCsvOutput {
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
}

/// Terrible duration parser, but makes our lives easier.
fn parse_duration(s: &str) -> Option<Duration> {
    if s.ends_with("ms") {
        if let Some(ms_str) = s.strip_suffix("ms") {
            let millis_parsed = ms_str.parse::<f64>().ok()? * 1_000_000.;
            Some(Duration::from_nanos(millis_parsed as u64))
        } else {
            None
        }
    } else if s.ends_with("µs") {
        if let Some(micros_str) = s.strip_suffix("µs") {
            let micros_parsed = micros_str.parse::<f64>().ok()? * 1_000.;
            Some(Duration::from_nanos(micros_parsed as u64))
        } else {
            None
        }
    } else if s.ends_with("ns") {
        if let Some(nanos_str) = s.strip_suffix("ns") {
            let nanos_parsed = nanos_str.parse::<f64>().ok()?;
            Some(Duration::from_nanos(nanos_parsed as u64))
        } else {
            None
        }
    }else if s.ends_with('s') {
        if let Some(s_seconds) = s.strip_suffix("s") {
            let secs_parsed = s_seconds.parse::<f64>().ok()?;
            Some(Duration::from_secs_f64(secs_parsed))
        } else {
            None
        }
    } else {
        None
    }
}

// static SOLUTION_RX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"Solution:\s*\[(.*?)\]").unwrap());
//                 let solution_full = SOLUTION_RX.captures(&std_out).context("No solution")?;
//                 let sol_full = solution_full.get(1).context("No solution")?
//                     .as_str()
//                     .split(',')
//                     .map(|s| s.trim())
//                     .filter_map(|v| {
//                         match v {
//                             "Even" => Some(Owner::Even),
//                             "Odd" => Some(Owner::Odd),
//                             _ => None
//                         }
//                     }).collect_vec();
//                 tracing::info!("Full solution: {sol_full:?}");