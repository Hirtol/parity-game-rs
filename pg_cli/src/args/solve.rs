use std::path::PathBuf;

use pg_graph::{
    explicit::{ParityGraph, register_game::RegisterGame},
    Owner,
    visualize::MermaidWriter,
};

#[derive(clap::Args, Debug)]
pub struct SolveCommand {
    /// Which solver to use, by default will use SmallProgressMeasure
    #[clap(subcommand)]
    solver: Option<Solver>,
    /// The `.pg` file to load
    game_path: PathBuf,
    /// Whether to print which vertices are won by which player.
    #[clap(short)]
    print_solution: bool,
    /// Export the solution to a Mermaid.js graph where the vertices are coloured according to their winner
    /// Green = Even, Red = Odd
    #[clap(short = 'm')]
    solution_mermaid: Option<PathBuf>,
    /// Whether to first convert the given game into an explicit expanded `k`-register game.
    ///
    /// Creates a `k`-register-index game, upper bound for valid results is `1 + log(n)`
    /// Provide `k = 0` to use the upper-bound k for correct results.
    /// Default assumes `k = 1 + log(n)`, where `n` is the amount of vertices in the parity game.
    #[clap(short)]
    register_game_k: Option<u32>,
}

#[derive(clap::Subcommand, Debug)]
pub enum Solver {
    /// Use the traditional small progress measure algorithm
    Spm,
    /// Use the recursive Zielonka algorithm
    Zielonka,
    #[clap(alias = "syz")]
    SymbolicZielonka,
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
            out
        }
    };
}

impl SolveCommand {
    #[tracing::instrument(name="Solve Parity Game",skip(self), fields(path=?self.game_path))]
    pub fn run(self) -> eyre::Result<()> {
        let parity_game = crate::utils::load_parity_game(&self.game_path)?;
        let register_game = if let Some(k) = self.register_game_k {
            let k = k as u8;

            tracing::debug!(k, "Constructing with register index");
            let register_game = timed_solve!(
                RegisterGame::construct_2021(&parity_game, k, Owner::Even),
                "Constructed Register Game"
            );

            Some((register_game.to_game()?, register_game))
        } else {
            None
        };

        let mermaid_output = self
            .solution_mermaid
            .map(|out| (out, MermaidWriter::write_mermaid(&parity_game).unwrap()));
        let solver = self.solver.unwrap_or(Solver::Spm);
        tracing::info!(?solver, "Using solver");

        let game_to_solve = if let Some((rg_pg, rg)) = &register_game {
            tracing::debug!(
                from_vertex = rg.original_game.vertex_count(),
                to_vertex = rg_pg.vertex_count(),
                ratio = rg_pg.vertex_count() / rg.original_game.vertex_count(),
                "Converted from PG to RG PG"
            );
            rg_pg
        } else {
            &parity_game
        };

        let solution = match solver {
            Solver::Spm => {
                let mut solver = pg_graph::explicit::solvers::small_progress::SmallProgressSolver::new(game_to_solve);

                timed_solve!(solver.run())
            }
            Solver::Zielonka => {
                let mut solver = pg_graph::explicit::solvers::zielonka::ZielonkaSolver::new(game_to_solve);

                let out = timed_solve!(solver.run());
                tracing::info!(n = solver.recursive_calls, "Solved with recursive calls");
                out
            }
            Solver::SymbolicZielonka => {
                let symbolic_game = timed_solve!(
                    pg_graph::symbolic::SymbolicParityGame::from_explicit(game_to_solve),
                    "Constructed Symbolic PG"
                )?;
                let mut solver =
                    pg_graph::symbolic::solvers::symbolic_zielonka::SymbolicZielonkaSolver::new(&symbolic_game);

                let out = timed_solve!(solver.run());
                tracing::info!(n = solver.recursive_calls, "Solved with recursive calls");
                out
            }
        };

        let solution = if let Some((_, rg)) = register_game {
            rg.project_winners_original(&solution.winners)
        } else {
            solution.winners
        };

        let (even_wins, odd_wins) = solution.iter().fold((0, 0), |acc, win| match win {
            Owner::Even => (acc.0 + 1, acc.1),
            Owner::Odd => (acc.0, acc.1 + 1),
        });
        tracing::info!(even_wins, odd_wins, "Results");

        if self.print_solution {
            tracing::info!("Solution: {:?}", solution);
        }

        if let Some((out_path, mut mermaid)) = mermaid_output {
            use std::fmt::Write;
            for (v_id, winner) in solution.iter().enumerate() {
                let fill = match winner {
                    Owner::Even => "#013220",
                    Owner::Odd => "#b10000",
                };
                writeln!(&mut mermaid, "style {v_id} fill:{fill}")?;
            }

            std::fs::write(out_path, mermaid)?;
        }

        Ok(())
    }
}
