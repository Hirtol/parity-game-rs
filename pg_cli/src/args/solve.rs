use std::path::PathBuf;
use pg_graph::Owner;
use pg_graph::solvers::register_game::RegisterGame;

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
}

#[derive(clap::Subcommand, Debug)]
pub enum Solver {
    /// Use the traditional small progress measure algorithm
    Spm,
    /// First convert the PG to a Register Game, and then use SPM to solve it.
    #[clap(name = "rgspm")]
    RegisterGameSpm {
        /// Create a `k`-register-index game, upper bound for valid results is `1 + log(n)`
        ///
        /// Default assumes `k = 1 + log(n)`, where `n` is the amount of vertices in the parity game.
        #[clap(short)]
        k: Option<u32>
    }
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

        tracing::info!(size=parity_game.vertex_count(), "Loaded parity game");
        let solver = self.solver.unwrap_or(Solver::Spm);
        tracing::info!(?solver, "Using solver");

        let solution = match solver {
            Solver::Spm => {
                let mut solver = pg_graph::solvers::small_progress::SmallProgressSolver::new(parity_game);

                timed_solve!(solver.run())
            }
            Solver::RegisterGameSpm{
                k
            } => {
                let k = k.unwrap_or_else(|| 1 + parity_game.vertex_count().ilog10()) as u8;
                tracing::debug!(k, "Constructing with register index");
                let register_game = timed_solve!(RegisterGame::construct(parity_game, k, Owner::Even), "Constructed Register Game");
                let game = register_game.to_game()?;
                let mut solver = pg_graph::solvers::small_progress::SmallProgressSolver::new(game);

                let solution = timed_solve!(solver.run());

                register_game.project_winners_original(&solution)
            }
        };

        let (even_wins, odd_wins) = solution.iter().fold((0, 0), |acc, win| {
            match win {
                Owner::Even => (acc.0 + 1, acc.1),
                Owner::Odd => (acc.0, acc.1 + 1)
            }
        });
        tracing::info!(even_wins, odd_wins, "Results");

        if self.print_solution {
            tracing::info!("Solution: {:?}", solution);
        }

        Ok(())
    }
}