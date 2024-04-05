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

impl SolveCommand {
    #[tracing::instrument]
    pub fn run(self) -> eyre::Result<()> {
        let parity_game = crate::utils::load_parity_game(&self.game_path)?;
        
        tracing::info!(size=parity_game.vertex_count(), "Loaded parity game");
        tracing::info!(?self.solver, "Using solver");
        
        let solution = match self.solver.unwrap_or(Solver::Spm) {
            Solver::Spm => {
                let mut solver = pg_graph::solvers::small_progress::SmallProgressSolver::new(parity_game);
                
                solver.run()
            }
            Solver::RegisterGameSpm{
                k
            } => {
                let k = k.unwrap_or_else(|| 1 + parity_game.vertex_count().ilog10()) as u8;
                let register_game = RegisterGame::construct(parity_game, k, Owner::Even);
                let game = register_game.to_game()?;
                let mut solver = pg_graph::solvers::small_progress::SmallProgressSolver::new(game);

                let solution = solver.run();
                
                register_game.project_winners_original(&solution)
            }
        };
        
        if self.print_solution {
            println!("Solution: {:?}", solution);
        }
        
        Ok(())
    }
}

