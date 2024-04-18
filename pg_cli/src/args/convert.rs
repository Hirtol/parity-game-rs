use std::path::PathBuf;
use pg_graph::Owner;
use pg_graph::register_game::RegisterGame;

#[derive(clap::Args, Debug)]
pub struct ConvertCommand {
    /// The `.pg` file to load
    game_path: PathBuf,
    /// What the given game should be converted into.
    #[clap(subcommand)]
    goal: ConversionGoal,
    /// Write the visualised result of the `goal` conversion into a Mermaid.js file
    #[clap(short, long)]
    mermaid_path: Option<PathBuf>,
    /// Write the result of the `goal` conversion into a `.pg` file
    #[clap(short, long)]
    pg_path: Option<PathBuf>,
}

#[derive(clap::Subcommand, Debug)]
pub enum ConversionGoal {
    /// Keep the current parity game
    #[clap(name = "pg")]
    ParityGame,
    /// Convert the given parity game into a register game.
    #[clap(name = "rg")]
    RegisterGame {
        /// Create a `k`-register-index game, upper bound for valid results is `1 + log(n)`
        ///
        /// Default assumes `k = 1 + log(n)`, where `n` is the amount of vertices in the parity game.
        #[clap(short)]
        k: Option<u32>
    }
}

impl ConvertCommand {
    #[tracing::instrument(name="Convert Parity Game", skip(self), fields(path=?self.game_path, goal=?self.goal))]
    pub fn run(self) -> eyre::Result<()> {
        let parity_game = crate::utils::load_parity_game(&self.game_path)?;
        
        match self.goal {
            ConversionGoal::ParityGame => {
                if let Some(path) = self.mermaid_path {
                    std::fs::write(&path, parity_game.to_mermaid())?;

                    tracing::info!(?path, "Wrote Mermaid graph to path")
                }

                if let Some(path) = self.pg_path {
                    std::fs::write(&path, parity_game.to_pg())?;

                    tracing::info!(?path, "Wrote PG graph to path")
                }
            }
            ConversionGoal::RegisterGame { k } => {
                let k = k.unwrap_or_else(|| 1 + parity_game.vertex_count().ilog2()) as u8;
                let had_nodes = parity_game.vertex_count();
                let register_game = RegisterGame::construct(&parity_game, k, Owner::Even);
                
                if let Some(path) = self.mermaid_path {
                    std::fs::write(&path, register_game.to_mermaid())?;
                    
                    tracing::info!(?path, "Wrote Mermaid graph to path")
                }
                
                if let Some(path) = self.pg_path {
                    let game = register_game.to_game()?;

                    tracing::debug!(from_vertex=had_nodes, to_vertex=game.vertex_count(), ratio=game.vertex_count() / had_nodes, "Converted from PG to RG PG");
                    
                    std::fs::write(&path, game.to_pg())?;
                    
                    tracing::info!(?path, "Wrote PG graph to path")
                }
            }
        }
        
        Ok(())
    }
}