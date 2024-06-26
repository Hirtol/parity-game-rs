use std::path::PathBuf;

use pg_graph::{
    explicit::{ParityGraph, register_game::RegisterGame},
    Owner,
    symbolic::{register_game::SymbolicRegisterGame, SymbolicParityGame},
    visualize::{DotWriter, MermaidWriter, VisualRegisterGame},
};
use pg_graph::symbolic::BDD;

#[derive(clap::Args, Debug)]
pub struct ConvertCommand {
    /// The `.pg` file to load
    game_path: PathBuf,
    /// What the given game should be converted into.
    #[clap(subcommand)]
    goal: ConversionGoal,
    /// Write the visualised result of the `goal` conversion into a GraphViz file.
    #[clap(short, long, global = true)]
    dot_path: Option<PathBuf>,
    /// Write the visualised result of the `goal` conversion into a Mermaid.js file
    #[clap(short, long, global = true)]
    mermaid_path: Option<PathBuf>,
    /// Write the result of the `goal` conversion into a `.pg` file
    #[clap(short, long, global = true)]
    pg_path: Option<PathBuf>,
}

#[derive(clap::Subcommand, Debug)]
pub enum ConversionGoal {
    /// Keep the current parity game
    #[clap(name = "pg")]
    ParityGame,
    /// Convert a given normal parity game into a symbolic parity game
    #[clap(name = "sy")]
    SymbolicParityGame,
    /// Convert the given parity game into a register game.
    #[clap(name = "rg")]
    RegisterGame {
        /// Create a `k`-register-index game, upper bound for valid results is `1 + log(n)`
        ///
        /// Default assumes `k = 1 + log(n)`, where `n` is the amount of vertices in the parity game.
        #[clap(short)]
        k: Option<u32>,
        #[clap(short, default_value = "new")]
        v: RgVersion,
        /// Whether this should use the paper renderer.
        #[clap(short)]
        p: bool,
    },
    #[clap(name = "srg")]
    SymbolicRegisterGame {
        /// Create a `k`-register-index game, upper bound for valid results is `1 + log(n)`
        ///
        /// Default assumes `k = 1 + log(n)`, where `n` is the amount of vertices in the parity game.
        #[clap(short)]
        k: Option<u32>,
    },
}

#[derive(Debug, clap::ValueEnum, Copy, Clone)]
pub enum RgVersion {
    Old,
    New,
}

impl ConvertCommand {
    #[tracing::instrument(name="Convert Parity Game", skip(self), fields(path=?self.game_path, goal=?self.goal))]
    pub fn run(self) -> eyre::Result<()> {
        let parity_game = crate::utils::load_parity_game(&self.game_path)?;

        match self.goal {
            ConversionGoal::ParityGame => {
                if let Some(path) = self.mermaid_path {
                    std::fs::write(&path, MermaidWriter::write_mermaid(&parity_game)?)?;

                    tracing::info!(?path, "Wrote Mermaid graph to path")
                }

                if let Some(path) = self.pg_path {
                    std::fs::write(&path, parity_game.to_pg())?;

                    tracing::info!(?path, "Wrote PG graph to path")
                }

                if let Some(path) = self.dot_path {
                    std::fs::write(&path, DotWriter::write_dot(&parity_game)?)?;

                    tracing::info!(?path, "Wrote GraphViz graph to path")
                }
            }
            ConversionGoal::RegisterGame { k, v, p } => {
                let k = k.unwrap_or_else(|| 1 + parity_game.vertex_count().ilog2()) as u8;
                let had_nodes = parity_game.vertex_count();
                let register_game = match v {
                    RgVersion::Old => RegisterGame::construct(&parity_game, k, Owner::Even),
                    RgVersion::New => RegisterGame::construct_2021(&parity_game, k, Owner::Even),
                };
                let rg_vtx = register_game.vertices.len();

                tracing::debug!(
                    from_vertex = register_game.original_game.vertex_count(),
                    to_vertex = rg_vtx,
                    ratio = rg_vtx / register_game.original_game.vertex_count(),
                    "Converted from PG to RG PG"
                );

                if let Some(path) = self.mermaid_path {
                    let out_str = if p {
                        MermaidWriter::write_mermaid(&VisualRegisterGame(&register_game))?
                    } else {
                        MermaidWriter::write_mermaid(&register_game)?
                    };
                    std::fs::write(&path, out_str)?;

                    tracing::info!(?path, "Wrote Mermaid graph to path")
                }

                if let Some(path) = self.dot_path {
                    let out_str = if p {
                        DotWriter::write_dot(&VisualRegisterGame(&register_game))?
                    } else {
                        DotWriter::write_dot(&register_game)?
                    };
                    std::fs::write(&path, out_str)?;

                    tracing::info!(?path, "Wrote GraphViz graph to path")
                }

                if let Some(path) = self.pg_path {
                    let game = register_game.to_game()?;

                    tracing::debug!(
                        from_vertex = had_nodes,
                        to_vertex = game.vertex_count(),
                        ratio = game.vertex_count() / had_nodes,
                        "Converted from PG to RG PG"
                    );

                    std::fs::write(&path, game.to_pg())?;

                    tracing::info!(?path, "Wrote PG graph to path")
                }
            }
            ConversionGoal::SymbolicParityGame => {
                let s_pg = SymbolicParityGame::<BDD>::from_explicit(&parity_game)?;
                s_pg.gc();

                tracing::info!(
                    parity_node_count = parity_game.vertex_count(),
                    symbolic_node_count = s_pg.bdd_node_count(),
                    "Converted to symbolic parity game"
                );

                if let Some(path) = self.dot_path {
                    std::fs::write(&path, DotWriter::write_dot_symbolic(&s_pg, [])?)?;

                    tracing::info!(?path, "Wrote GraphViz graph to path")
                }
            }
            ConversionGoal::SymbolicRegisterGame { k } => {
                let k = k.unwrap_or_else(|| 1 + parity_game.vertex_count().ilog2()) as u8;
                let s_rg: SymbolicRegisterGame<BDD> = SymbolicRegisterGame::from_symbolic(&parity_game, k, Owner::Even)?;
                s_rg.gc();

                tracing::info!(
                    parity_node_count = parity_game.vertex_count(),
                    symbolic_node_count = s_rg.bdd_node_count(),
                    "Converted to symbolic parity game"
                );

                if let Some(path) = self.dot_path {
                    std::fs::write(&path, DotWriter::write_dot_symbolic_register(&s_rg, [])?)?;

                    tracing::info!(?path, "Wrote GraphViz graph to path")
                }
            }
        }

        Ok(())
    }
}
