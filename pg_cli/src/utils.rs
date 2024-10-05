use std::{fmt::Debug, path::Path};

use pg_graph::explicit::{ParityGameBuilder, ParityGraph};

/// Load the parity game, with an expected `.pg` format, from the provided path.
#[tracing::instrument(level = "debug", skip_all)]
pub fn load_parity_game(pg_path: impl AsRef<Path> + Debug) -> eyre::Result<pg_graph::explicit::ParityGame> {
    let txt = std::fs::read_to_string(pg_path)?;
    let mut builder = ParityGameBuilder::new();
    pg_parser::parse_pg(&mut txt.as_str(), &mut builder).map_err(|e| eyre::eyre!(e))?;
    let parity_game = builder.build();
    tracing::info!(size = parity_game.vertex_count(), edges = parity_game.edge_count(), "Loaded parity game");

    Ok(parity_game)
}
