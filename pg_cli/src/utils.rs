use std::{fmt::Debug, path::Path};

use pg_graph::explicit::ParityGraph;

/// Load the parity game, with an expected `.pg` format, from the provided path.
#[tracing::instrument(level = "debug", skip_all)]
pub fn load_parity_game(pg_path: impl AsRef<Path> + Debug) -> eyre::Result<pg_graph::explicit::ParityGame> {
    let txt = std::fs::read_to_string(pg_path)?;
    let pg_parsed = pg_parser::parse_pg(&mut txt.as_str()).map_err(|e| eyre::eyre!(e))?;
    let parity_game = pg_graph::explicit::ParityGame::new(pg_parsed)?;
    tracing::info!(size = parity_game.vertex_count(), "Loaded parity game");

    Ok(parity_game)
}
