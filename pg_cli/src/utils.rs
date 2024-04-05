use std::fmt::Debug;
use std::path::Path;

/// Load the parity game, with an expected `.pg` format, from the provided path.
#[tracing::instrument(level = "debug")]
pub fn load_parity_game(pg_path: impl AsRef<Path> + Debug) -> eyre::Result<pg_graph::ParityGame> {
    let txt = std::fs::read_to_string(pg_path)?;
    let pg_parsed = pg_parser::parse_pg(&mut txt.as_str()).map_err(|e| eyre::eyre!(e))?;
    let parity_game = pg_graph::ParityGame::new(pg_parsed)?;
    
    Ok(parity_game)
}