#![feature(is_sorted)]

pub use datatypes::*;
pub use parity_game::*;

mod datatypes;
pub mod explicit;
pub mod parity_game;
pub mod register_game;
pub mod symbolic;
pub mod visualize;

#[cfg(test)]
pub mod tests {
    use crate::ParityGame;
    use pg_parser::parse_pg;
    use std::path::PathBuf;

    pub fn load_example(example: impl AsRef<str>) -> ParityGame {
        let input = std::fs::read_to_string(example_dir().join(example.as_ref())).unwrap();
        let pg = parse_pg(&mut input.as_str()).unwrap();
        ParityGame::new(pg).unwrap()
    }

    pub fn example_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("game_examples")
    }
}
