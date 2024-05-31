#![feature(is_sorted)]

pub use datatypes::*;
pub use parity_game::*;

pub mod parity_game;
pub mod symbolic;
mod datatypes;
pub mod register_game;
pub mod visualize;
pub mod explicit;


#[cfg(test)]
pub mod tests {
    use std::path::PathBuf;
    use pg_parser::parse_pg;
    use crate::ParityGame;

    pub fn load_example(example: impl AsRef<str>) -> ParityGame {
        let input = std::fs::read_to_string(example_dir().join(example.as_ref())).unwrap();
        let pg = parse_pg(&mut input.as_str()).unwrap();
        ParityGame::new(pg).unwrap()
    }
    
    pub fn example_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap()
            .join("game_examples")
    }
}