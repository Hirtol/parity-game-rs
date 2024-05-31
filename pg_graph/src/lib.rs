#![feature(is_sorted)]

pub use datatypes::*;

mod datatypes;
pub mod explicit;
pub mod symbolic;
pub mod visualize;

#[cfg(test)]
pub mod tests {
    use std::path::PathBuf;

    use pg_parser::parse_pg;

    use crate::explicit::ParityGame;

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
