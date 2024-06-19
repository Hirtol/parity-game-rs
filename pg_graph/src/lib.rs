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

    fn trivial_pg() -> eyre::Result<ParityGame> {
        let mut pg = r#"parity 1;
0 1 1 0 "0";"#;
        let pg = pg_parser::parse_pg(&mut pg).unwrap();
        ParityGame::new(pg)
    }

    pub fn trivial_pg_2() -> eyre::Result<ParityGame> {
        let mut pg = r#"parity 2;
0 0 0 0,1 "0";
1 1 1 1,0 "1";"#;
        let pg = pg_parser::parse_pg(&mut pg).unwrap();
        ParityGame::new(pg)
    }
}
