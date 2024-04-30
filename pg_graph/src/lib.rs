#![feature(is_sorted)]

pub use datatypes::*;
pub use parity_game::*;

pub mod solvers;
pub mod parity_game;
pub mod symbolic;
mod datatypes;
pub mod register_game;
pub mod visualize;


#[cfg(test)]
pub mod tests {
    use std::path::PathBuf;

    pub fn example_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap()
            .join("game_examples")
    }
}