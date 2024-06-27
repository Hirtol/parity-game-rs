use oxidd_core::util::OutOfMemory;

pub use parity_game::SymbolicParityGame;

pub mod helpers;
pub mod oxidd_extensions;
pub mod parity_game;
pub mod register_game;
pub mod register_game_one_hot;
pub mod sat;
pub mod solvers;

pub type BDD = oxidd::bdd::BDDFunction;
pub type BCDD = oxidd::bcdd::BCDDFunction;
pub type Result<T> = std::result::Result<T, BddError>;

#[derive(Debug, Clone, thiserror::Error)]
pub enum BddError {
    #[error("Failed to allocate memory")]
    AllocError(OutOfMemory),
    #[error("Given input was not mapped before hand")]
    NoMatchingInput,
    #[error("No input was provided")]
    NoInput,
}

impl From<OutOfMemory> for BddError {
    fn from(value: OutOfMemory) -> Self {
        Self::AllocError(value)
    }
}