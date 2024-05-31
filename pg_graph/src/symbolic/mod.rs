use oxidd::bdd::BDDFunction;
use oxidd_core::util::OutOfMemory;

pub use parity_game::SymbolicParityGame;

pub mod helpers;
pub mod oxidd_extensions;
pub mod parity_game;
pub mod register_game;
pub mod solvers;

pub type BDD = BDDFunction;
pub type Result<T> = std::result::Result<T, BddError>;

#[derive(Debug, Clone, thiserror::Error)]
pub enum BddError {
    #[error("Failed to allocate memory")]
    AllocError(OutOfMemory),
    #[error("No input was provided")]
    NoInput,
}

impl From<OutOfMemory> for BddError {
    fn from(value: OutOfMemory) -> Self {
        Self::AllocError(value)
    }
}
