use fixedbitset::FixedBitSet;
pub use parity_game::*;

pub mod parity_game;
pub mod register_game;
pub mod solvers;
pub mod reduced_register_game;

pub mod register_tree;

pub trait BitsetExtensions {
    fn ones_vertices(&self) -> impl Iterator<Item = VertexId>;
    
    fn zero_vertices(&self) -> impl Iterator<Item = VertexId>;
}

impl BitsetExtensions for FixedBitSet {
    fn ones_vertices(&self) -> impl Iterator<Item=VertexId> {
        self.ones().map(VertexId::new)
    }

    fn zero_vertices(&self) -> impl Iterator<Item=VertexId> {
        self.zeroes().map(VertexId::new)
    }
}
