use fixedbitset::FixedBitSet;
pub use parity_game::*;
use petgraph::graph::IndexType;

pub mod parity_game;
pub mod register_game;
pub mod solvers;
pub mod reduced_register_game;

pub mod register_tree;

pub type VertexSet = FixedBitSet;

pub trait BitsetExtensions {
    /// Create an empty [FixedBitSet] with the capacity which matches the amount of original vertices in `pg`
    fn empty_game<Ix: IndexType, PG: ParityGraph<Ix>>(pg: &PG) -> Self;

    fn ones_vertices<Ix: IndexType>(&self) -> impl Iterator<Item = VertexId<Ix>>;
    
    fn zero_vertices<Ix: IndexType>(&self) -> impl Iterator<Item = VertexId<Ix>>;
}

impl BitsetExtensions for FixedBitSet {
    fn empty_game<Ix: IndexType, PG: ParityGraph<Ix>>(pg: &PG) -> Self {
        Self::with_capacity(pg.original_vertex_count())
    }

    fn ones_vertices<Ix: IndexType>(&self) -> impl Iterator<Item=VertexId<Ix>> {
        self.ones().map(VertexId::new)
    }

    fn zero_vertices<Ix: IndexType>(&self) -> impl Iterator<Item=VertexId<Ix>> {
        self.zeroes().map(VertexId::new)
    }
}
