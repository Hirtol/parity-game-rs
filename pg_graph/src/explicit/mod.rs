use fixedbitset::sparse::SparseBitSetRef;
use fixedbitset::FixedBitSet;
use itertools::Itertools;
pub use parity_game::*;
use petgraph::graph::IndexType;

pub mod parity_game;
pub mod register_game;
pub mod solvers;
pub mod reduced_register_game;
pub mod register_tree;
pub mod trace;

pub mod pearce;

pub type VertexSet = FixedBitSet;

pub trait BitsetExtensions {
    /// Create an empty [FixedBitSet] with the capacity which matches the amount of original vertices in `pg`
    fn empty_game<Ix: IndexType, PG: ParityGraph<Ix>>(pg: &PG) -> Self;

    fn ones_vertices<Ix: IndexType>(&self) -> impl DoubleEndedIterator<Item = VertexId<Ix>>;

    fn printable_vertices(&self) -> Vec<usize> {
        self.ones_vertices::<usize>().map(|v| v.index()).collect_vec()
    }
    
    fn print_vertices(&self) {
        println!("Vertices: {:?}", self.ones_vertices::<usize>().collect_vec());
    }
}

impl BitsetExtensions for FixedBitSet {
    fn empty_game<Ix: IndexType, PG: ParityGraph<Ix>>(pg: &PG) -> Self {
        Self::with_capacity(pg.original_vertex_count())
    }

    fn ones_vertices<Ix: IndexType>(&self) -> impl DoubleEndedIterator<Item=VertexId<Ix>> {
        self.ones().map(VertexId::new)
    }
}

impl<'a> BitsetExtensions for SparseBitSetRef<'a> {
    fn empty_game<Ix: IndexType, PG: ParityGraph<Ix>>(pg: &PG) -> Self {
        todo!()
    }

    fn ones_vertices<Ix: IndexType>(&self) -> impl DoubleEndedIterator<Item=VertexId<Ix>> {
        self.ones().map(VertexId::new)
    }
}