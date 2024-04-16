use std::collections::HashMap;
use crate::{Owner, VertexId};

pub mod small_progress;
pub mod zielonka;
pub mod tangle_learning;

#[derive(Debug, Clone, Default)]
pub struct SolverOutput {
    /// Contains a map of each vertex id to the player which has a winning strategy from that vertex.
    pub winners: Vec<Owner>,
    /// Optionally can contain the strategy for each player, mapping their owned vertices to one of their edges.
    pub strategy: Option<HashMap<Owner, HashMap<VertexId, VertexId>>>
}

