use crate::{
    explicit::{
        reduced_register_game::RegisterParityGraph,
        register_game::RegisterGame,
        ParityGraph, VertexId,
    },
    Owner,
};
use petgraph::graph::IndexType;
use std::collections::VecDeque;

pub mod fully_reduced_reg_zielonka;
pub mod register_zielonka;
pub mod small_progress;
pub mod tangle_learning;
pub mod zielonka;

#[derive(Debug, Clone, Default)]
pub struct SolverOutput {
    /// Contains a map of each vertex id to the player which has a winning strategy from that vertex.
    pub winners: Vec<Owner>,
    /// Optionally can contain the strategy for the entire game, mapping each implicit VertexId (index in vector) to one
    /// of the possible edges (value at index).
    pub strategy: Option<Vec<VertexId>>,
}

#[derive(Default)]
pub struct AttractionComputer<Ix> {
    queue: VecDeque<VertexId<Ix>>,
}

impl AttractionComputer<u32> {
    pub fn attractor_set_bit<T: ParityGraph<u32>>(
        &mut self,
        game: &T,
        player: Owner,
        starting_set: impl IntoIterator<Item = VertexId<u32>>,
    ) -> hi_sparse_bitset::BitSet<hi_sparse_bitset::config::_256bit> {
        // let mut attract_set = ahash::HashSet::from_iter(starting_set);
        self.queue.extend(starting_set);
        
        let mut bit_set = hi_sparse_bitset::BitSet::<hi_sparse_bitset::config::_256bit>::new();
        // let mut bit_set = roaring::RoaringBitmap::from_iter(self.queue.iter().map(|v| v.index() as u32));
        //
        self.queue.iter().for_each(|v| { bit_set.insert(v.index()); });

        while let Some(next_item) = self.queue.pop_back() {
            for predecessor in game.predecessors(next_item) {
                if bit_set.contains(predecessor.index()) {
                    continue;
                }

                let should_add = if game.owner(predecessor) == player {
                    // *any* edge needs to lead to the attraction set, since this is a predecessor of an item already in the attraction set we know that already!
                    true
                } else {
                    game.edges(predecessor).all(|v| bit_set.contains(v.index()))
                };

                // Only add to the attraction set if we should
                if should_add {
                    bit_set.insert(predecessor.index());
                    self.queue.push_back(predecessor);
                }
            }
        }

        bit_set
    }
}

impl<Ix: IndexType> AttractionComputer<Ix> {
    pub fn new() -> Self {
        Self {
            queue: Default::default(),
        }
    }

    /// Calculate the attraction set for the given starting set.
    ///
    /// This resulting set will contain all vertices which:
    /// * If a vertex is owned by `player`, then if any edge leads to the attraction set it will be added to the resulting set.
    /// * If a vertex is _not_ owned by `player`, then only if _all_ edges lead to the attraction set will it be added.
    pub fn attractor_set<T: ParityGraph<Ix>>(
        &mut self,
        game: &T,
        player: Owner,
        starting_set: impl IntoIterator<Item = VertexId<Ix>>,
    ) -> ahash::HashSet<VertexId<Ix>> {
        let mut attract_set = ahash::HashSet::from_iter(starting_set);
        self.queue.extend(&attract_set);

        while let Some(next_item) = self.queue.pop_back() {
            for predecessor in game.predecessors(next_item) {
                if attract_set.contains(&predecessor) {
                    continue;
                }
                
                let should_add = if game.owner(predecessor) == player {
                    // *any* edge needs to lead to the attraction set, since this is a predecessor of an item already in the attraction set we know that already!
                    true
                } else {
                    game.edges(predecessor).all(|v| attract_set.contains(&v))
                };

                // Only add to the attraction set if we should
                if should_add && attract_set.insert(predecessor) {
                    self.queue.push_back(predecessor);
                }
            }
        }

        attract_set
    }

    pub fn attractor_set_bit_fixed<T: ParityGraph<Ix>>(
        &mut self,
        game: &T,
        player: Owner,
        starting_set: impl IntoIterator<Item = VertexId<Ix>>,
    ) -> fixedbitset::FixedBitSet {
        self.queue.extend(starting_set);
        let mut bit_set = fixedbitset::FixedBitSet::with_capacity(game.original_vertex_count());
        bit_set.extend(self.queue.iter().map(|v| v.index()));

        while let Some(next_item) = self.queue.pop_back() {
            for predecessor in game.predecessors(next_item) {
                if bit_set.contains(predecessor.index()) {
                    continue;
                }

                let should_add = if game.owner(predecessor) == player {
                    // *any* edge needs to lead to the attraction set, since this is a predecessor of an item already in the attraction set we know that already!
                    true
                } else {
                    game.edges(predecessor).all(|v| bit_set.contains(v.index()))
                };

                // Only add to the attraction set if we should
                if should_add {
                    bit_set.insert(predecessor.index());
                    self.queue.push_back(predecessor);
                }
            }
        }

        bit_set
    }

    /// Calculate the attraction set for the given player.
    ///
    /// **Assumes that edges are grouped by the original vertex id consecutively**
    #[allow(clippy::collapsible_else_if)]
    pub fn attractor_set_reg_game<T: RegisterParityGraph<Ix>>(
        &mut self,
        game: &T,
        reg_game: &RegisterGame,
        player: Owner,
        starting_set: impl IntoIterator<Item = VertexId<Ix>>,
    ) -> ahash::HashSet<VertexId<Ix>> {
        let mut attract_set = ahash::HashSet::from_iter(starting_set);
        let is_aligned = player == reg_game.controller;
        self.queue.extend(&attract_set);

        while let Some(next_item) = self.queue.pop_back() {
            for predecessor in game.predecessors(next_item) {
                if attract_set.contains(&predecessor) {
                    continue;
                }
                let predecessor_owner = game.owner(predecessor);
                let should_add = if is_aligned {
                    if predecessor_owner == player {
                        // *any* edge needs to lead to the attraction set, since this is a predecessor of an item already in the attraction set we know that already!
                        true
                    } else {
                        // We pretend `E_i` vertices exist, but this requires that all these imagined `E_i` vertices are part of the
                        // attraction set. Thus, as the owner of this vertex != player we need all chunks to have at least one attraction set edge.
                        let mut iter = game.edges(predecessor);
                        let v = iter.next().expect("No edges for vertex");
                        let mut current_original_vertex_id = game.underlying_vertex_id(v);
                        let mut current_group_has_edge = attract_set.contains(&v);
                        let mut output = true;

                        for v in iter {
                            let node_original_v = game.underlying_vertex_id(v);

                            if current_original_vertex_id != node_original_v {
                                if !current_group_has_edge {
                                    output = false;
                                    break;
                                }
                                current_original_vertex_id = node_original_v;
                                current_group_has_edge = attract_set.contains(&v);
                            } else if !current_group_has_edge && attract_set.contains(&v) {
                                current_group_has_edge = true;
                            }
                        }
                        if !current_group_has_edge {
                            output = false;
                        }
                        // Shorter, but substantially slower way of computing the above
                        // game.edges(predecessor)
                        //         .chunk_by(|v| game.get(*v).unwrap().original_v)
                        //         .into_iter()
                        //         .all(|(_, mut chunk)| {
                        //             chunk.any(|v| attract_set.contains(&v))
                        //         });
                        output
                    }
                } else {
                    if predecessor_owner == player {
                        // First filter on the underlying original_graph_id to ensure those are equal and THEN check if there's any alternatives
                        // This way we model the existence of an E_i vertex
                        let next_item_v = game.underlying_vertex_id(next_item);
                        game.edges(predecessor)
                            .filter(|&v| game.underlying_vertex_id(v) == next_item_v)
                            .all(|v| attract_set.contains(&v))
                    } else {
                        game.edges(predecessor).all(|v| attract_set.contains(&v))
                    }
                };

                // Only add to the attraction set if we should
                if should_add && attract_set.insert(predecessor) {
                    self.queue.push_back(predecessor);
                }
            }
        }

        attract_set
    }

    /// Calculate the attraction set for the given player.
    ///
    /// **Assumes that edges are grouped by the original vertex id consecutively**
    #[allow(clippy::collapsible_else_if)]
    pub fn attractor_set_reg_game_full_reduced<T: RegisterParityGraph<Ix>>(
        &mut self,
        game: &T,
        controller: Owner,
        player: Owner,
        starting_set: impl IntoIterator<Item = VertexId<Ix>>,
    ) -> ahash::HashSet<VertexId<Ix>> {
        let mut attract_set = ahash::HashSet::from_iter(starting_set);
        let is_aligned = player == controller;
        self.queue.extend(&attract_set);

        while let Some(next_item) = self.queue.pop_back() {
            for predecessor in game.predecessors(next_item) {
                if attract_set.contains(&predecessor) {
                    continue;
                }
                
                let predecessor_owner = game.owner(predecessor);
                let should_add = if is_aligned {
                    if predecessor_owner == player {
                        // *any* edge needs to lead to the attraction set, since this is a predecessor of an item already in the attraction set we know that already!
                        true
                    } else {
                        // We pretend `E_i` vertices exist, but this requires that all these imagined `E_i` vertices are part of the
                        // attraction set. Thus, as the owner of this vertex != player we need all chunks to have at least one attraction set edge.
                        game.grouped_edges(predecessor)
                            .all(|mut group| group.any(|v| attract_set.contains(&v)))
                    }
                } else {
                    if predecessor_owner == player {
                        // First filter on the underlying original_graph_id to ensure those are equal and THEN check if there's any alternatives
                        // This way we model the existence of an E_i vertex
                        let next_item_v = game.underlying_vertex_id(next_item);
                        game.edges_for_root_vertex(predecessor, next_item_v)
                            .all(|v| attract_set.contains(&v))
                    } else {
                        game.edges(predecessor).all(|v| attract_set.contains(&v))
                    }
                };

                // Only add to the attraction set if we should
                if should_add && attract_set.insert(predecessor) {
                    self.queue.push_back(predecessor);
                }
            }
        }

        attract_set
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        explicit::VertexId,
        Owner,
    };
    use std::collections::HashSet;

    #[test]
    pub fn test_attract_set_computation() -> eyre::Result<()> {
        let mut pg = r#"parity 3;
0 1 1 0,1 "0";
1 1 0 2 "1";
2 2 0 2 "2";"#;
        let pg = crate::tests::parse_pg_from_str(pg);

        let mut attract = super::AttractionComputer::new();
        let set = attract.attractor_set(&pg, Owner::Even, [VertexId::new(2)]);

        assert_eq!(set, HashSet::from_iter(vec![VertexId::new(2), VertexId::new(1)]));

        Ok(())
    }
}
