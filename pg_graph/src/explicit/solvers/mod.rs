use crate::explicit::{BitsetExtensions, VertexSet};
use crate::{
    explicit::{
        reduced_register_game::RegisterParityGraph,
        register_game::RegisterGame,
        ParityGraph, VertexId,
    },
    Owner,
};
use fixedbitset::{generic::BitSet, FixedBitSet};
use itertools::Itertools;
use petgraph::graph::IndexType;
use std::borrow::Cow;
use std::collections::VecDeque;

pub mod fully_reduced_reg_zielonka;
pub mod register_zielonka;
pub mod small_progress;
pub mod tangle_learning;
pub mod zielonka;
pub mod priority_promotion;

#[derive(Debug, Clone, Default)]
pub struct SolverOutput {
    /// Contains a map of each vertex id to the player which has a winning strategy from that vertex.
    pub winners: Vec<Owner>,
    /// Optionally can contain the strategy for the entire game, mapping each implicit VertexId (index in vector) to one
    /// of the possible edges (value at index).
    pub strategy: Option<Vec<VertexId>>,
}

impl SolverOutput {
    pub fn from_winning(vertex_count: usize, winning_odd: &VertexSet) -> Self {
        let mut winners = vec![Owner::Even; vertex_count];
        for idx in winning_odd.ones() {
            winners[idx] = Owner::Odd;
        }

        SolverOutput {
            winners,
            strategy: None,
        }
    }
}

#[derive(Default)]
pub struct AttractionComputer<Ix> {
    queue: VecDeque<VertexId<Ix>>,
}

fn print_type<T>(_: &T) {
    tracing::error!("{:?}", std::any::type_name::<T>());
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
    ) -> VertexSet {
        let mut attract_set = VertexSet::empty_game(game);
        attract_set.extend(starting_set.into_iter().map(|v| v.index()));

        self.attractor_set_bit(game, player, Cow::Owned(attract_set))
    }

    /// Calculate the attraction set for the given starting set.
    ///
    /// This resulting set will contain all vertices which:
    /// * If a vertex is owned by `player`, then if any edge leads to the attraction set it will be added to the resulting set.
    /// * If a vertex is _not_ owned by `player`, then only if _all_ edges lead to the attraction set will it be added.
    #[inline]
    pub fn attractor_set_bit<T: ParityGraph<Ix>>(
        &mut self,
        game: &T,
        player: Owner,
        starting_set: Cow<'_, FixedBitSet>
    ) -> VertexSet {
        self.queue.extend(starting_set.ones_vertices());
        let mut attract_set = starting_set.into_owned();

        while let Some(next_item) = self.queue.pop_back() {
            for predecessor in game.predecessors(next_item) {
                if attract_set.contains(predecessor.index()) {
                    continue;
                }

                let should_add = if game.owner(predecessor) == player {
                    // *any* edge needs to lead to the attraction set, since this is a predecessor of an item already in the attraction set we know that already!
                    true
                } else {
                    use fixedbitset::specific::SubBitSet;
                    // let left = game.edges_bit(predecessor).is_subset(&attract_set);
                    // let right = game.edges(predecessor).all(|v| attract_set.contains(v.index()));
                    // 
                    // if left != right {
                    //     tracing::debug!("LEFT AND RIGHT DONT AGREE: {} - {}", left, right);
                    //     print_type(&game.edges_bit(predecessor));
                    //     // let state = game.edges_bit(predecessor).as_simd_blocks().collect_vec();
                    //     // tracing::debug!(?state, "Edges");
                    //     tracing::debug!(state=?attract_set.as_simd_blocks().collect_vec(), "Attr");
                    //     tracing::debug!(edges=?game.edges(predecessor).collect_vec(), "Actual");
                    //     std::process::exit(0);
                    // }
                    // right
                    // left

                    game.edges_bit(predecessor).is_subset(&attract_set)
                    // game.edges(predecessor).all(|v| attract_set.contains(v.index()))
                };

                // Only add to the attraction set if we should
                if should_add {
                    attract_set.insert(predecessor.index());
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
    pub fn attractor_set_reg_game<T: RegisterParityGraph<Ix>>(
        &mut self,
        game: &T,
        reg_game: &RegisterGame,
        player: Owner,
        starting_set: impl IntoIterator<Item = VertexId<Ix>>,
    ) -> VertexSet {
        self.queue.extend(starting_set);
        let mut attract_set = VertexSet::empty_game(game);
        attract_set.extend(self.queue.iter().map(|v| v.index()));
        let is_aligned = player == reg_game.controller;

        while let Some(next_item) = self.queue.pop_back() {
            for predecessor in game.predecessors(next_item) {
                if attract_set.contains(predecessor.index()) {
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
                        let mut current_group_has_edge = attract_set.contains(v.index());
                        let mut output = true;

                        for v in iter {
                            let node_original_v = game.underlying_vertex_id(v);

                            if current_original_vertex_id != node_original_v {
                                if !current_group_has_edge {
                                    output = false;
                                    break;
                                }
                                current_original_vertex_id = node_original_v;
                                current_group_has_edge = attract_set.contains(v.index());
                            } else if !current_group_has_edge && attract_set.contains(v.index()) {
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
                        game.edges_for_root_vertex(predecessor, game.underlying_vertex_id(next_item))
                            .all(|v| attract_set.contains(v.index()))
                    } else {
                        use fixedbitset::specific::SubBitSet;
                        game.edges_bit(predecessor).is_subset(&attract_set)
                        // game.edges(predecessor).all(|v| attract_set.contains(v.index()))
                    }
                };

                // Only add to the attraction set if we should
                if should_add {
                    attract_set.insert(predecessor.index());
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
    ) -> VertexSet {
        self.queue.extend(starting_set);
        let mut attract_set = VertexSet::empty_game(game);
        attract_set.extend(self.queue.iter().map(|v| v.index()));

        let is_aligned = player == controller;

        while let Some(next_item) = self.queue.pop_back() {
            for predecessor in game.predecessors(next_item) {
                if attract_set.contains(predecessor.index()) {
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
                            .all(|mut group| group.any(|v| attract_set.contains(v.index())))
                    }
                } else {
                    if predecessor_owner == player {
                        // First filter on the underlying original_graph_id to ensure those are equal and THEN check if there's any alternatives
                        // This way we model the existence of an E_i vertex
                        game.edges_for_root_vertex(predecessor, game.underlying_vertex_id(next_item))
                            .all(|v| attract_set.contains(v.index()))
                    } else {
                        game.edges(predecessor).all(|v| attract_set.contains(v.index()))
                    }
                };

                // Only add to the attraction set if we should
                if should_add {
                    attract_set.insert(predecessor.index());
                    self.queue.push_back(predecessor);
                }
            }
        }

        attract_set
    }
}

#[cfg(test)]
mod tests {
    use crate::explicit::BitsetExtensions;
    use crate::{
        explicit::VertexId,
        Owner,
    };
    use std::collections::HashSet;

    #[test]
    pub fn test_attract_set_computation() -> eyre::Result<()> {
        let pg = r#"parity 3;
0 1 1 0,1 "0";
1 1 0 2 "1";
2 2 0 2 "2";"#;
        let pg = crate::tests::parse_pg_from_str(pg);

        let mut attract = super::AttractionComputer::new();
        let set = attract.attractor_set(&pg, Owner::Even, [VertexId::new(2)]);

        assert_eq!(set.ones_vertices::<u32>().collect::<HashSet<_>>(), HashSet::from_iter(vec![VertexId::new(2), VertexId::new(1)]));

        Ok(())
    }
}
