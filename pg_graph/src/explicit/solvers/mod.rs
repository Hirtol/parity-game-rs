use crate::explicit::reduced_register_game::{RegisterParityGraph, RegisterVertex};
use crate::explicit::register_game::{GameRegisterVertex, RegisterGame};
use crate::{explicit::{ParityGraph, VertexId}, Owner, ParityVertex};
use itertools::Itertools;
use petgraph::graph::IndexType;
use std::collections::VecDeque;

pub mod small_progress;
pub mod tangle_learning;
pub mod zielonka;
pub mod register_zielonka;
pub mod fully_reduced_reg_zielonka;

#[derive(Debug, Clone, Default)]
pub struct SolverOutput {
    /// Contains a map of each vertex id to the player which has a winning strategy from that vertex.
    pub winners: Vec<Owner>,
    /// Optionally can contain the strategy for the entire game, mapping each implicit VertexId (index in vector) to one
    /// of the possible edges (value at index).
    pub strategy: Option<Vec<VertexId>>,
}

pub struct AttractionComputer<Ix> {
    queue: VecDeque<VertexId<Ix>>,
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
    pub fn attractor_set<V: ParityVertex + 'static, T: ParityGraph<Ix, V>>(
        &mut self,
        game: &T,
        player: Owner,
        starting_set: impl IntoIterator<Item = VertexId<Ix>>,
    ) -> ahash::HashSet<VertexId<Ix>> {
        let mut attract_set = ahash::HashSet::from_iter(starting_set);
        self.queue.extend(&attract_set);

        while let Some(next_item) = self.queue.pop_back() {
            for predecessor in game.predecessors(next_item) {
                let vertex = game.get(predecessor).expect("Invalid predecessor");
                let should_add = if vertex.owner() == player {
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

    /// Calculate the attraction set for the given player.
    ///
    /// **Assumes that edges are grouped by the original vertex id consecutively**
    #[allow(clippy::collapsible_else_if)]
    pub fn attractor_set_reg_game<T: ParityGraph<Ix, GameRegisterVertex>>(
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
                let predecessor_v = game.get(predecessor).expect("Invalid predecessor");
                let should_add = if is_aligned {
                    if predecessor_v.owner == player {
                        // *any* edge needs to lead to the attraction set, since this is a predecessor of an item already in the attraction set we know that already!
                        true
                    } else {
                        // We pretend `E_i` vertices exist, but this requires that all these imagined `E_i` vertices are part of the
                        // attraction set. Thus, as the owner of this vertex != player we need all chunks to have at least one attraction set edge.
                        let mut iter = game.edges(predecessor);
                        let v = iter.next().expect("No edges for vertex");
                        let node = game.get(v).unwrap();
                        let mut current_original_vertex_id = node.original_v;
                        let mut current_group_has_edge = attract_set.contains(&v);
                        let mut output = true;

                        for v in iter {
                            let node = game.get(v).unwrap();

                            if current_original_vertex_id != node.original_v {
                                if !current_group_has_edge {
                                    output = false;
                                    break;
                                }
                                current_original_vertex_id = node.original_v;
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
                    if predecessor_v.owner == player {
                        // First filter on the underlying original_graph_id to ensure those are equal and THEN check if there's any alternatives
                        // This way we model the existence of an E_i vertex
                        let next_item_v = game.get(next_item).expect("Invalid");
                        game.edges(predecessor)
                            .filter(|&v| game.get(v).map(|w| w.original_v == next_item_v.original_v).unwrap_or_default())
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
    pub fn attractor_set_reg_game_full_reduced<T: RegisterParityGraph<Ix, RegisterVertex>>(
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
                let predecessor_v = game.get(predecessor).expect("Invalid predecessor");
                let should_add = if is_aligned {
                    if predecessor_v.owner == player {
                        // *any* edge needs to lead to the attraction set, since this is a predecessor of an item already in the attraction set we know that already!
                        true
                    } else {
                        // We pretend `E_i` vertices exist, but this requires that all these imagined `E_i` vertices are part of the
                        // attraction set. Thus, as the owner of this vertex != player we need all chunks to have at least one attraction set edge.
                        game.grouped_edges(predecessor)
                                .all(|mut group| {
                                    group.any(|v| attract_set.contains(&v))
                                })
                    }
                } else {
                    if predecessor_v.owner == player {
                        // First filter on the underlying original_graph_id to ensure those are equal and THEN check if there's any alternatives
                        // This way we model the existence of an E_i vertex
                        let next_item_v = game.get(next_item).expect("Invalid");
                        game.edges_for_root_vertex(predecessor, next_item_v.original_graph_id)
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
    use std::collections::HashSet;

    use crate::{
        explicit::{ParityGame, VertexId},
        Owner,
    };

    #[test]
    pub fn test_attract_set_computation() -> eyre::Result<()> {
        let mut pg = r#"parity 3;
0 1 1 0,1 "0";
1 1 0 2 "1";
2 2 0 2 "2";"#;
        let pg = pg_parser::parse_pg(&mut pg).unwrap();
        let pg = ParityGame::new(pg)?;

        let mut attract = super::AttractionComputer::new();
        let set = attract.attractor_set(&pg, Owner::Even, [VertexId::new(2)]);

        assert_eq!(set, HashSet::from_iter(vec![VertexId::new(2), VertexId::new(1)]));

        Ok(())
    }
}
