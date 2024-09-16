use std::collections::VecDeque;

use crate::explicit::register_game::RegisterGame;
use crate::{
    explicit::{ParityGraph, VertexId},
    Owner,
};

pub mod small_progress;
pub mod tangle_learning;
pub mod zielonka;
pub mod register_zielonka;

#[derive(Debug, Clone, Default)]
pub struct SolverOutput {
    /// Contains a map of each vertex id to the player which has a winning strategy from that vertex.
    pub winners: Vec<Owner>,
    /// Optionally can contain the strategy for the entire game, mapping each implicit VertexId (index in vector) to one
    /// of the possible edges (value at index).
    pub strategy: Option<Vec<VertexId>>,
}

pub struct AttractionComputer {
    queue: VecDeque<VertexId>,
}

impl AttractionComputer {
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
    pub fn attractor_set<T: ParityGraph>(
        &mut self,
        game: &T,
        player: Owner,
        starting_set: impl IntoIterator<Item = VertexId>,
    ) -> ahash::HashSet<VertexId> {
        let mut attract_set = ahash::HashSet::from_iter(starting_set);
        self.queue.extend(&attract_set);

        while let Some(next_item) = self.queue.pop_back() {
            for predecessor in game.predecessors(next_item) {
                let vertex = game.get(predecessor).expect("Invalid predecessor");
                let should_add = if vertex.owner == player {
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

    pub fn attractor_set_reg_game<T: ParityGraph>(
        &mut self,
        game: &T,
        reg_game: &RegisterGame,
        player: Owner,
        starting_set: impl IntoIterator<Item = VertexId>,
    ) -> ahash::HashSet<VertexId> {
        let mut attract_set = ahash::HashSet::from_iter(starting_set);
        let is_aligned = player == reg_game.controller;
        self.queue.extend(&attract_set);

        while let Some(next_item) = self.queue.pop_back() {
            let next_item_v = game.get(next_item).expect("Invalid");
            let priority_next_aligned = reg_game.controller.priority_aligned(next_item_v.priority);
            
            for predecessor in game.predecessors(next_item) {
                let vertex = game.get(predecessor).expect("Invalid predecessor");
                let should_add = if vertex.owner == player {
                    if is_aligned {
                        // *any* edge needs to lead to the attraction set, since this is a predecessor of an item already in the attraction set we know that already!
                        true
                    } else {
                        let all_unaligned= game.edges(predecessor).all(|v| !reg_game.controller.priority_aligned(game.get(v).unwrap().priority));
                        if all_unaligned {
                            // Check if this edge is the smallest possible, if so then this edge is attracting
                            let smaller_edges = game.edges(predecessor).filter(|v| game.get(*v).unwrap().priority < vertex.priority).count();
                            smaller_edges == 0
                        } else {
                            false
                        }
                    }
                } else {
                    // if is_aligned {
                        game.edges(predecessor).all(|v| attract_set.contains(&v))
                    // } else {
                    //     
                    // }
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
