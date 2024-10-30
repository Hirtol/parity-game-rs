use crate::explicit::reduced_register_game::RegisterParityGraph;
use crate::explicit::register_game::GameRegisterVertex;
use crate::explicit::solvers::Dominion;
use crate::{explicit::{
    solvers::{AttractionComputer, SolverOutput},
    BitsetExtensions, ParityGame, ParityGraph, VertexId, VertexSet,
}, Owner, Priority, Vertex};
use fixedbitset::FixedBitSet;
use itertools::Itertools;
use petgraph::graph::IndexType;
use soa_rs::Soars;
use std::borrow::Cow;

pub struct PPSolver<'a, Vertex: Soars> {
    pub promotions: usize,
    game: &'a ParityGame<u32, Vertex>,
    attract: AttractionComputer<u32>,
}

impl<'a, Vert: Soars> PPSolver<'a, Vert> {
    pub fn new(game: &'a ParityGame<u32, Vert>) -> Self {
        PPSolver {
            game,
            promotions: 0,
            attract: AttractionComputer::new(game.vertices.len()),
        }
    }
}

impl<'a> PPSolver<'a, Vertex> {
    #[tracing::instrument(name = "Run Priority Promotion", skip(self))]
    pub fn run(&mut self) -> SolverOutput {
        let (even, odd) = self.priority_promotion(self.game);
        SolverOutput::from_winning(self.game.original_vertex_count(), &odd)
    }

    /// Return (W_even, W_odd)
    fn priority_promotion<T: ParityGraph<u32>>(&mut self, game: &T) -> (FixedBitSet, FixedBitSet) {
        let (mut winning_even, mut winning_odd) = (
            FixedBitSet::with_capacity(game.vertex_count()),
            FixedBitSet::with_capacity(game.vertex_count()),
        );

        let mut current_game = game.create_subgame([]);
        while current_game.vertex_count() > 0 {
            let d = current_game.priority_max();
            // Mapping from Vertex id -> Priority region
            let mut region = self.game.vertices.priority().to_vec();

            let dominion = self.search_dominion(&current_game, &mut region, d);

            tracing::debug!(
                priority = dominion.dominating_p,
                size = dominion.vertices.count_ones(..),
                "Found dominion"
            );

            match Owner::from_priority(dominion.dominating_p) {
                Owner::Even => winning_even.union_with(&dominion.vertices),
                Owner::Odd => winning_odd.union_with(&dominion.vertices),
            }

            current_game.shrink_subgame(&dominion.vertices);
        }

        (winning_even, winning_odd)
    }

    /// Search for a dominion in the given game.
    ///
    /// The `region` array should reflect the original priorities of the vertices.
    /// Return (W_even, W_odd)
    fn search_dominion(
        &mut self,
        current_game: &impl ParityGraph<u32>,
        region: &mut [Priority],
        region_priority: Priority,
    ) -> Dominion {
        let mut partial_subgame = current_game.create_subgame([]);
        let mut region_priority = region_priority;

        loop {
            let region_owner = Owner::from_priority(region_priority);
            let starting_set = partial_subgame
                .vertices_index()
                .filter(|v| region[v.index()] == region_priority)
                .collect_vec();

            let attraction_set =
                self.attract
                    .attractor_set(&partial_subgame, region_owner, starting_set.iter().copied());

            // Find all vertices where alpha-bar can 'escape' the quasi-dominion
            let base_escape_targets = attraction_set
                .ones_vertices()
                .filter(|v| current_game.owner(*v) != region_owner)
                .flat_map(|v| {
                    current_game
                        .edges(v)
                        .filter(|succ| !attraction_set.contains(succ.index()))
                })
                .collect_vec();
            // In the official algorithm this is defined on the complete attraction set, but the only way for
            // a vertex to be in the attraction set whilst having no edges pointing towards it is to be part of the starting set!
            // We can therefore skip quite a few vertices by just checking the initial set.
            let any_open_alpha_vertices = starting_set
                .iter()
                .copied()
                .filter(|v| current_game.owner(*v) == region_owner)
                .any(|v| current_game.edges(v).all(|succ| !attraction_set.contains(succ.index())));

            // We first check if there is any alpha-vertex which _has_ to leave our quasi-dominion.
            // Subsequently, if not, check if there are any escape targets for alpha-bar in the current partial_subgame.
            if any_open_alpha_vertices
                || base_escape_targets
                    .iter()
                    .any(|v| partial_subgame.vertex_in_subgame(*v))
            {
                // There are escapes in the current partial_subgame, expand the next (priority wise smaller) region.
                for v_id in attraction_set.ones() {
                    region[v_id] = region_priority;
                }
                // Next state
                partial_subgame.shrink_subgame(&attraction_set);
                region_priority = partial_subgame.priority_max();
            } else if !base_escape_targets.is_empty() {
                self.promotions += 1;
                // We only have escapes in the full game, so closed region in the current partial_subgame,
                // the opposing player can only escape to regions which dominate the current `d` priority
                let next_priority = base_escape_targets
                    .into_iter()
                    .map(|v| region[v.index()])
                    .min()
                    .expect("No priority available");

                // Update region priorities, reset lower region priorities to their original values as we merge regions
                for v_id in attraction_set.ones() {
                    region[v_id] = next_priority;
                }
                for (rp, original_priority) in region.iter_mut().zip(self.game.vertices.priority()) {
                    if *rp < next_priority {
                        *rp = *original_priority
                    }
                }

                let to_exclude = current_game
                    .vertices_index()
                    .filter(|v| region[v.index()] > next_priority);
                let next_state = current_game.create_subgame(to_exclude);

                partial_subgame = next_state;
                region_priority = next_priority;
            } else {
                // Found a dominion!
                // There were no escape vertices, thus the quasi-dominion is alpha-closed, and therefore an actual dominion
                let full_dominion =
                    self.attract
                        .attractor_set_bit(current_game, region_owner, Cow::Owned(attraction_set));

                return Dominion {
                    dominating_p: region_priority,
                    vertices: full_dominion,
                };
            }
        }
    }
}

impl<'a> PPSolver<'a, GameRegisterVertex> {
    #[tracing::instrument(name = "Run Priority Promotion Register", skip(self))]
    pub fn run(&mut self) -> SolverOutput {
        let (even, odd) = self.priority_promotion(self.game);
        SolverOutput::from_winning(self.game.original_vertex_count(), &odd)
    }

    /// Return (W_even, W_odd)
    fn priority_promotion<T: RegisterParityGraph<u32>>(&mut self, game: &T) -> (FixedBitSet, FixedBitSet)
        where T::Parent: RegisterParityGraph<u32> {
        let (mut winning_even, mut winning_odd) = (
            FixedBitSet::with_capacity(game.vertex_count()),
            FixedBitSet::with_capacity(game.vertex_count()),
        );

        let mut current_game = game.create_subgame([]);
        while current_game.vertex_count() > 0 {
            let d = current_game.priority_max();
            // Mapping from Vertex id -> Priority region
            let mut region = self.game.vertices.priority().to_vec();

            let dominion = self.search_dominion(&current_game, &mut region, d);

            tracing::debug!(
                priority = dominion.dominating_p,
                size = dominion.vertices.count_ones(..),
                "Found dominion"
            );

            match Owner::from_priority(dominion.dominating_p) {
                Owner::Even => winning_even.union_with(&dominion.vertices),
                Owner::Odd => winning_odd.union_with(&dominion.vertices),
            }

            current_game.shrink_subgame(&dominion.vertices);
        }

        (winning_even, winning_odd)
    }

    /// Search for a dominion in the given game.
    ///
    /// The `region` array should reflect the original priorities of the vertices.
    /// Return (W_even, W_odd)
    fn search_dominion(
        &mut self,
        current_game: &impl RegisterParityGraph<u32>,
        region: &mut [Priority],
        region_priority: Priority,
    ) -> Dominion {
        let mut partial_subgame = current_game.create_subgame([]);
        let mut region_priority = region_priority;

        loop {
            let region_owner = Owner::from_priority(region_priority);
            let starting_set = partial_subgame
                .vertices_index()
                .filter(|v| region[v.index()] == region_priority)
                .collect_vec();

            let attraction_set =
                self.attract
                    .attractor_set(&partial_subgame, region_owner, starting_set.iter().copied());

            // Find all vertices where alpha-bar can 'escape' the quasi-dominion
            let base_escape_targets = attraction_set
                .ones_vertices()
                .filter(|v| current_game.owner(*v) != region_owner)
                .flat_map(|v| {
                    current_game
                        .edges(v)
                        .filter(|succ| !attraction_set.contains(succ.index()))
                })
                .collect_vec();
            // In the official algorithm this is defined on the complete attraction set, but the only way for
            // a vertex to be in the attraction set whilst having no edges pointing towards it is to be part of the starting set!
            // We can therefore skip quite a few vertices by just checking the initial set.
            let any_open_alpha_vertices = starting_set
                .iter()
                .copied()
                .filter(|v| current_game.owner(*v) == region_owner)
                .any(|v| current_game.edges(v).all(|succ| !attraction_set.contains(succ.index())));

            // We first check if there is any alpha-vertex which _has_ to leave our quasi-dominion.
            // Subsequently, if not, check if there are any escape targets for alpha-bar in the current partial_subgame.
            if any_open_alpha_vertices
                || base_escape_targets
                .iter()
                .any(|v| partial_subgame.vertex_in_subgame(*v))
            {
                // There are escapes in the current partial_subgame, expand the next (priority wise smaller) region.
                for v_id in attraction_set.ones() {
                    region[v_id] = region_priority;
                }
                // Next state
                partial_subgame.shrink_subgame(&attraction_set);
                region_priority = partial_subgame.priority_max();
            } else if !base_escape_targets.is_empty() {
                self.promotions += 1;
                // We only have escapes in the full game, so closed region in the current partial_subgame,
                // the opposing player can only escape to regions which dominate the current `d` priority
                let next_priority = base_escape_targets
                    .into_iter()
                    .map(|v| region[v.index()])
                    .min()
                    .expect("No priority available");

                // let thing = attraction_set.ones_vertices().map(|v| current_game.underlying_vertex_id(v)).collect::<ahash::HashSet<_>>();
                // let new_starting_set = partial_subgame.vertices_index().filter(|v| thing.contains(v)).chain(attraction_set.ones_vertices());
                // let attraction_set = self.attract.attractor_set(&partial_subgame, region_owner,new_starting_set);

                // Update region priorities, reset lower region priorities to their original values as we merge regions
                for v_id in attraction_set.ones() {
                    region[v_id] = next_priority;
                }
                for (rp, original_priority) in region.iter_mut().zip(self.game.vertices.priority()) {
                    if *rp < next_priority {
                        *rp = *original_priority
                    }
                }

                let to_exclude = current_game
                    .vertices_index()
                    .filter(|v| region[v.index()] > next_priority);
                let next_state = current_game.create_subgame(to_exclude);

                partial_subgame = next_state;
                region_priority = next_priority;
            } else {
                // Found a dominion!
                // There were no escape vertices, thus the quasi-dominion is alpha-closed, and therefore an actual dominion
                let mut full_dominion =
                    self.attract
                        .attractor_set_bit(current_game, region_owner, Cow::Owned(attraction_set));

                // let thing = full_dominion.ones_vertices().map(|v| current_game.underlying_vertex_id(v)).collect::<ahash::HashSet<_>>();
                // for non in full_dominion.zero_vertices().collect_vec() {
                //     if thing.contains(&current_game.underlying_vertex_id(non)) {
                //         full_dominion.insert(non.index())
                //     }
                // }

                return Dominion {
                    dominating_p: region_priority,
                    vertices: full_dominion,
                };
            }
        }
    }
}

#[cfg(test)]
pub mod test {
    use crate::{explicit::solvers::priority_promotion::PPSolver, tests};

    #[test]
    pub fn verify_correctness() {
        for name in tests::examples_iter() {
            println!("Running test for: {name}...");
            let (game, compare) = tests::load_and_compare_example(&name);
            let mut pp_solver = PPSolver::new(&game);
            let solution = pp_solver.run();
            compare(solution);
            println!("{name} correct!")
        }
    }
}
