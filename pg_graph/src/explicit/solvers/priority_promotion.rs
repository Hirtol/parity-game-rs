use crate::{
    explicit::{
        solvers::{AttractionComputer, SolverOutput},
        BitsetExtensions, ParityGame, ParityGraph, SubGame, VertexId,
    },
    Owner, ParityVertexSoa, Priority,
};
use ahash::HashMapExt;
use fixedbitset::FixedBitSet;
use itertools::Itertools;

pub struct PPSolver<'a> {
    pub promotions: usize,
    game: &'a ParityGame<u32>,
    attract: AttractionComputer<u32>,
}

impl<'a> PPSolver<'a> {
    pub fn new(game: &'a ParityGame<u32>) -> Self {
        PPSolver {
            game,
            promotions: 0,
            attract: AttractionComputer::new(),
        }
    }

    #[tracing::instrument(name = "Run Priority Promotion", skip(self))]
    pub fn run(&mut self) -> SolverOutput {
        let (even, odd) = self.priority_promotion(self.game);

        let mut winners = vec![Owner::Even; self.game.vertex_count()];
        for idx in odd.ones() {
            winners[idx] = Owner::Odd;
        }

        SolverOutput {
            winners,
            strategy: None,
        }
    }

    /// Return (W_even, W_odd)
    fn priority_promotion<T: ParityGraph<u32>>(&mut self, game: &T) -> (FixedBitSet, FixedBitSet) {
        let (mut winning_even, mut winning_odd) = (
            FixedBitSet::with_capacity(game.vertex_count()),
            FixedBitSet::with_capacity(game.vertex_count()),
        );
        // Mapping from Vertex id -> Priority region
        // let mut region: Vec<Priority> = game.vertices_index().map(|v| game.priority(v)).collect_vec();
        let mut region: Vec<Priority> = self.game.vertices.priority.clone();

        let mut current_game = game.create_subgame([]);
        while current_game.vertex_count() > 0 {
            let d = current_game.priority_max();
            region = self.game.vertices.priority.clone();

            let (w_even, w_odd) = self.search_dominion(&current_game, &mut region, d);
            winning_even.union_with(&w_even);
            winning_odd.union_with(&w_odd);

            tracing::trace!(
                "Found dominion for `d` ({d}) - {} - {}",
                w_even.count_ones(..),
                current_game.vertex_count()
            );

            current_game.shrink_subgame(&w_even);
            current_game.shrink_subgame(&w_odd);
        }

        (
            winning_even,
            winning_odd,
        )
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
    ) -> (FixedBitSet, FixedBitSet) {
        let mut partial_subgame = current_game.create_subgame([]);
        let mut region_priority = region_priority;

        loop {
            let region_owner = Owner::from_priority(region_priority);
            let starting_set = current_game
                .vertices_index()
                .filter(|v| region[v.index()] == region_priority)
                .collect_vec();

            let attraction_set = self.attract.attractor_set(&partial_subgame, region_owner, starting_set.iter().copied());

            // Find all vertices where alpha-bar can 'escape' the quasi-dominion
            let base_escape_targets = attraction_set.ones_vertices()
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
            let any_open_alpha_vertices = starting_set.iter().copied()
                .filter(|v| current_game.owner(*v) == region_owner)
                .any(|v| {
                    current_game
                        .edges(v)
                        .all(|succ| !attraction_set.contains(succ.index()))
                });

            // We first check if there is any alpha-vertex which _has_ to leave our quasi-dominion.
            // Subsequently, if not, check if there are any escape targets for alpha-bar in the current partial_subgame.
            if any_open_alpha_vertices || base_escape_targets.iter().any(|v| partial_subgame.vertex_in_subgame(*v)) {
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
                for (v_id, rp) in region.iter_mut().enumerate() {
                    if *rp < next_priority {
                        *rp = self.game.vertices.priority[v_id]
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
                        .attractor_set(current_game, region_owner, attraction_set.ones_vertices());

                tracing::trace!(priority=region_priority, size=full_dominion.count_ones(..), "Found dominion");

                return match region_owner {
                    Owner::Even => (full_dominion, FixedBitSet::with_capacity(current_game.original_vertex_count())),
                    Owner::Odd => (FixedBitSet::with_capacity(current_game.original_vertex_count()), full_dominion),
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
