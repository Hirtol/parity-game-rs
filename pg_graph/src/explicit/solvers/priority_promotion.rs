use crate::{explicit::{
    solvers::{AttractionComputer, SolverOutput},
    BitsetExtensions, ParityGame, ParityGraph, VertexId,
}, Owner, ParityVertexSoa, Priority};
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
        let (even, odd) = self.priority_promotion_2(self.game);
        println!("EVEN: {even:?}\nODD: {odd:?}");
        let mut winners = vec![Owner::Even; self.game.vertex_count()];
        for idx in odd {
            winners[idx.index()] = Owner::Odd;
        }

        SolverOutput {
            winners,
            strategy: None,
        }
    }

    fn priority_promotion<T: ParityGraph<u32>>(&mut self, game: &T) -> (Vec<VertexId>, Vec<VertexId>) {
        self.promotions += 1;
        // Mapping of priority
        let mut regions: ahash::HashMap<Priority, FixedBitSet> = ahash::HashMap::new();
        // Mapping from Vertex id -> Priority region
        let mut region: Vec<Priority> = game.vertices_index().map(|v| game.priority(v)).collect_vec();
        let mut d = game.priority_max();

        let mut dominions: ahash::HashMap<Owner, FixedBitSet> = ahash::HashMap::new();
        for p in game.priorities_unique().sorted().rev() {
            for v in game.vertices_index_by_priority(p) {
                if region[v.index()] > p {
                    continue;
                }
                println!("{region:?}");

                let mut d = p;

                loop {
                    let region_owner = Owner::from_priority(d);
                    let sub_game = game.create_subgame(
                        region
                            .iter()
                            .enumerate()
                            .filter(|(_, p)| **p <= d)
                            .map(|(id, _)| VertexId::new(id)),
                    );
                    let starting_set = game.vertices_index_by_priority(d).chain(
                        region
                            .iter()
                            .enumerate()
                            .filter(|(_, &p)| p == d)
                            .map(|id| VertexId::new(id.0)),
                    );
                    let attraction_set = self
                        .attract
                        .attractor_set_bit_fixed(&sub_game, region_owner, starting_set);

                    let open_alpha_vertices = {
                        let mut vertices_owned_by_alpha = attraction_set
                            .ones_vertices()
                            .filter(|&v_id| game.owner(v_id) == region_owner);

                        vertices_owned_by_alpha
                            .any(|v_id| game.edges(v_id).all(|succ| !attraction_set.contains(succ.index())))
                    };

                    let escape_vertices = attraction_set
                        .ones_vertices()
                        .filter(|&v_id| game.owner(v_id) != region_owner)
                        .flat_map(|v_id| game.edges(v_id))
                        .filter(|successor| !attraction_set.contains(successor.index()))
                        .collect_vec();

                    let any_intersection = escape_vertices.iter().any(|v_id| sub_game.vertex_in_subgame(*v_id));

                    if open_alpha_vertices || any_intersection {
                        // Update the next priority
                        let new_d = sub_game
                            .vertices_index()
                            .filter(|v| !attraction_set.contains(v.index()))
                            .map(|v| sub_game.priority(v))
                            .max()
                            .expect("Empty subgame");

                        // Set regions
                        for v_id in attraction_set.ones() {
                            region[v_id] = d;
                        }
                        regions
                            .entry(d)
                            .and_modify(|v| v.union_with(&attraction_set))
                            .or_insert(attraction_set);

                        d = new_d;
                    } else if !escape_vertices.is_empty() {
                        // Merge regions and reset
                        // Set p to lowest escape
                        for v_id in attraction_set.ones() {
                            region[v_id] = d;
                        }
                        regions
                            .entry(d)
                            .and_modify(|v| v.union_with(&attraction_set))
                            .or_insert(attraction_set);
                        for (v_id, target_region) in region.iter_mut().enumerate() {
                            let v_p = game.priority(VertexId::new(v_id));
                            if v_p < d {
                                *target_region = v_p;
                            }
                        }
                        for (_, set) in regions.iter_mut().filter(|p| *p.0 < d) {
                            set.clear();
                        }

                        d = escape_vertices
                            .into_iter()
                            .map(|v| region[v.index()])
                            .min()
                            .expect("Impossible");
                    } else {
                        // Found a dominion
                        let real_attr =
                            self.attract
                                .attractor_set_bit_fixed(game, region_owner, attraction_set.ones_vertices());
                        dominions
                            .entry(region_owner)
                            .and_modify(|s| s.union_with(&real_attr))
                            .or_insert(real_attr);
                        break;
                    }
                }
            }
        }

        for (owner, dominion) in dominions {
            println!("Owner: {owner:?}, dominion: {:?}", dominion.ones().collect_vec())
        }

        // (dominions.entry(Owner::Even).or_default().ones_vertices().collect_vec(), dominions.entry(Owner::Odd).or_default().ones_vertices().collect_vec())
        todo!("Found dominions")
    }

    fn priority_promotion_2<T: ParityGraph<u32>>(&mut self, game: &T) -> (Vec<VertexId>, Vec<VertexId>) {
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

            let (w_even, w_odd) = self.search_dominion(&current_game, &current_game, &mut region, d);
            winning_even.union_with(&w_even);
            winning_odd.union_with(&w_odd);

            println!("Found dominion for `d` ({d})");

            current_game.shrink_subgame(&w_even);
            current_game.shrink_subgame(&w_odd);
        }

        (
            winning_even.ones_vertices().collect(),
            winning_odd.ones_vertices().collect(),
        )
    }

    /// Return (W_even, W_odd)
    fn search_dominion(
        &mut self,
        current_game: &impl ParityGraph<u32>,
        partial_subgame: &impl ParityGraph<u32>,
        region: &mut [Priority],
        region_priority: Priority,
    ) -> (FixedBitSet, FixedBitSet) {
        tracing::debug!("{}", partial_subgame.vertex_count());
        let region_owner = Owner::from_priority(region_priority);
        let starting_set = current_game
            .vertices_index()
            .filter(|v| region[v.index()] == region_priority);

        let attraction_set = self
            .attract
            .attractor_set_bit_fixed(partial_subgame, region_owner, starting_set);
        
        // Find all vertices where alpha-bar can 'escape' the quasi-dominion
        let base_escape_vertices = attraction_set.ones_vertices()
            .filter(|v| current_game.owner(*v) != region_owner)
            .filter(|v| {
                current_game
                    .edges(*v)
                    .any(|succ| !attraction_set.contains(succ.index()))
        }).collect_vec();
        let base_open_vertices = attraction_set.ones_vertices()
            .filter(|v| current_game.owner(*v) == region_owner)
            .filter(|v| {
                current_game
                    .edges(*v)
                    .all(|succ| !attraction_set.contains(succ.index()))
            }).collect_vec();
        
        // Check if it's a dominion, if there are no escape vertices the quasi-dominion is alpha-closed, thus it is an actual dominion
        if base_escape_vertices.is_empty() && base_open_vertices.is_empty() {
            tracing::debug!("Found dominion");
            let full_dominion = self.attract.attractor_set_bit_fixed(current_game, region_owner, attraction_set.ones_vertices());
            match region_owner {
                Owner::Even => (full_dominion, FixedBitSet::with_capacity(self.game.vertex_count())),
                Owner::Odd => (FixedBitSet::with_capacity(self.game.vertex_count()), full_dominion),
            }
        } else {
            // Find all vertices where alpha-bar can 'escape' the quasi-dominion
            let escape_vertices = attraction_set.ones_vertices()
                .filter(|v| partial_subgame.owner(*v) != region_owner)
                .filter(|v| {
                    partial_subgame
                        .edges(*v)
                        .any(|succ| !attraction_set.contains(succ.index()))
                });
            let open_vertices = attraction_set.ones_vertices()
                .filter(|v| partial_subgame.owner(*v) == region_owner)
                .filter(|v| {
                    partial_subgame
                        .edges(*v)
                        .all(|succ| !attraction_set.contains(succ.index()))
                });
            
            // We know that the quasi-dominion has escapes, now check if that's true in our current partial_subgame, or only in the full game.
            if escape_vertices.count() != 0 || open_vertices.count() != 0 {
                // There are escapes in the current state, find the lowest priority to escape to.
                for v_id in attraction_set.ones() {
                    region[v_id] = region_priority;
                }
                let next_state = partial_subgame.create_subgame_bit(&attraction_set);
                let next_priority = next_state.priority_max();
                tracing::debug!(next_priority, "Found escapes, looking at next lowest priority");
                self.search_dominion(current_game, &next_state, region, next_priority)
            } else {
                self.promotions += 1;
                // Only true in the full game, so closed region in the current state, the opposing player can only escape to regions which dominate the current `d` priority
                let next_priority = base_escape_vertices.into_iter()
                    .flat_map(|v| current_game.edges(v))
                    .filter(|succ| !attraction_set.contains(succ.index()))
                    .map(|v| region[v.index()])
                    .min()
                    .expect("No priority available");

                // Update region priorities, reset lower region priorities to their original values as we merge regions
                for v_id in attraction_set.ones() {
                    region[v_id] = region_priority;
                }
                for (v_id, rp) in region.iter_mut().enumerate() {
                    if *rp < region_priority {
                        *rp = self.game.vertices.priority[v_id]
                    }
                }
                
                let to_exclude = partial_subgame.vertices_index().filter(|v| region[v.index()] > next_priority);
                let next_state = partial_subgame.create_subgame(to_exclude);
                tracing::debug!(next_priority, "Locally closed: {:?}", partial_subgame.vertices_index().filter(|v| region[v.index()] > next_priority).collect_vec());
                tracing::debug!("Next state: {:?}", next_state.vertices_index().map(|v_id| (v_id, next_state.vertex(v_id), region[v_id.index()])).collect_vec());
                self.search_dominion(current_game, &next_state, region, next_priority)
            }
        }
    }
}

#[cfg(test)]
pub mod test {
    use std::time::Instant;

    use crate::{explicit::solvers::priority_promotion::PPSolver, tests::load_example, Owner};

    #[test]
    pub fn test_solve_tue_example() {
        let game = load_example("tue_example.pg");
        let mut solver = PPSolver::new(&game);

        let solution = solver.run().winners;

        println!("Solution: {:#?}", solution);

        assert!(solution.iter().all(|win| *win == Owner::Odd));
    }

    #[test]
    pub fn test_solve_action_converter() {
        let game = load_example("ActionConverter.tlsf.ehoa.pg");
        let mut solver = PPSolver::new(&game);

        let now = Instant::now();
        let solution = solver.run().winners;

        println!("Solution: {:#?}", solution);
        println!("Took: {:?}", now.elapsed());
        assert_eq!(
            solution,
            vec![
                Owner::Even,
                Owner::Odd,
                Owner::Even,
                Owner::Even,
                Owner::Even,
                Owner::Even,
                Owner::Odd,
                Owner::Odd,
                Owner::Even,
            ]
        )
    }
}
