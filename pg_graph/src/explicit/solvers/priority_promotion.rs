use crate::{
    explicit::{
        solvers::{AttractionComputer, SolverOutput},
        BitsetExtensions, ParityGame, ParityGraph, VertexId,
    },
    Owner, Priority,
};
use ahash::HashMapExt;
use fixedbitset::FixedBitSet;
use itertools::Itertools;

pub struct PPSolver<'a> {
    pub iterations: usize,
    game: &'a ParityGame<u32>,
    attract: AttractionComputer<u32>,
}

impl<'a> PPSolver<'a> {
    pub fn new(game: &'a ParityGame<u32>) -> Self {
        PPSolver {
            game,
            iterations: 0,
            attract: AttractionComputer::new(),
        }
    }

    #[tracing::instrument(name = "Run Priority Promotion", skip(self))]
    pub fn run(&mut self) -> SolverOutput {
        let (even, odd) = self.priority_promotion(self.game);
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
        self.iterations += 1;
        // Mapping of priority
        let mut regions: ahash::HashMap<Priority, FixedBitSet> = ahash::HashMap::new();
        // Mapping from Vertex id -> Priority region
        let mut region: Vec<Priority> = game.vertices_index().map(|v| game.priority(v)).collect_vec();
        let mut d = game.priority_max();

        let mut dominions: ahash::HashMap<Owner, FixedBitSet> = ahash::HashMap::new();

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

            let open_alpha_vertices ={
                let mut vertices_owned_by_alpha = attraction_set
                    .ones_vertices()
                    .filter(|&v_id| game.owner(v_id) == region_owner);

                vertices_owned_by_alpha.any(|v_id| game.edges(v_id).all(|succ| !attraction_set.contains(succ.index())))
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

                d = escape_vertices.into_iter().map(|v| region[v.index()]).min().expect("Impossible");
            } else {
                // Found a dominion
                dominions.entry(region_owner).and_modify(|s| s.union_with(&attraction_set)).or_insert(attraction_set);
                break;
            }
        }
        
        for (owner, dominion) in dominions {
            println!("Owner: {owner:?}, dominion: {:?}", dominion.ones().collect_vec())
        }

        todo!("Found dominions")
    }
}

#[cfg(test)]
pub mod test {
    use std::time::Instant;

    use crate::explicit::solvers::priority_promotion::PPSolver;
    use crate::{
        explicit::solvers::zielonka::ZielonkaSolver,
        tests::load_example,
        Owner,
    };

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
        let mut solver = ZielonkaSolver::new(&game);

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
