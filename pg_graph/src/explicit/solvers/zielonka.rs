use crate::explicit::{BitsetExtensions, VertexSet};
use crate::{explicit::{
    solvers::{AttractionComputer, SolverOutput},
    ParityGame, ParityGraph,
}, Owner};
use itertools::Itertools;

pub struct ZielonkaSolver<'a> {
    pub recursive_calls: usize,
    game: &'a ParityGame<u32>,
    attract: AttractionComputer<u32>,
}

impl<'a> ZielonkaSolver<'a> {
    pub fn new(game: &'a ParityGame<u32>) -> Self {
        ZielonkaSolver {
            game,
            recursive_calls: 0,
            attract: AttractionComputer::new(game.vertex_count()),
        }
    }

    #[tracing::instrument(name = "Run Zielonka", skip(self))]
    // #[profiling::function]
    pub fn run(&mut self) -> SolverOutput {
        let (_, odd) = self.zielonka(self.game);
        SolverOutput::from_winning(self.game.vertex_count(), &odd)
    }

    fn zielonka<T: ParityGraph<u32>>(&mut self, game: &T) -> (VertexSet, VertexSet) {
        self.recursive_calls += 1;
        // If all the vertices are ignord
        if game.vertex_count() == 0 {
            (VertexSet::default(), VertexSet::default())
        } else {
            let d = game.priority_max();
            let attraction_owner = Owner::from_priority(d);
            let starting_set = game.vertices_index_by_priority(d);
            
            let attraction_set = self.attract.attractor_set(game, attraction_owner, starting_set);

            let sub_game = game.create_subgame_bit(&attraction_set);

            let (mut even, mut odd) = self.zielonka(&sub_game);

            let (attraction_owner_set, not_attraction_owner_set) = if attraction_owner.is_even() {
                (&mut even, &mut odd)
            } else {
                (&mut odd, &mut even)
            };
            if not_attraction_owner_set.is_clear() {
                attraction_owner_set.union_with(&attraction_set);
                (even, odd)
            } else {
                let b_attr = self.attract.attractor_set(
                    game,
                    attraction_owner.other(),
                    not_attraction_owner_set.ones_vertices(),
                );

                // If the attractor set doesn't grow for the opposing player then we can pre-emptively conclude
                // that the dominions don't change, without recursing further. See "An Improved Recursive Algorithm for Parity Games".
                if b_attr == *not_attraction_owner_set {
                    attraction_owner_set.union_with(&attraction_set);
                    return (even, odd)
                }

                let sub_game = game.create_subgame_bit(&b_attr);

                let (mut even, mut odd) = self.zielonka(&sub_game);
                let not_attraction_owner_set = if attraction_owner.is_even() {
                    &mut odd
                } else {
                    &mut even
                };
                not_attraction_owner_set.union_with(&b_attr);

                (even, odd)
            }
        }
    }
}

#[cfg(test)]
pub mod test {
    use crate::{
        explicit::solvers::zielonka::ZielonkaSolver,
        tests::load_example,
        Owner,
    };
    use std::time::Instant;

    #[test]
    pub fn test_solve_tue_example() {
        let game = load_example("tue_example.pg");
        let mut solver = ZielonkaSolver::new(&game);

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
