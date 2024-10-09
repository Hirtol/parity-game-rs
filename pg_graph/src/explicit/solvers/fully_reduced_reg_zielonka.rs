use crate::explicit::reduced_register_game::{ReducedRegisterGame, RegisterParityGraph};
use crate::{explicit::{
    solvers::{AttractionComputer, SolverOutput}
    , ParityGraph, VertexId,
}, Owner};

pub struct ZielonkaSolver<'a> {
    pub recursive_calls: usize,
    rg: &'a ReducedRegisterGame<'a>,
    attract: AttractionComputer<u32>,
}

impl<'a> ZielonkaSolver<'a> {
    pub fn new(rg: &'a ReducedRegisterGame<'a>) -> Self {
        ZielonkaSolver {
            recursive_calls: 0,
            attract: AttractionComputer::new(),
            rg,
        }
    }

    #[tracing::instrument(name = "Run Zielonka", skip(self))]
    // #[profiling::function]
    pub fn run(&mut self) -> SolverOutput {
        let (even, odd) = self.zielonka(self.rg);
        let mut winners = vec![Owner::Even; self.rg.vertex_count()];
        for idx in odd {
            winners[idx.index()] = Owner::Odd;
        }

        SolverOutput {
            winners,
            strategy: None,
        }
    }

    fn zielonka<T: RegisterParityGraph<u32>>(&mut self, game: &T) -> (Vec<VertexId>, Vec<VertexId>)
        where T::Parent: RegisterParityGraph<u32> {
        self.recursive_calls += 1;
        // If all the vertices are ignord
        if game.vertex_count() == 0 {
            (Vec::new(), Vec::new())
        } else {
            let d = game.priority_max();
            let attraction_owner = Owner::from_priority(d);
            let starting_set = game.vertices_index_by_priority(d);

            let attraction_set = self.attract.attractor_set_reg_game_full_reduced(game, self.rg.controller, attraction_owner, starting_set);

            let sub_game = game.create_subgame(attraction_set.iter().copied());

            let (mut even, mut odd) = self.zielonka(&sub_game);
            let (attraction_owner_set, not_attraction_owner_set) = if attraction_owner.is_even() {
                (&mut even, &mut odd)
            } else {
                (&mut odd, &mut even)
            };

            if not_attraction_owner_set.is_empty() {
                attraction_owner_set.extend(attraction_set);
                (even, odd)
            } else {
                let b_attr = self.attract.attractor_set_reg_game_full_reduced(
                    game,
                    self.rg.controller,
                    attraction_owner.other(),
                    not_attraction_owner_set.iter().copied(),
                );
                let sub_game = game.create_subgame(b_attr.iter().copied());

                let (mut even, mut odd) = self.zielonka(&sub_game);
                let not_attraction_owner_set = if attraction_owner.is_even() {
                    &mut odd
                } else {
                    &mut even
                };
                not_attraction_owner_set.extend(b_attr);

                (even, odd)
            }
        }
    }
}

#[cfg(test)]
pub mod test {
    use std::time::Instant;

    use crate::{
        explicit::solvers::zielonka::ZielonkaSolver,
        tests::load_example,
        Owner,
    };

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