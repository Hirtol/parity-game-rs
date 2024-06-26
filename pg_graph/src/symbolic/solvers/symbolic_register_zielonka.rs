use oxidd_core::util::OptBool;
use oxidd_core::WorkerManager;

use crate::{
    explicit::solvers::SolverOutput,
    Owner,
    symbolic
    ,
};
use crate::symbolic::oxidd_extensions::GeneralBooleanFunction;
use crate::symbolic::register_game::SymbolicRegisterGame;
use crate::symbolic::sat::TruthAssignmentsIterator;

pub struct SymbolicRegisterZielonkaSolver<'a, F: GeneralBooleanFunction> {
    pub recursive_calls: usize,
    game: &'a SymbolicRegisterGame<F>,
}

impl<'a, F: GeneralBooleanFunction> SymbolicRegisterZielonkaSolver<'a, F>
    where for<'id> F::Manager<'id>: WorkerManager,
          for<'b, 'c> TruthAssignmentsIterator<'b, 'c, F>: Iterator<Item=Vec<OptBool>>
{
    pub fn new(game: &'a SymbolicRegisterGame<F>) -> Self {
        SymbolicRegisterZielonkaSolver {
            game,
            recursive_calls: 0,
        }
    }

    #[tracing::instrument(name = "Run Symbolic Zielonka", skip(self))]
    pub fn run(&mut self) -> SolverOutput {
        let (even, odd) = self.zielonka(self.game).expect("Failed to compute solution");

        let (w_even, w_odd) = self.game.project_winning_regions(&even, &odd).expect("Impossible");

        let mut winners = vec![Owner::Odd; w_odd.len() + w_even.len()];

        for idx in w_even {
            winners[idx as usize] = Owner::Even;
        }

        SolverOutput {
            winners,
            strategy: None,
        }
    }

    /// Run the symbolic solver algorithm and return (W_even, W_odd) (the winning regions of even and odd respectively).
    #[tracing::instrument(name = "Run Symbolic Zielonka", skip(self))]
    pub fn run_symbolic(&mut self) -> (F, F) {
        self.zielonka(self.game).expect("Failed to compute solution")
    }

    fn zielonka(&mut self, game: &SymbolicRegisterGame<F>) -> symbolic::Result<(F, F)> {
        self.recursive_calls += 1;

        // If all the vertices are ignord
        if game.vertices == game.base_false {
            Ok((game.base_false.clone(), game.base_false.clone()))
        } else {
            let d = game.priority_max();
            let attraction_owner = Owner::from_priority(d);
            let starting_set = game.priorities.get(&d).expect("Impossible");
            let attraction_set = game.attractor_priority_set(attraction_owner, starting_set, d)?;

            let sub_game = game.create_subgame(&attraction_set)?;

            let (mut even, mut odd) = self.zielonka(&sub_game)?;
            let (attraction_owner_set, not_attraction_owner_set) = if attraction_owner.is_even() {
                (&mut even, &mut odd)
            } else {
                (&mut odd, &mut even)
            };

            if *not_attraction_owner_set == game.base_false {
                *attraction_owner_set = attraction_owner_set.or(&attraction_set)?;
                Ok((even, odd))
            } else {
                let b_attr = game.attractor_set(attraction_owner.other(), not_attraction_owner_set)?;
                let sub_game = game.create_subgame(&b_attr)?;

                let (mut even, mut odd) = self.zielonka(&sub_game)?;
                let not_attraction_owner_set = if attraction_owner.is_even() {
                    &mut odd
                } else {
                    &mut even
                };
                *not_attraction_owner_set = not_attraction_owner_set.or(&b_attr)?;

                Ok((even, odd))
            }
        }
    }
}

#[cfg(test)]
pub mod test {
    use crate::Owner;
    use crate::symbolic::BDD;
    use crate::symbolic::register_game::SymbolicRegisterGame;

    use super::SymbolicRegisterZielonkaSolver;

    #[test]
    pub fn test_solve_tue_example() {
        let (game, compare) = crate::tests::load_and_compare_example("tue_example.pg");
        let s_pg: SymbolicRegisterGame<BDD> = SymbolicRegisterGame::from_symbolic(&game, 1, Owner::Even).unwrap();
        
        let mut solver = SymbolicRegisterZielonkaSolver::new(&s_pg);
        let solution = solver.run();

        compare(solution);
    }

    #[test]
    pub fn test_solve_two_counters_1_example() {
        let (game, compare) = crate::tests::load_and_compare_example("two_counters_1.pg");
        let s_pg: SymbolicRegisterGame<BDD> = SymbolicRegisterGame::from_symbolic(&game, 2, Owner::Even).unwrap();
        
        let mut solver = SymbolicRegisterZielonkaSolver::new(&s_pg);
        let solution = solver.run();

        compare(solution);
    }

    #[test]
    pub fn test_solve_action_converter() {
        let (game, compare) = crate::tests::load_and_compare_example("ActionConverter.tlsf.ehoa.pg");
        let s_pg: SymbolicRegisterGame<BDD> = SymbolicRegisterGame::from_symbolic(&game, 0, Owner::Even).unwrap();

        let mut solver = SymbolicRegisterZielonkaSolver::new(&s_pg);
        let solution = solver.run();

        compare(solution)
    }
}
