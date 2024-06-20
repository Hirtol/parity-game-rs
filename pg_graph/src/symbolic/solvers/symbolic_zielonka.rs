use oxidd_core::function::{BooleanFunction, Function};
use oxidd_core::util::OptBool;
use oxidd_core::WorkerManager;

use crate::{
    explicit::solvers::SolverOutput,
    Owner,
    symbolic,
    symbolic::{BDD, parity_game::SymbolicParityGame},
};
use crate::symbolic::oxidd_extensions::GeneralBooleanFunction;
use crate::symbolic::sat::TruthAssignmentsIterator;

pub struct SymbolicZielonkaSolver<'a, F: GeneralBooleanFunction> {
    pub recursive_calls: usize,
    game: &'a SymbolicParityGame<F>,
}

impl<'a, F: GeneralBooleanFunction> SymbolicZielonkaSolver<'a, F>
    where for<'id> F::Manager<'id>: WorkerManager,
          for<'b, 'c> TruthAssignmentsIterator<'b, 'c, F>: Iterator<Item=Vec<OptBool>> {
    pub fn new(game: &'a SymbolicParityGame<F>) -> Self {
        SymbolicZielonkaSolver {
            game,
            recursive_calls: 0,
        }
    }

    #[tracing::instrument(name = "Run Symbolic Zielonka", skip(self))]
    // #[profiling::function]
    pub fn run(&mut self) -> SolverOutput {
        let (even, odd) = self.zielonka(self.game).expect("Failed to compute solution");

        let even = self.game.vertices_of_bdd(&even);

        let mut winners = vec![Owner::Odd; self.game.vertex_count()];

        for idx in even {
            winners[idx.index()] = Owner::Even;
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

    fn zielonka(&mut self, game: &SymbolicParityGame<F>) -> symbolic::Result<(F, F)> {
        self.recursive_calls += 1;

        // If all the vertices are ignord
        if game.vertices == game.base_false {
            Ok((game.base_false.clone(), game.base_false.clone()))
        } else {
            let d = game.priority_max();
            let attraction_owner = Owner::from_priority(d);
            let starting_set = game.priorities.get(&d).expect("Impossible");
            let attraction_set = game.attractor_set(attraction_owner, starting_set)?;

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
    use std::time::Instant;

    use pg_parser::parse_pg;

    use crate::{explicit::ParityGame, Owner, symbolic::parity_game::SymbolicParityGame, tests::example_dir};

    use super::SymbolicZielonkaSolver;

    #[test]
    pub fn test_solve_tue_example() {
        let input = std::fs::read_to_string(example_dir().join("tue_example.pg")).unwrap();
        let pg = parse_pg(&mut input.as_str()).unwrap();
        let game = ParityGame::new(pg).unwrap();
        let s_pg = SymbolicParityGame::from_explicit_bdd(&game).unwrap();
        let mut solver = SymbolicZielonkaSolver::new(&s_pg);

        let solution = solver.run().winners;

        println!("Solution: {:#?}", solution);

        assert!(solution.iter().all(|win| *win == Owner::Odd));
    }

    #[test]
    pub fn test_solve_action_converter() {
        let input = std::fs::read_to_string(example_dir().join("ActionConverter.tlsf.ehoa.pg")).unwrap();
        let pg = parse_pg(&mut input.as_str()).unwrap();
        let game = ParityGame::new(pg).unwrap();
        let s_pg = SymbolicParityGame::from_explicit_bdd(&game).unwrap();
        let mut solver = SymbolicZielonkaSolver::new(&s_pg);

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
