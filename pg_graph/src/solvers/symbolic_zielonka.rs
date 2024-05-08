use oxidd_core::function::{BooleanFunction, Function};
use crate::{solvers::{SolverOutput}, Owner, symbolic};
use crate::symbolic::{BDD, SymbolicParityGame};
use crate::visualize::DotWriter;

pub struct SymbolicZielonkaSolver<'a> {
    pub recursive_calls: usize,
    game: &'a SymbolicParityGame,
}

impl<'a> SymbolicZielonkaSolver<'a> {
    pub fn new(game: &'a SymbolicParityGame) -> Self {
        SymbolicZielonkaSolver {
            game,
            recursive_calls: 0,
        }
    }

    #[tracing::instrument(name = "Run Symbolic Zielonka", skip(self))]
    // #[profiling::function]
    pub fn run(&mut self) -> SolverOutput {
        let (even, odd) = self.zielonka(self.game).expect("Failed to compute solution");
        
        self.game.gc();
        std::fs::write("out_sym_zielonka.dot", DotWriter::write_dot_symbolic(self.game, [
            (&even, "W_even".into()),
            (&odd, "W_odd".into())
        ]).unwrap());
        // let mut winners = vec![Owner::Even; self.game.vertex_count()];
        // for idx in odd {
        //     winners[idx.index()] = Owner::Odd;
        // }
        // 
        // SolverOutput {
        //     winners,
        //     strategy: None,
        // }
        SolverOutput {
            winners: vec![],
            strategy: None,
        }
    }

    fn zielonka(&mut self, game: &SymbolicParityGame) -> symbolic::Result<(BDD, BDD)> {
        self.recursive_calls += 1;
        // std::fs::write(format!("out_sym_zielonka_{}.dot", self.recursive_calls), DotWriter::write_dot_symbolic(self.game, []).unwrap());
        println!("COUNT: {:?}", game.vertices.node_count());
        
        if self.recursive_calls > 5 {
            panic!();
        }
        
        // If all the vertices are ignord
        if game.vertices == game.base_false {
            Ok((game.base_false.clone(), game.base_false.clone()))
        } else {
            let d = game.priority_max();
            println!("Max priority: {d}");
            let attraction_owner = Owner::from_priority(d);
            let starting_set = game.priorities.get(&d).expect("Impossible");
            let attraction_set = game.attractor_set(attraction_owner, starting_set)?;
            game.gc();
            std::fs::write(format!("out_sym_zielonka_{}.dot", self.recursive_calls), DotWriter::write_dot_symbolic(game, [
                (&attraction_set, "attraction".into())
            ]).unwrap());
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

    use crate::{
        Owner,
        ParityGame,
        solvers::zielonka::ZielonkaSolver, tests::example_dir,
    };
    use crate::solvers::symbolic_zielonka::SymbolicZielonkaSolver;
    use crate::symbolic::SymbolicParityGame;

    #[test]
    pub fn test_solve_tue_example() {
        let input = std::fs::read_to_string(example_dir().join("tue_example.pg")).unwrap();
        let pg = parse_pg(&mut input.as_str()).unwrap();
        let game = ParityGame::new(pg).unwrap();
        let s_pg = SymbolicParityGame::from_explicit(&game).unwrap();
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
