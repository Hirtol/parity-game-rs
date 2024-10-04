use crate::{explicit::{
    solvers::{AttractionComputer, SolverOutput},
    ParityGame, ParityGraph, VertexId,
}, Owner};

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
            attract: AttractionComputer::new(),
        }
    }

    #[tracing::instrument(name = "Run Zielonka", skip(self))]
    // #[profiling::function]
    pub fn run(&mut self) -> SolverOutput {
        let (even, odd) = self.zielonka(self.game);
        let mut winners = vec![Owner::Even; self.game.vertex_count()];
        for idx in odd {
            winners[idx.index()] = Owner::Odd;
        }

        SolverOutput {
            winners,
            strategy: None,
        }
    }

    fn zielonka<T: ParityGraph<u32>>(&mut self, game: &T) -> (Vec<VertexId>, Vec<VertexId>) {
        self.recursive_calls += 1;
        // If all the vertices are ignord
        if game.vertex_count() == 0 {
            (Vec::new(), Vec::new())
        } else {
            let d = game.priority_max();
            let attraction_owner = Owner::from_priority(d);
            let starting_set = game.vertices_index_by_priority(d);
            
            let attraction_set = self.attract.attractor_set_bit_fixed(game, attraction_owner,self.game.vertex_count(), starting_set);

            let sub_game = game.create_subgame_bit(&attraction_set);

            let (mut even, mut odd) = self.zielonka(&sub_game);
            let (attraction_owner_set, not_attraction_owner_set) = if attraction_owner.is_even() {
                (&mut even, &mut odd)
            } else {
                (&mut odd, &mut even)
            };

            if not_attraction_owner_set.is_empty() {
                attraction_owner_set.extend(attraction_set.ones().into_iter().map(|v| VertexId::new(v as usize)));
                (even, odd)
            } else {
                let b_attr = self.attract.attractor_set_bit_fixed(
                    game,
                    attraction_owner.other(),
                    self.game.vertex_count(),
                    not_attraction_owner_set.iter().copied(),
                );
                let sub_game = game.create_subgame_bit(&b_attr);

                let (mut even, mut odd) = self.zielonka(&sub_game);
                let not_attraction_owner_set = if attraction_owner.is_even() {
                    &mut odd
                } else {
                    &mut even
                };
                not_attraction_owner_set.extend(b_attr.ones().into_iter().map(|v| VertexId::new(v as usize)));

                (even, odd)
            }
        }
    }
}

#[cfg(test)]
pub mod test {
    use std::time::Instant;

    use pg_parser::parse_pg;

    use crate::{
        explicit::{solvers::zielonka::ZielonkaSolver, ParityGame},
        tests::{example_dir, load_example},
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
