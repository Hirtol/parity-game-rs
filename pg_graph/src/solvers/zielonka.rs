use std::collections::{VecDeque};

use itertools::Itertools;

use crate::{solvers::SolverOutput, Owner, ParityGame, VertexId};

pub struct ZielonkaSolver<'a> {
    game: &'a ParityGame,
    pub recursive_calls: usize,
}

impl<'a> ZielonkaSolver<'a> {
    pub fn new(game: &'a ParityGame) -> Self {
        ZielonkaSolver { game, recursive_calls: 0 }
    }

    #[tracing::instrument(name = "Run Zielonka", skip(self))]
    // #[profiling::function]
    pub fn run(&mut self) -> SolverOutput {
        let (even, odd) = self.zielonka(ahash::HashSet::default());
        let mut winners = vec![Owner::Even; self.game.vertex_count()];
        for idx in odd {
            winners[idx.index()] = Owner::Odd;
        }
        
        SolverOutput {
            winners,
            strategy: None,
        }
    }

    fn zielonka(&mut self, ignored: ahash::HashSet<VertexId>) -> (Vec<VertexId>, Vec<VertexId>) {
        self.recursive_calls += 1;
        // If all the vertices are ignord
        if ignored.len() == self.game.vertex_count() {
            (Vec::new(), Vec::new())
        } else {
            let d = self
                .game
                .vertices_index()
                .filter(|v| !ignored.contains(v))
                .flat_map(|v| self.game.get(v))
                .map(|v| v.priority)
                .max().unwrap_or_default();
            let attraction_owner = Owner::from_priority(d);
            let starting_set = self.game.vertices_by_priority_idx(d).filter(|(idx, _)| !ignored.contains(idx)).map(|(idx, _)| idx);
            let attraction_set = attractor_set(attraction_owner, starting_set, &ignored, self.game);
            
            let mut ignore_set = ignored.clone();
            ignore_set.extend(&attraction_set);
            
            let (mut even, mut odd) = self.zielonka(ignore_set);
            let (attraction_owner_set, not_attraction_owner_set) = if attraction_owner.is_even() {
                (&mut even, &mut odd)
            } else {
                (&mut odd, &mut even)
            };
            
            if not_attraction_owner_set.is_empty() {
                attraction_owner_set.extend(attraction_set);
                (even, odd)
            } else {
                let b_attr = attractor_set(attraction_owner.other(), not_attraction_owner_set.iter().copied(), &ignored, self.game);
                let mut ignore_set = ignored.clone();
                ignore_set.extend(&b_attr);
                let (mut even, mut odd) = self.zielonka(ignore_set);
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

pub fn attractor_set(
    player: Owner,
    starting_set: impl IntoIterator<Item = VertexId>,
    ignored: &ahash::HashSet<VertexId>,
    game: &ParityGame,
) -> ahash::HashSet<VertexId> {
    let mut attract_set = ahash::HashSet::from_iter(starting_set);
    let mut explore_queue = VecDeque::from_iter(attract_set.iter().copied());

    while let Some(next_item) = explore_queue.pop_back() {
        for predecessor in game.predecessors(next_item).filter(|p| !ignored.contains(p)) {
            let vertex = &game[predecessor];
            let should_add = if vertex.owner == player {
                // *any* edge needs to lead to the attraction set, since this is a predecessor of an item already in the attraction set we know that already!
                true
            } else {
                game.edges(predecessor).filter(|p| !ignored.contains(p)).all(|v| attract_set.contains(&v))
            };

            // Only add to the attraction set if we should
            if should_add && attract_set.insert(predecessor) {
                explore_queue.push_back(predecessor);
            }
        }
    }

    attract_set
}

#[cfg(test)]
pub mod test {
    use std::collections::HashSet;
    use std::time::Instant;
    use pg_parser::parse_pg;

    use crate::{Owner, ParityGame, VertexId};
    use crate::solvers::small_progress::SmallProgressSolver;
    use crate::solvers::zielonka::ZielonkaSolver;
    use crate::tests::example_dir;

    #[test]
    pub fn test_attract_set_computation() -> eyre::Result<()> {
        let mut pg = r#"parity 3;
0 1 1 0,1 "0";
1 1 0 2 "1";
2 2 0 2 "2";"#;
        let pg = pg_parser::parse_pg(&mut pg).unwrap();
        let pg = ParityGame::new(pg)?;

        let set = super::attractor_set(Owner::Even, [VertexId::new(2)], &Default::default(), &pg);

        assert_eq!(set, HashSet::from_iter(vec![VertexId::new(2), VertexId::new(1)]));

        let mut solver = ZielonkaSolver::new(&pg);
        
        println!("{:#?}", solver.run());
        
        Ok(())
    }

    #[test]
    pub fn test_solve_tue_example() {
        let input = std::fs::read_to_string(example_dir().join("tue_example.pg")).unwrap();
        let pg = parse_pg(&mut input.as_str()).unwrap();
        let game = ParityGame::new(pg).unwrap();
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
        assert_eq!(solution, vec![
            Owner::Even,
            Owner::Odd,
            Owner::Even,
            Owner::Even,
            Owner::Even,
            Owner::Even,
            Owner::Odd,
            Owner::Odd,
            Owner::Even,
        ])
    }
}
