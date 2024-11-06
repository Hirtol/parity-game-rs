use crate::explicit::{BitsetExtensions, SubGame, VertexSet};
use crate::{explicit::{
    solvers::{AttractionComputer, SolverOutput},
    ParityGame, ParityGraph,
}, Owner, Priority};
use itertools::Itertools;
use std::borrow::Cow;

pub struct ZielonkaSolver<'a> {
    pub recursive_calls: usize,
    pub iterations: usize,
    game: &'a ParityGame<u32>,
    attract: AttractionComputer<u32>,
    /// The minimum precision needed before we can return early.
    /// 
    /// If the game has no self-loops this can be `1`, otherwise it needs to be `0`.
    min_precision: usize,
}

impl<'a> ZielonkaSolver<'a> {
    pub fn new(game: &'a ParityGame<u32>) -> Self {
        ZielonkaSolver {
            game,
            recursive_calls: 0,
            attract: AttractionComputer::new(game.vertex_count()),
            min_precision: if game.graph_edges().any(|(source, target)| source == target) { 0 } else { 1 },
        }
    }

    #[tracing::instrument(name = "Run Zielonka", skip(self))]
    // #[profiling::function]
    pub fn run(&mut self) -> SolverOutput {
        tracing::debug!("Searching with min_precision: {}", self.min_precision);
        let (_, odd) = self.zielonka(&mut self.game.create_subgame([]), self.game.vertex_count(), self.game.vertex_count());
        SolverOutput::from_winning(self.game.vertex_count(), &odd)
    }
    
    fn zielonka<T: ParityGraph<u32>>(&mut self, game: &mut SubGame<u32, T>, precision_even: usize, precision_odd: usize) -> (VertexSet, VertexSet) {
        self.recursive_calls += 1;
        // If all the vertices are ignored
        if game.vertex_count() == 0 {
            (VertexSet::empty_game(game), VertexSet::empty_game(game))
        } else {
            let d = game.priority_max();
            let (new_p_even, new_p_odd) = match Owner::from_priority(d) {
                Owner::Even => if precision_even <= self.min_precision {
                    println!("Hitting early exit even");
                    return (VertexSet::empty_game(game), VertexSet::empty_game(game));
                } else {
                    (precision_even, precision_odd / 2)
                }
                Owner::Odd => if precision_odd <= self.min_precision {
                    println!("Hitting early exit odd");
                    return (VertexSet::empty_game(game), VertexSet::empty_game(game));
                } else {
                    (precision_even / 2, precision_odd)
                }
            };
            // let (new_p_even, new_p_odd) = (precision_even, precision_odd);
            let (mut result_even, mut result_odd) = (VertexSet::empty_game(game), VertexSet::empty_game(game));
            tracing::debug!("Starting: {d}");
            // Half precision calls
            loop {
                let attraction_owner = Owner::from_priority(d);
                let starting_set = game.vertices_index_by_priority(d);

                let attraction_set = self.attract.attractor_set(game, attraction_owner, starting_set);
                let mut sub_game = game.create_subgame_bit(&attraction_set);

                let (mut even, mut odd) = self.zielonka(&mut sub_game, new_p_even, new_p_odd);

                let (attraction_owner_set, not_attraction_owner_set) = if attraction_owner.is_even() {
                    (&mut even, &mut odd)
                } else {
                    (&mut odd, &mut even)
                };
                
                if not_attraction_owner_set.is_clear() {
                    tracing::debug!("Alpha {d}\n{:?}", attraction_set.printable_vertices());
                    attraction_owner_set.union_with(&attraction_set);

                    if attraction_owner.is_even() {
                        result_even.union_with(attraction_owner_set);
                    } else {
                        result_odd.union_with(attraction_owner_set);
                    }
                    // game.shrink_subgame(&attraction_set);
                    break;
                } else {
                    let b_attr = self.attract.attractor_set_bit(
                        game,
                        attraction_owner.other(),
                        Cow::Borrowed(not_attraction_owner_set),
                    );
                    tracing::debug!("BATTR {d}\n{:?}", b_attr.printable_vertices());

                    let not_attraction_owner_set = if attraction_owner.is_even() {
                        &mut result_odd
                    } else {
                        &mut result_even
                    };
                    // Note the winning region of the opponent.
                    not_attraction_owner_set.union_with(&b_attr);
                    // Shrink the game for the next loop
                    game.shrink_subgame(&b_attr);
                }
            };

            tracing::debug!("Full Precision: {d}");

            // Full precision call
            let discovered_dominion = {
                let attraction_owner = Owner::from_priority(d);
                let starting_set = game.vertices_index_by_priority(d);

                let attraction_set = self.attract.attractor_set(game, attraction_owner, starting_set);
                let mut sub_game = game.create_subgame_bit(&attraction_set);

                let (mut even, mut odd) = self.zielonka(&mut sub_game, precision_even, precision_odd);

                let (attraction_owner_set, not_attraction_owner_set) = if attraction_owner.is_even() {
                    (&mut even, &mut odd)
                } else {
                    (&mut odd, &mut even)
                };

                if not_attraction_owner_set.is_clear() {
                    tracing::debug!("Alpha {d}\n{:?}", attraction_set.printable_vertices());
                    attraction_owner_set.union_with(&attraction_set);

                    if attraction_owner.is_even() {
                        result_even.union_with(attraction_owner_set);
                    } else {
                        result_odd.union_with(attraction_owner_set);
                    }
                    // game.shrink_subgame(&attraction_set);

                    false
                } else {
                    let b_attr = self.attract.attractor_set_bit(
                        game,
                        attraction_owner.other(),
                        Cow::Borrowed(not_attraction_owner_set),
                    );
                    tracing::debug!("BATTR {d}\n{:?}", b_attr.printable_vertices());
                    let not_attraction_owner_set = if attraction_owner.is_even() {
                        &mut result_odd
                    } else {
                        &mut result_even
                    };
                    // Note the winning region of the opponent.
                    not_attraction_owner_set.union_with(&b_attr);
                    // Shrink the game for the next loop
                    game.shrink_subgame(&b_attr);
                    true
                }
            };

            tracing::debug!("Ending: {d}");

            if !discovered_dominion {
                tracing::debug!("Did not discover greater dominion, returning: {:?} == {:?}", result_even.printable_vertices(), result_odd.printable_vertices());
                return (result_even, result_odd);
            } else {
                tracing::debug!("Discovered greater dominion, continuing");
            }

            // If we found a dominion for the opposing player then we need to do a new sequence of half-precision calls
            loop {
                let attraction_owner = Owner::from_priority(d);
                let starting_set = game.vertices_index_by_priority(d);

                let attraction_set = self.attract.attractor_set(game, attraction_owner, starting_set);
                let mut sub_game = game.create_subgame_bit(&attraction_set);

                let (mut even, mut odd) = self.zielonka(&mut sub_game, new_p_even, new_p_odd);

                let (attraction_owner_set, not_attraction_owner_set) = if attraction_owner.is_even() {
                    (&mut even, &mut odd)
                } else {
                    (&mut odd, &mut even)
                };

                if not_attraction_owner_set.is_clear() {
                    attraction_owner_set.union_with(&attraction_set);

                    if attraction_owner.is_even() {
                        result_even.union_with(attraction_owner_set);
                    } else {
                        result_odd.union_with(attraction_owner_set);
                    }
                    // Unnecessary as this concludes the computation, but for clarity.
                    // game.shrink_subgame(&attraction_set);

                    break;
                } else {
                    let b_attr = self.attract.attractor_set_bit(
                        game,
                        attraction_owner.other(),
                        Cow::Borrowed(not_attraction_owner_set),
                    );
                    let not_attraction_owner_set = if attraction_owner.is_even() {
                        &mut result_odd
                    } else {
                        &mut result_even
                    };
                    // Note the winning region of the opponent.
                    not_attraction_owner_set.union_with(&b_attr);
                    // Shrink the game for the next loop
                    game.shrink_subgame(&b_attr);
                }
            };

            
            (result_even, result_odd)
        }
    }
    
    #[inline]
    fn zielonka_step<T: ParityGraph<u32>>(&mut self, game: &mut SubGame<u32, T>,
                                          d: Priority,
                                          result_even: &mut VertexSet,
                                          result_odd: &mut VertexSet,
                                          new_p_even: usize,
                                          new_p_odd: usize) -> bool {
        self.iterations += 1;

        let attraction_owner = Owner::from_priority(d);
        let starting_set = game.vertices_index_by_priority(d);

        let attraction_set = self.attract.attractor_set(game, attraction_owner, starting_set);
        let mut sub_game = game.create_subgame_bit(&attraction_set);

        let (mut even, mut odd) = self.zielonka(&mut sub_game, new_p_even, new_p_odd);

        let (attraction_owner_set, not_attraction_owner_set) = if attraction_owner.is_even() {
            (&mut even, &mut odd)
        } else {
            (&mut odd, &mut even)
        };

        if not_attraction_owner_set.is_clear() {
            tracing::debug!("Alpha {d}\n{:?}", attraction_set.printable_vertices());
            attraction_owner_set.union_with(&attraction_set);

            if attraction_owner.is_even() {
                result_even.union_with(attraction_owner_set);
            } else {
                result_odd.union_with(attraction_owner_set);
            }
            // game.shrink_subgame(&attraction_set);
            false
        } else {
            let b_attr = self.attract.attractor_set_bit(
                game,
                attraction_owner.other(),
                Cow::Borrowed(not_attraction_owner_set),
            );
            tracing::debug!("BATTR {d}\n{:?}", b_attr.printable_vertices());

            let not_attraction_owner_set = if attraction_owner.is_even() {
                result_odd
            } else {
                result_even
            };
            // Note the winning region of the opponent.
            not_attraction_owner_set.union_with(&b_attr);
            // Shrink the game for the next loop
            game.shrink_subgame(&b_attr);
            true
        }
        
    }
}

#[cfg(test)]
pub mod test {
    use crate::explicit::solvers::qpt_zielonka::ZielonkaSolver;
    use crate::{tests, tests::load_example, Owner};
    use std::time::Instant;

    #[test]
    pub fn verify_correctness() {
        for name in tests::examples_iter() {
            if name.contains("two_counters") {
                continue;
            }
            println!("Running test for: {name}...");
            let (game, compare) = tests::load_and_compare_example(&name);
            let mut qpt_zielonka = ZielonkaSolver::new(&game);
            let solution = qpt_zielonka.run();
            compare(solution);
            println!("{name} correct!")
        }
    }

    #[test]
    pub fn test_solve_basic_paper_example() {
        let (game, compare) = tests::load_and_compare_example("basic_paper_example.pg");
        let mut solver = ZielonkaSolver::new(&game);

        let solution = solver.run();

        println!("Solution: {:#?}", solution);

        compare(solution);
    }

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
