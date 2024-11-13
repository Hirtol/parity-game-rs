use crate::explicit::{BitsetExtensions, SubGame, VertexSet};
use crate::{explicit::{
    solvers::{AttractionComputer, SolverOutput},
    ParityGame, ParityGraph,
}, Owner, Priority};
use fixedbitset::specific::SubBitSet;
use itertools::Itertools;
use std::borrow::Cow;

pub struct ZielonkaSolver<'a> {
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
            iterations: 0,
            attract: AttractionComputer::new(game.vertex_count()),
            min_precision: if game.graph_edges().any(|(source, target)| source == target) { 0 } else { 1 },
        }
    }

    #[tracing::instrument(name = "Run Zielonka", skip(self))]
    // #[profiling::function]
    pub fn run(&mut self) -> SolverOutput {
        crate::debug!("Searching with min_precision: {}", self.min_precision);
        let (even, odd) = self.zielonka(&mut self.game.create_subgame([]), self.game.vertex_count(), self.game.vertex_count());
        tracing::debug!("Even Winning: {:?}", even.printable_vertices());
        tracing::debug!("Odd Winning: {:?}", odd.printable_vertices());
        // if even.count_ones(..) + odd.count_ones(..) < self.game.vertex_count() {
        //     panic!("Fewer vertices than expected were in the winning regions");
        // }
        SolverOutput::from_winning(self.game.vertex_count(), &odd)
    }
    
    fn zielonka<T: ParityGraph<u32>>(&mut self, game: &mut SubGame<u32, T>, precision_even: usize, precision_odd: usize) -> (VertexSet, VertexSet) {
        // If all the vertices are ignored
        if game.vertex_count() == 0 {
            (VertexSet::default(), VertexSet::default())
        } else {
            let d = game.priority_max();
            let region_owner = Owner::from_priority(d);
            let (new_p_even, new_p_odd) = match region_owner {
                Owner::Even => if precision_even <= self.min_precision {
                    crate::trace!("Hitting early exit even");
                    return (VertexSet::empty_game(game), VertexSet::empty_game(game));
                } else {
                    (precision_even, precision_odd / 2)
                }
                Owner::Odd => if precision_odd <= self.min_precision {
                    crate::trace!("Hitting early exit odd");
                    return (VertexSet::empty_game(game), VertexSet::empty_game(game));
                } else {
                    (precision_even / 2, precision_odd)
                }
            };

            let (mut result_even, mut result_odd) = (VertexSet::empty_game(game), VertexSet::empty_game(game));
            crate::debug!("Starting: {d}");
            // Half precision calls
            while self.zielonka_step(game, d, &mut result_even, &mut result_odd, new_p_even, new_p_odd) {}

            // If the amount of vertices that remain are less than half our original precision then we know that no
            // winning region of size > half the vertices can exist, and we can safely stop here.
            let can_skip = match region_owner {
                Owner::Even => game.vertex_count() < new_p_odd,
                Owner::Odd => game.vertex_count() < new_p_even
            };
            if can_skip {
                crate::debug!(n = game.vertex_count(), new_p_odd, new_p_even, "Returning early, less than half remain");
                match region_owner {
                    Owner::Even => {
                        result_even.union_with(&game.game_vertices)
                    }
                    Owner::Odd => {
                        result_odd.union_with(&game.game_vertices)
                    }
                }
                return (result_even, result_odd);
            }

            // Full precision call
            crate::debug!("Full Precision: {d}");
            let discovered_dominion = self.zielonka_step(game, d, &mut result_even, &mut result_odd, precision_even, precision_odd);

            if discovered_dominion {
                crate::debug!("Discovered greater dominion ({d}), continuing");
                // If we found a dominion for the opposing player then we need to do a new sequence of half-precision calls
                while self.zielonka_step(game, d, &mut result_even, &mut result_odd, new_p_even, new_p_odd) {}
            } else {
                crate::debug!("Did not discover greater dominion ({d}), returning: {:?} == {:?}", result_even.printable_vertices(), result_odd.printable_vertices());
                // return (result_even, result_odd);
            }
            
            match region_owner {
                Owner::Even => {
                    result_even.union_with(&game.game_vertices)
                }
                Owner::Odd => {
                    result_odd.union_with(&game.game_vertices)
                }
            }
            crate::debug!("Ending: {d}");
            
            (result_even, result_odd)
        }
    }

    /// Perform one step in the Zielonka algorithm, passing through the parameters to a recursive `zielonka` call.
    ///
    /// Will return `true` if another step with the same parameters needs to be performed, or `false` otherwise.
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
            crate::debug!("Alpha {d}\n{:?}", attraction_set.printable_vertices());
            // attraction_owner_set.union_with(&attraction_set);

            // if attraction_owner.is_even() {
            //     result_even.union_with(attraction_owner_set);
            // } else {
            //     result_odd.union_with(attraction_owner_set);
            // }
            // Not sure if we should shrink the sub-game here or not.
            // As we've found a locally closed region it would make sense to, mirroring the case of `!not_attraction_owner_set.is_clear()`,
            // but it also shrinks the total iterative calls substantially to the point where it feels like it might be incorrect.
            // We get valid results either way with or without the below, but if we stumble upon a magical counter-example consider removing this call.
            // game.shrink_subgame(attraction_owner_set);
            false
        } else {
            let b_attr = self.attract.attractor_set_bit(
                game,
                attraction_owner.other(),
                Cow::Borrowed(not_attraction_owner_set),
            );
            crate::debug!("BATTR {d}\n{:?}", b_attr.printable_vertices());

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
    // #[tracing_test::traced_test]
    pub fn verify_correctness() {
        for name in tests::examples_iter() {
            if name.contains("two_counters_14p") {
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
