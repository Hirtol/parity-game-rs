use crate::explicit::solvers::tangle_learning::{PearceTangleScc, Tangle, TangleCollection, TangleManager, NO_STRATEGY};
use crate::explicit::solvers::Dominion;
use crate::explicit::{BitsetExtensions, SubGame, VertexId, VertexSet};
use crate::{explicit::{
    solvers::{AttractionComputer, SolverOutput},
    ParityGame, ParityGraph,
}, Owner, Priority};
use std::borrow::Cow;

pub struct ZielonkaSolver<'a> {
    pub iterations: usize,
    game: &'a ParityGame<u32>,
    attract: AttractionComputer<u32>,
    /// The minimum precision needed before we can return early.
    /// 
    /// If the game has no self-loops this can be `1`, otherwise it needs to be `0`.
    min_precision: usize,
    pearce: PearceTangleScc,
    tangles: TangleManager,
    strategy: Vec<VertexId>,
    pub tangles_found: u32,
}

impl<'a> ZielonkaSolver<'a> {
    pub fn new(game: &'a ParityGame<u32>) -> Self {
        ZielonkaSolver {
            game,
            iterations: 0,
            attract: AttractionComputer::new(game.vertex_count()),
            min_precision: if game.graph_edges().any(|(source, target)| source == target) { 0 } else { 1 },
            pearce: PearceTangleScc::new(game.vertex_count()),
            tangles: TangleManager::default(),
            strategy: vec![VertexId::new(NO_STRATEGY as usize); game.vertex_count()],
            tangles_found: 0,
        }
    }

    #[tracing::instrument(name = "Run Zielonka", skip(self))]
    // #[profiling::function]
    pub fn run(&mut self) -> SolverOutput {
        crate::debug!("Searching with min_precision: {}", self.min_precision);
        let mut strategy = vec![VertexId::new(NO_STRATEGY as usize); self.game.vertex_count()];
        let ((even, odd), _) = self.zielonka(&mut self.game.create_subgame([]), self.game.vertex_count(), self.game.vertex_count(), strategy.as_mut_slice());
        tracing::debug!("Discovered: {} tangles", self.tangles_found);
        // tracing::debug!("Even Winning: {:?}", even.printable_vertices());
        // tracing::debug!("Odd Winning: {:?}", odd.printable_vertices());
        if even.count_ones(..) + odd.count_ones(..) < self.game.vertex_count() {
            panic!("Fewer vertices than expected were in the winning regions");
        }
        SolverOutput::from_winning(self.game.vertex_count(), &odd)
    }

    /// Returns the winning regions `(W_even, W_odd)` as well as a flag indicating whether the results that were obtained
    /// used an early cut-off using the `precision_even/odd` parameters (`false` if so).
    fn zielonka<T: ParityGraph<u32>>(&mut self, game: &mut SubGame<u32, T>, precision_even: usize, precision_odd: usize, strategy: &mut [VertexId]) -> ((VertexSet, VertexSet), bool) {
        // If all the vertices are ignored
        if game.vertex_count() == 0 {
            ((VertexSet::default(), VertexSet::default()), true)
        } else {
            let d = game.priority_max();
            let region_owner = Owner::from_priority(d);
            let (new_p_even, new_p_odd) = match region_owner {
                Owner::Even => if precision_even <= self.min_precision {
                    crate::debug!(d, "Hitting early exit even");
                    return ((VertexSet::empty_game(game), VertexSet::empty_game(game)), false);
                } else {
                    (precision_even, precision_odd / 2)
                }
                Owner::Odd => if precision_odd <= self.min_precision {
                    crate::debug!(d, "Hitting early exit odd");
                    return ((VertexSet::empty_game(game), VertexSet::empty_game(game)), false);
                } else {
                    (precision_even / 2, precision_odd)
                }
            };

            let mut authentic_result = true;
            let (mut result_even, mut result_odd) = (VertexSet::empty_game(game), VertexSet::empty_game(game));
            crate::debug!("Starting: {d}");
            // Half precision calls
            while self.zielonka_step(game, d, &mut result_even, &mut result_odd, new_p_even, new_p_odd, &mut authentic_result, strategy) {}

            // If the amount of vertices that remain are less than half our original precision then we know that no
            // winning region of size > half the vertices can exist, and we can safely stop here.
            let can_skip = match region_owner {
                Owner::Even => game.vertex_count() <= new_p_odd,
                Owner::Odd => game.vertex_count() <= new_p_even
            };
            if !can_skip {
                // Full precision call
                crate::debug!("Full Precision: {d}");
                let discovered_dominion = self.zielonka_step(game, d, &mut result_even, &mut result_odd, precision_even, precision_odd, &mut authentic_result, strategy);

                if discovered_dominion {
                    crate::debug!("Discovered greater dominion ({d}), continuing");
                    // If we found a dominion for the opposing player then we need to do a new sequence of half-precision calls
                    while self.zielonka_step(game, d, &mut result_even, &mut result_odd, new_p_even, new_p_odd, &mut authentic_result, strategy) {}
                } else {
                    crate::debug!("Did not discover greater dominion ({d})");
                }
            } else {
                crate::debug!(d, n = game.vertex_count(), new_p_odd, new_p_even, "Returning early, less than half remain");
            }

            // Whatever remains is guaranteed _not_ to be an opponent's dominion, so we can add it to our winning region.
            // Since we've done both our full-precision and half-precision calls we can make the assumption that it is indeed won by us.
            crate::debug!(remains =?game.game_vertices.printable_vertices(), "Combining remaining game with winning region");
            match region_owner {
                Owner::Even => {
                    result_even.union_with(&game.game_vertices)
                }
                Owner::Odd => {
                    result_odd.union_with(&game.game_vertices)
                }
            }
            crate::debug!("Ending: {d}, returning: {:?} == {:?}", result_even.printable_vertices(), result_odd.printable_vertices());

            ((result_even, result_odd), authentic_result)
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
                                          new_p_odd: usize,
                                          authentic: &mut bool,
                                          strategy: &mut [VertexId]
    ) -> bool {
        self.iterations += 1;

        let attraction_owner = Owner::from_priority(d);
        let starting_set = AttractionComputer::make_starting_set(game, game.vertices_index_by_priority(d));
        // Reset strategy of top vertices.
        for v in starting_set.ones() {
            strategy[v] = VertexId::new(NO_STRATEGY as usize);
        }

        let tangle_attractor = self.attract.attractor_set_tangle(game, attraction_owner, Cow::Owned(starting_set), &self.tangles, strategy);
        // Check if there are any escapes from this region, or if it's locally closed.
        // We only care about the vertices with the region priority.
        let leaks = tangle_attractor
            .ones_vertices()
            .filter(|v| game.priority(*v) == d)
            .any(|v| {
                if game.owner(v) == attraction_owner {
                    self.strategy[v.index()].index() == NO_STRATEGY as usize
                } else {
                    game
                        .edges(v)
                        .any(|succ| !tangle_attractor.contains(succ.index()))
                }
            });
        crate::debug!(?leaks, "Leaks");
        
        // If there are any leaks from our region then it's not worth finding tangles.
        let mut temp_tangles = TangleCollection::default();
        if !leaks {
            let tangle_subgame = SubGame::from_vertex_set(game.parent, tangle_attractor.clone());
            let dominion = self.extract_tangles(game.parent, &tangle_subgame, d, strategy, &mut temp_tangles);
        }

        let mut sub_game = game.create_subgame_bit(&tangle_attractor);

        let ((mut even, mut odd), recursive_authentic) = self.zielonka(&mut sub_game, new_p_even, new_p_odd, strategy);
        *authentic &= recursive_authentic;
        self.tangles.collection.merge(temp_tangles);

        let (attraction_owner_set, not_attraction_owner_set) = if attraction_owner.is_even() {
            (&mut even, &mut odd)
        } else {
            (&mut odd, &mut even)
        };

        if not_attraction_owner_set.is_clear() {
            crate::debug!(recursive_authentic, "Alpha {d} - Attractor: {:?}", tangle_attractor.printable_vertices());
            // We can be sure of our result, no undetermined vertices.
            // This is only `true` if we didn't cut our recursive call tree off prematurely.
            if recursive_authentic {
                attraction_owner_set.union_with(&tangle_attractor);

                if attraction_owner.is_even() {
                    result_even.union_with(attraction_owner_set);
                } else {
                    result_odd.union_with(attraction_owner_set);
                }

                game.shrink_subgame(attraction_owner_set);
            }

            false
        } else {
            let b_attr = self.attract.attractor_set_tangle(
                game,
                attraction_owner.other(),
                Cow::Borrowed(not_attraction_owner_set),
                &self.tangles,
                strategy
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

    fn extract_tangles<T: ParityGraph<u32>>(
        &mut self,
        current_game: &T,
        tangle_sub_game: &SubGame<u32, T>,
        region_priority: Priority,
        strategy: &mut [VertexId<u32>],
        tangles: &mut TangleCollection,
    ) -> Option<Dominion> {
        let top_vertices = tangle_sub_game.vertices_index_by_priority(region_priority);
        let mut dominion: Option<Dominion> = None;
        let region_owner = Owner::from_priority(region_priority);

        self.pearce.run(
            tangle_sub_game,
            region_owner,
            top_vertices,
            strategy,
            |tangle, top_vertex| {
                // Ensure non-trivial SCC (more than one vertex OR self-loop)
                let tangle_size = tangle.len();
                let is_tangle = tangle_size != 1
                    || strategy[top_vertex.index()] == top_vertex
                    || current_game.has_edge(top_vertex, top_vertex);
                if !is_tangle {
                    crate::trace!("Skipping trivial tangle");
                    return;
                }

                let id = self.tangles_found;
                // Count all escapes that remain in the GREATER unsolved game
                let final_tangle = AttractionComputer::make_starting_set(current_game, tangle.copied());
                let mut target_escapes = Vec::new();

                for v in final_tangle.ones_vertices() {
                    // Don't care about alpha-vertices, as they are guaranteed not to be leaks due to our earlier checks,
                    // and can simply choose to remain in the tangle (else it wouldn't be an SCC)
                    if current_game.owner(v) != region_owner {
                        for target in current_game.edges(v) {
                            // Only mark as escapes those vertices which actually _are_ escapes.
                            if !final_tangle.contains(target.index()) && !target_escapes.contains(&target) {
                                target_escapes.push(target);
                            }
                        }
                    }
                }

                // Check if there are any escapes in our greater game, if not we already know it's a dominion.
                // We can then just merge it with other found dominating tangles this cycle
                if target_escapes.is_empty() {
                    tracing::info!(?tangle_size, "Found a tangle which was a dominion");

                    if let Some(existing) = &mut dominion {
                        existing.vertices.union_with(&final_tangle);
                    } else {
                        dominion = Some(Dominion {
                            dominating_p: region_priority,
                            vertices: final_tangle,
                        })
                    }
                } else {
                    target_escapes.sort_unstable();
                    let targets = self
                        .tangles
                        .escape_list
                        .push_collection_itr(target_escapes.iter().map(|v| v.index()));

                    let new_tangle = Tangle {
                        id,
                        owner: region_owner,
                        priority: region_priority,
                        vertices: final_tangle,
                        escape_set: targets,
                    };

                    crate::info!(tangle_size, ?new_tangle, "Found new tangle");

                    self.tangles_found += 1;
                    tangles.insert_tangle(new_tangle);
                }
            },
        );

        dominion
    }
}

#[cfg(test)]
pub mod test {
    use crate::explicit::solvers::qpt_tangle_zielonka::ZielonkaSolver;
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
