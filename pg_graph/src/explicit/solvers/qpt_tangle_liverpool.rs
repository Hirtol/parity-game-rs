use crate::explicit::solvers::tangle_learning::{TangleManager, NO_STRATEGY};
use crate::explicit::solvers::Dominion;
use crate::explicit::{BitsetExtensions, OptimisedGraph, SubGame, VertexId, VertexSet};
use crate::{explicit::{
    solvers::{AttractionComputer, SolverOutput},
    ParityGame, ParityGraph,
}, Owner, Priority};
use std::borrow::Cow;

pub struct LiverpoolSolver<'a> {
    pub iterations: usize,
    pub tangles: TangleManager,
    game: &'a ParityGame<u32>,
    attract: AttractionComputer<u32>,
    strategy: Vec<VertexId<u32>>,
    /// The minimum precision needed before we can return early.
    ///
    /// If the game has no self-loops this can be `1`, otherwise it needs to be `0`.
    min_precision: usize,
    /// For quickly returning without doing tangle analysis in trivial cases.
    max_priority: Priority,
}

impl<'a> LiverpoolSolver<'a> {
    pub fn new(game: &'a ParityGame<u32>) -> Self {
        LiverpoolSolver {
            game,
            iterations: 0,
            attract: AttractionComputer::new(game.vertex_count()),
            strategy: vec![VertexId::new(NO_STRATEGY as usize); game.vertex_count()],
            min_precision: if game.graph_edges().any(|(source, target)| source == target) { 0 } else { 1 },
            tangles: TangleManager::new(game.vertex_count()),
            max_priority: 0,
        }
    }

    #[tracing::instrument(name = "Run Liverpool Tangle Zielonka", skip(self))]
    // #[profiling::function]
    pub fn run(&mut self) -> SolverOutput {
        crate::debug!("Searching with min_precision: {}", self.min_precision);
        let mut current_game = self.game.create_subgame([]);
        let (mut w_even, mut w_odd) = (self.game.empty_vertex_set(), self.game.empty_vertex_set());

        while current_game.vertex_count() != 0 {
            let n_vertices = current_game.vertex_count();
            // We can ignore the original regions returned here by QLZ, as we will only ever receive dominions from
            // our top level call.
            let (_, _, dominion) = self.zielonka(&current_game, &current_game, n_vertices, n_vertices);
            // If we encounter a dominion deep in the search tree we destroy said tree and restart after removing it from the game.
            // The algorithm will still be quasi-polynomial, as it is at worst a linear multiplier to the complexity
            if let Some(dominion) = dominion {
                let owner = Owner::from_priority(dominion.dominating_p);
                let full_dominion = self.attract.attractor_tangle_rec(
                    &current_game,
                    owner,
                    Cow::Owned(dominion.vertices),
                    &mut self.tangles,
                    &mut self.strategy,
                );

                crate::debug!("Found full dominion of size: {}, restarting", full_dominion.count_ones(..));

                current_game.shrink_subgame(&full_dominion);
                self.tangles.intersect_tangles(&current_game);
                let (our_win, _) = us_and_them(owner, &mut w_even, &mut w_odd);
                our_win.union_with(&full_dominion);
                self.max_priority = 0;
            }
        }

        tracing::debug!("Discovered: {} tangles and {} dominions", self.tangles.tangles_found, self.tangles.dominions_found);
        SolverOutput::from_winning(self.game.vertex_count(), &w_odd)
    }

    /// Returns the winning regions `(W_even, W_odd)` as well as a flag indicating whether the results that were obtained
    /// used an early cut-off using the `precision_even/odd` parameters (`false` if so).
    fn zielonka<T: ParityGraph<u32> + OptimisedGraph<u32>>(&mut self, root_game: &SubGame<u32, T>, game: &SubGame<u32, T>, precision_even: usize, precision_odd: usize) -> (VertexSet, VertexSet, Option<Dominion>) {
        self.iterations += 1;
        if precision_odd == 0 {
            crate::debug!("End of precision; presumed won by player Even: {:?}", game.game_vertices.printable_vertices());
            return (game.game_vertices.clone(), game.empty_vertex_set(), None)
        }
        if precision_even == 0 {
            crate::debug!("End of precision; presumed won by player Odd: {:?}", game.game_vertices.printable_vertices());
            return (game.empty_vertex_set(), game.game_vertices.clone(), None)
        }
        if game.vertex_count() == 0 {
            return (game.empty_vertex_set(), game.empty_vertex_set(), None)
        }

        let d = game.priority_max();
        if self.max_priority < d {
            self.max_priority = d;
        }
        let region_owner = Owner::from_priority(d);

        let (new_p_even, new_p_odd) = match region_owner {
            Owner::Even => (precision_even, precision_odd / 2),
            Owner::Odd => (precision_even / 2, precision_odd),
        };
        // Half-precision call
        let (region_even, region_odd, dom) = self.zielonka(root_game, game, new_p_even, new_p_odd);
        if dom.is_some() {
            return (VertexSet::new(), VertexSet::new(), dom);
        }
        crate::info!(d, ?region_owner, precision_even, precision_odd, "In");

        // If the amount of vertices that remain are less than half our original precision then we know that no
        // winning region of size > half the vertices can exist, and we can safely stop here.
        let can_skip = match region_owner {
            Owner::Even => game.vertex_count() <= new_p_odd,
            Owner::Odd => game.vertex_count() <= new_p_even
        };
        if can_skip {
            crate::debug!(d, n = game.vertex_count(), new_p_odd, new_p_even, "Returning early, less than half remain");
            return (region_even, region_odd, None)
        }

        let (g_1, _) = us_and_them(region_owner, region_even, region_odd);
        let g_1 = SubGame::from_vertex_set(game.parent, g_1);
        // We need to use a compressed priority starting set, or else the assumption that even and odd alternate is violated
        // This would cause problems in the full precision call below.
        let starting_set = g_1.vertices_by_compressed_priority(d);
        let starting_set = AttractionComputer::make_starting_set(game, starting_set);
        
        crate::debug!("Starting vertices attractor: {:?}", starting_set.printable_vertices());
        // Reset the strategy for top vertices for better leak detection
        for v in starting_set.ones() {
            self.strategy[v] = VertexId::new(NO_STRATEGY as usize);
        }
        
        let mut g_1_attr = self.attract.attractor_tangle_rec(&g_1, region_owner, Cow::Borrowed(&starting_set), &mut self.tangles, &mut self.strategy);
        crate::debug!("H region: {} - {:?} - {:?}", d, g_1_attr.printable_vertices(), g_1.game_vertices.printable_vertices());

        // ** Try extract tangles **
        if !self.tangles.any_leaks_in_region(&g_1, d, &g_1_attr, &starting_set, &self.strategy) {
            // We know it's globally closed if this holds, no need to find tangles.
            if d == self.max_priority {
                let dom = Some(Dominion {
                    dominating_p: d,
                    vertices: g_1_attr,
                });
                return (VertexSet::new(), VertexSet::new(), dom);
            }

            let tangle_subgame = SubGame::from_vertex_set(game.parent, g_1_attr);
            let dom = self.tangles.extract_tangles(root_game, &tangle_subgame, &starting_set, d, &self.strategy);
            // If we find a dominion immediately destroy the call tree and return.
            // This essentially mimics regular tangle learning's behaviour.
            if dom.is_some() {
                return (VertexSet::new(), VertexSet::new(), dom);
            }
            // Put this back in its place
            g_1_attr = tangle_subgame.game_vertices;
        }

        let h_game = g_1.create_subgame_bit(&g_1_attr);

        // Full precision call
        crate::debug!("Full Precision: {d} - {:?}", h_game.game_vertices.printable_vertices());
        let (region_even, region_odd, dom) = self.zielonka(root_game, &h_game, precision_even, precision_odd);
        if dom.is_some() {
            return (VertexSet::new(), VertexSet::new(), dom);
        }
        
        let opponent = region_owner.other();
        let (opponent_dominion, our_region) = us_and_them(opponent, region_even, region_odd);
        let o_extended_dominion = self.attract.attractor_tangle_rec(&g_1, opponent, Cow::Borrowed(&opponent_dominion), &mut self.tangles, &mut self.strategy);

        let g_2 = g_1.create_subgame_bit(&o_extended_dominion);

        // Check if the opponent attracted from our winning region, in which case we need to recalculate
        // Otherwise we can skip the last half-precision call.
        if o_extended_dominion != opponent_dominion {
            // Remove the vertices which were attracted from our winning region, and expand the right side of our tree
            let (mut even_out, mut odd_out, dom) = self.zielonka(root_game, &g_2, new_p_even, new_p_odd);
            if dom.is_some() {
                return (VertexSet::new(), VertexSet::new(), dom);
            }
            let (opponent_result, _) = even_and_odd(opponent, &mut even_out, &mut odd_out);
            opponent_result.union_with(&o_extended_dominion);
            (even_out, odd_out, None)
        } else {
            let (e, o) = even_and_odd(region_owner, g_2.game_vertices, o_extended_dominion);
            (e, o, None)
        }
    }
}

/// Return (Ours, Theirs) parts, depending on the argument `us`.
fn us_and_them<T>(us: Owner, even: T, odd: T) -> (T, T) {
    match us {
        Owner::Even => {
            (even, odd)
        }
        Owner::Odd => {
            (odd, even)
        }
    }
}

/// Return (Even, Odd) parts, depending on the argument `us`.
fn even_and_odd<T>(us: Owner, ours: T, them: T) -> (T, T) {
    match us {
        Owner::Even => {
            (ours, them)
        }
        Owner::Odd => {
            (them, ours)
        }
    }
}

#[cfg(test)]
pub mod test {
    use crate::explicit::solvers::qpt_tangle_liverpool::LiverpoolSolver;
    use crate::{tests, tests::load_example, Owner};
    use std::time::Instant;
    use tracing_test::traced_test;

    #[test]
    // #[tracing_test::traced_test]
    pub fn verify_correctness() {
        for name in tests::examples_iter() {
            println!("Running test for: {name}...");
            let (game, compare) = tests::load_and_compare_example(&name);
            let mut qpt_zielonka = LiverpoolSolver::new(&game);
            let solution = qpt_zielonka.run();
            compare(solution);
            println!("{name} correct!")
        }
    }

    #[test]
    #[traced_test]
    pub fn test_solve_basic_paper_example() {
        let (game, compare) = tests::load_and_compare_example("basic_paper_example.pg");
        let mut solver = LiverpoolSolver::new(&game);

        let solution = solver.run();

        println!("Solution: {:#?}", solution);

        compare(solution);
    }

    #[test]
    // #[traced_test]
    pub fn test_solve_two_counters() {
        let (game, compare) = tests::load_and_compare_example("two_counters_2.pg");
        let mut solver = LiverpoolSolver::new(&game);

        let solution = solver.run();

        println!("Solution: {:#?}", solution);

        compare(solution);
    }

    #[test]
    pub fn test_solve_tue_example() {
        let game = load_example("tue_example.pg");
        let mut solver = LiverpoolSolver::new(&game);

        let solution = solver.run().winners;

        println!("Solution: {:#?}", solution);

        assert!(solution.iter().all(|win| *win == Owner::Odd));
    }

    #[test]
    #[traced_test]
    pub fn test_solve_action_converter() {
        let game = load_example("ActionConverter.tlsf.ehoa.pg");
        let mut solver = LiverpoolSolver::new(&game);

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
