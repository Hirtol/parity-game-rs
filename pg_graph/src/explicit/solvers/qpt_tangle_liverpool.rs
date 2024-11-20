use crate::explicit::solvers::tangle_learning::{TangleManager, NO_STRATEGY};
use crate::explicit::{BitsetExtensions, SubGame, VertexId, VertexSet};
use crate::{explicit::{
    solvers::{AttractionComputer, SolverOutput},
    ParityGame, ParityGraph,
}, Owner};
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
        }
    }

    #[tracing::instrument(name = "Run Zielonka", skip(self))]
    // #[profiling::function]
    pub fn run(&mut self) -> SolverOutput {
        crate::debug!("Searching with min_precision: {}", self.min_precision);
        let (even, odd) = self.zielonka(&mut self.game.create_subgame([]), self.game.vertex_count(), self.game.vertex_count());
        // println!("Even Winning: {:?}", even.printable_vertices());
        // println!("Odd Winning: {:?}", odd.printable_vertices());
        // if even.count_ones(..) + odd.count_ones(..) < self.game.vertex_count() {
        //     panic!("Fewer vertices than expected were in the winning regions");
        // }
        tracing::debug!("Discovered: {} tangles", self.tangles.tangles_found);
        let odd = match Owner::from_priority(self.game.priority_max()) {
            Owner::Even => AttractionComputer::make_starting_set(self.game, even.zeroes().map(VertexId::new)),
            Owner::Odd => odd
        };

        SolverOutput::from_winning(self.game.vertex_count(), &odd)
    }

    /// Returns the winning regions `(W_even, W_odd)` as well as a flag indicating whether the results that were obtained
    /// used an early cut-off using the `precision_even/odd` parameters (`false` if so).
    fn zielonka<T: ParityGraph<u32>>(&mut self, game: &mut SubGame<u32, T>, precision_even: usize, precision_odd: usize) -> (VertexSet, VertexSet) {
        self.iterations += 1;
        let (mut result_even, mut result_odd) = (VertexSet::empty_game(game), VertexSet::empty_game(game));
        if precision_odd == 0 {
            result_even.union_with(&game.game_vertices);
            crate::debug!("End of precision; presumed won by player Even: {:?}", result_even.printable_vertices());
            return (result_even, result_odd)
        }
        if precision_even == 0 {
            result_odd.union_with(&game.game_vertices);
            crate::debug!("End of precision; presumed won by player Odd: {:?}", result_odd.printable_vertices());
            return (result_even, result_odd)
        }
        // If all the vertices are ignored
        if game.vertex_count() == 0 {
            return (result_even, result_odd)
        }

        let d = game.priority_max();
        let region_owner = Owner::from_priority(d);

        let (new_p_even, new_p_odd) = match region_owner {
            Owner::Even => (precision_even, precision_odd / 2),
            Owner::Odd => (precision_even / 2, precision_odd),
        };
        // Half-precision call
        let (region_even, region_odd) = self.zielonka(game, new_p_even, new_p_odd);
        crate::info!(d, ?region_owner, precision_even, precision_odd, "In");

        // If the amount of vertices that remain are less than half our original precision then we know that no
        // winning region of size > half the vertices can exist, and we can safely stop here.
        let can_skip = match region_owner {
            Owner::Even => game.vertex_count() <= new_p_odd,
            Owner::Odd => game.vertex_count() <= new_p_even
        };
        if can_skip {
            crate::debug!(d, n = game.vertex_count(), new_p_odd, new_p_even, "Returning early, less than half remain");
            return (region_even, region_odd)
        }

        let (g_1, _) = us_and_them(region_owner, region_even, region_odd);
        let g_1 = SubGame::from_vertex_set(game.parent, g_1);
        let starting_set = g_1.vertices_by_compressed_priority(d);
        let starting_set = AttractionComputer::make_starting_set(game, starting_set);
        
        crate::debug!("Starting vertices attractor: {:?}", starting_set.printable_vertices());
        // Reset the strategy for top vertices for better leak detection
        // let mut strat = vec![VertexId::new(NO_STRATEGY as usize); game.original_vertex_count()];
        for v in starting_set.ones() {
            self.strategy[v] = VertexId::new(NO_STRATEGY as usize);
        }
        
        let g_1_attr = self.attract.attractor_set_tangle(&g_1, region_owner, Cow::Borrowed(&starting_set), &self.tangles, &mut self.strategy);
        crate::debug!("H region: {} - {:?} - {:?}", d, g_1_attr.printable_vertices(), g_1.game_vertices.printable_vertices());

        // ** Try extract tangles **
        if !self.tangles.any_leaks_in_region(&g_1, d, &g_1_attr, &starting_set, &self.strategy) {
            // tracing::debug!("No leaks, checking for tangles: {d}");
            let _ = self.tangles.extract_tangles(game.parent, &g_1, d, &mut self.strategy);
        }

        let mut h_game = g_1.create_subgame_bit(&g_1_attr);

        // Full precision call
        crate::debug!("Full Precision: {d} - {:?}", h_game.game_vertices.printable_vertices());

        // PROBLEM: The below assumes this is won by opponent, but what if we have two consecutive priorities aligned with us!
        // solution: Use compressed priority attractors
        let (region_even, region_odd) = self.zielonka(&mut h_game, precision_even, precision_odd);
        let opponent = region_owner.other();
        let (opponent_dominion, our_region) = us_and_them(opponent, region_even, region_odd);
        let o_extended_dominion = self.attract.attractor_set_tangle(&g_1, opponent, Cow::Borrowed(&opponent_dominion), &self.tangles, &mut self.strategy);

        let mut g_2 = g_1.create_subgame_bit(&o_extended_dominion);

        // Check if the opponent attracted from our winning region, in which case we need to recalculate
        // Otherwise we can skip the last half-precision call.
        if o_extended_dominion != opponent_dominion {
            // Remove the vertices which were attracted from our winning region, and expand the right side of our tree
            let (opponent_result, _) = even_and_odd(opponent, &mut result_even, &mut result_odd);
            opponent_result.union_with(&o_extended_dominion);
            let (even_out, odd_out) = self.zielonka(&mut g_2, new_p_even, new_p_odd);
            result_even.union_with(&even_out);
            result_odd.union_with(&odd_out);
            self.tangles.merge_tangles();
            (result_even, result_odd)
        } else {
            self.tangles.merge_tangles();
            even_and_odd(region_owner, g_2.game_vertices, o_extended_dominion)
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
