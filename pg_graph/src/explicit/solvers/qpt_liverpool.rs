use crate::explicit::{BitsetExtensions, SubGame, VertexId, VertexSet};
use crate::{explicit::{
    solvers::{AttractionComputer, SolverOutput},
    ParityGame, ParityGraph,
}, Owner, Priority};
use std::borrow::Cow;

pub struct LiverpoolSolver<'a> {
    pub iterations: usize,
    game: &'a ParityGame<u32>,
    attract: AttractionComputer<u32>,
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
            min_precision: if game.graph_edges().any(|(source, target)| source == target) { 0 } else { 1 },
        }
    }

    #[tracing::instrument(name = "Run Liverpool Zielonka", skip(self))]
    // #[profiling::function]
    pub fn run(&mut self) -> SolverOutput {
        crate::debug!("Searching with min_precision: {}", self.min_precision);
        let (_even, odd) = self.zielonka(
            &self.game.create_subgame([]),
            self.game.vertex_count(),
            self.game.vertex_count()
        );

        SolverOutput::from_winning(self.game.vertex_count(), &odd)
    }

    /// Returns the winning regions `(W_even, W_odd)` as well as a flag indicating whether the results that were obtained
    /// used an early cut-off using the `precision_even/odd` parameters (`false` if so).
    fn zielonka<T: ParityGraph<u32>>(&mut self, game: &SubGame<u32, T>, precision_even: usize, precision_odd: usize) -> (VertexSet, VertexSet) {
        self.iterations += 1;
        if precision_odd == 0 {
            crate::debug!("End of precision; presumed won by player Even: {:?}", game.game_vertices.printable_vertices());
            return (game.game_vertices.clone(), VertexSet::new())
        }
        if precision_even == 0 {
            crate::debug!("End of precision; presumed won by player Odd: {:?}", game.game_vertices.printable_vertices());
            return (VertexSet::new(), game.game_vertices.clone())
        }
        if game.vertex_count() == 0 {
            return (VertexSet::new(), VertexSet::new())
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

        let (g_1, opp) = us_and_them(region_owner, region_even, region_odd);
        let g_1 = SubGame::from_vertex_set(game.parent, g_1);
        // If 'our' region is empty, then we know there are no dominions up to precision_ours, and can presume it won by the opponent.
        if g_1.vertex_count() == 0 {
            return even_and_odd(region_owner, g_1.game_vertices, opp)
        }
        
        // We need to use a compressed priority starting set, or else the assumption that even and odd alternate is violated
        // This would cause problems in the full precision call below.
        let starting_set = g_1.vertices_by_compressed_priority(d);
        let starting_set = AttractionComputer::make_starting_set(game, starting_set);

        crate::debug!("Starting vertices attractor: {:?}", starting_set.printable_vertices());

        let g_1_attr = self.attract.attractor_set_bit(&g_1, region_owner, Cow::Borrowed(&starting_set));
        crate::debug!("H region: {} - {:?} - {:?}", d, g_1_attr.printable_vertices(), g_1.game_vertices.printable_vertices());

        let h_game = g_1.create_subgame_bit(&g_1_attr);

        // Full precision call
        crate::debug!("Full Precision: {d} - {:?}", h_game.game_vertices.printable_vertices());
        let (region_even, region_odd) = self.zielonka(&h_game, precision_even, precision_odd);
        crate::info!(d, ?region_owner, precision_even, precision_odd, "In 2");

        let opponent = region_owner.other();
        let (opponent_dominion, _) = us_and_them(opponent, region_even, region_odd);
        let mut o_extended_dominion = self.attract.attractor_set_bit(&g_1, opponent, Cow::Borrowed(&opponent_dominion));

        let g_2 = g_1.create_subgame_bit(&o_extended_dominion);

        // Check if the opponent attracted from our winning region, in which case we need to recalculate
        // Otherwise we can skip the last half-precision call.
        // We can also skip if the remaining game to be explored is empty.
        if o_extended_dominion != opponent_dominion && g_2.vertex_count() != 0 {
            // Remove the vertices which were attracted from our winning region, and expand the right side of our tree
            let (mut even_out, mut odd_out) = self.zielonka(&g_2, new_p_even, new_p_odd);
            // Combine with the opponent's dominions which we found in the first and second recursive calls
            let (opponent_result, _) = us_and_them(opponent, &mut even_out, &mut odd_out);
            opponent_result.union_with(&o_extended_dominion);
            opponent_result.union_with(&opp);
            
            (even_out, odd_out)
        } else {
            // Combine with the opponent's dominions which we found in the first and second recursive calls
            o_extended_dominion.union_with(&opp);
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
    use crate::explicit::solvers::qpt_liverpool::LiverpoolSolver;
    use crate::{tests, tests::load_example, Owner};
    use std::time::Instant;
    use tracing_test::traced_test;

    #[test]
    // #[tracing_test::traced_test]
    pub fn verify_correctness() {
        for name in tests::examples_iter() {
            // if name.contains("two_counters_14p") {
            //     continue;
            // }
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
