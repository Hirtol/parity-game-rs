use crate::explicit::{BitsetExtensions, SubGame, VertexSet};
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

    #[tracing::instrument(name = "Run Zielonka", skip(self))]
    // #[profiling::function]
    pub fn run(&mut self) -> SolverOutput {
        crate::debug!("Searching with min_precision: {}", self.min_precision);
        let (even, odd) = self.zielonka(&mut self.game.create_subgame([]), self.game.vertex_count(), self.game.vertex_count());
        println!("Even Winning: {:?}", even.printable_vertices());
        println!("Odd Winning: {:?}", odd.printable_vertices());
        if even.count_ones(..) + odd.count_ones(..) < self.game.vertex_count() {
            panic!("Fewer vertices than expected were in the winning regions");
        }
        SolverOutput::from_winning(self.game.vertex_count(), &odd)
    }
    
    /// Returns the winning regions `(W_even, W_odd)` as well as a flag indicating whether the results that were obtained
    /// used an early cut-off using the `precision_even/odd` parameters (`false` if so).
    fn zielonka<T: ParityGraph<u32>>(&mut self, game: &mut SubGame<u32, T>, precision_even: usize, precision_odd: usize) -> (VertexSet, VertexSet) {
        self.iterations += 1;
        let (mut result_even, mut result_odd) = (VertexSet::empty_game(game), VertexSet::empty_game(game));
        if precision_odd == 0 {
            result_odd.union_with(&game.game_vertices);
            return (result_even, result_odd)
        }
        if precision_even == 0 {
            result_odd.union_with(&game.game_vertices);
            return (result_even, result_odd)
        }
        // If all the vertices are ignored
        if game.vertex_count() == 0 {
            return (result_even, result_odd)
        }
        
        let d = game.priority_max();
        let region_owner = Owner::from_priority(d);
        let (our_result, opponent_result) = match region_owner {
            Owner::Even => (&mut result_even, &mut result_odd),
            Owner::Odd => (&mut result_odd, &mut result_even)
        };

        let (new_p_even, new_p_odd) = match region_owner {
            Owner::Even => (precision_even, precision_odd / 2),
            Owner::Odd => (precision_even / 2, precision_odd),
        };
        // Half-precision call
        let (region_even, region_odd) = self.zielonka(game, new_p_even, new_p_odd);

        // If the amount of vertices that remain are less than half our original precision then we know that no
        // winning region of size > half the vertices can exist, and we can safely stop here.
        let can_skip = match region_owner {
            Owner::Even => game.vertex_count() <= new_p_odd,
            Owner::Odd => game.vertex_count() <= new_p_even
        };
        if can_skip {
            crate::debug!(d, n = game.vertex_count(), new_p_odd, new_p_even, "Returning early, less than half remain");
            return (result_even, result_odd)
        }
        
        let region = match region_owner {
            Owner::Even => {
                region_even
            }
            Owner::Odd => {
                region_odd
            }
        };
        let mut our_winning_region = SubGame::from_vertex_set(game.parent, region);
        let starting_set = our_winning_region.vertices_index_by_priority(d);
        let our_attractor = self.attract.attractor_set(&our_winning_region, region_owner, starting_set);
        
        // We ignore the half-precision regions after this, but we keep our larger attractor set as our winning region
        our_result.union_with(&our_attractor);
        
        let mut h = our_winning_region.create_subgame_bit(&our_attractor);
        
        // Full precision call
        crate::debug!("Full Precision: {d}");
        let (region_even, region_odd) = self.zielonka(&mut h, precision_even, precision_odd);
        let opponent = region_owner.other();
        let opponent_region = match opponent {
            Owner::Even => region_even,
            Owner::Odd => region_odd,
        };
        let opponent_attract = self.attract.attractor_set_bit(&h, opponent, Cow::Owned(opponent_region));
        opponent_result.union_with(&opponent_attract);
        
        // Check if the opponent attracted from our winning region, in which case we need to recalculate
        if !opponent_attract.is_disjoint(&our_winning_region.game_vertices) {
            // Remove the vertices which were attracted from our winning region, and expand the right side of our tree
            our_winning_region.shrink_subgame(&opponent_attract);
            our_result.difference_with(&opponent_attract);
            
            let (even_out, odd_out) = self.zielonka(&mut our_winning_region, new_p_even, new_p_odd);
            result_even.union_with(&even_out);
            result_odd.union_with(&odd_out);
            
            (result_even, result_odd)
        } else {
            (result_even, result_odd)
        }
    }
}

#[cfg(test)]
pub mod test {
    use crate::explicit::solvers::qpt_liverpool::LiverpoolSolver;
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
            let mut qpt_zielonka = LiverpoolSolver::new(&game);
            let solution = qpt_zielonka.run();
            compare(solution);
            println!("{name} correct!")
        }
    }

    #[test]
    pub fn test_solve_basic_paper_example() {
        let (game, compare) = tests::load_and_compare_example("basic_paper_example.pg");
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
