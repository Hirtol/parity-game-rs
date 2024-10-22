use std::borrow::Cow;

use crate::explicit::solvers::{AttractionComputer, Dominion};
use crate::explicit::VertexSet;
use crate::{datatypes::Priority, explicit::{solvers::SolverOutput, ParityGame, ParityGraph, SubGame, VertexId}, Owner, VertexVec};

const NO_STRATEGY: u32 = u32::MAX;

pub struct TangleSolver<'a, Vertex> {
    game: &'a ParityGame<u32, Vertex>,
    attract: AttractionComputer<u32>,
}

impl<'a, Vertex> TangleSolver<'a, Vertex> {
    pub fn new(game: &'a ParityGame<u32, Vertex>) -> Self {
        TangleSolver {
            game,
            attract: AttractionComputer::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tangle {
    pub owner: Owner,
    pub priority: Priority,
    pub vertices: VertexSet,
    pub escapes: Vec<VertexId>,
}

impl<'a> TangleSolver<'a, VertexVec> {
    #[tracing::instrument(name = "Run Tangle Learning", skip(self))]
    pub fn run(&mut self) -> SolverOutput {
        let (even, odd) = self.tangle_solver(self.game);
        SolverOutput::from_winning(self.game.original_vertex_count(), &odd)
    }

    /// Return (W_even, W_odd)
    fn tangle_solver<T: ParityGraph<u32>>(&mut self, game: &T) -> (VertexSet, VertexSet) {
        let (mut winning_even, mut winning_odd) = (
            VertexSet::with_capacity(game.vertex_count()),
            VertexSet::with_capacity(game.vertex_count()),
        );

        let mut current_game = game.create_subgame([]);
        let mut tangles = Vec::new();
        let mut strategy = vec![VertexId::new(NO_STRATEGY as usize); game.original_vertex_count()];

        while current_game.vertex_count() > 0 {
            let (new_tangles, dominion) = self.search_dominion(&current_game, tangles, strategy.as_mut_slice());
            tangles = new_tangles;

            let full_dominion = self.attract.attractor_set_bit(&current_game, Owner::from_priority(dominion.dominating_p), Cow::Borrowed(&dominion.vertices));

            tracing::debug!(
                priority = dominion.dominating_p,
                size = full_dominion.count_ones(..),
                "Found dominion"
            );

            match Owner::from_priority(dominion.dominating_p) {
                Owner::Even => winning_even.union_with(&full_dominion),
                Owner::Odd => winning_odd.union_with(&full_dominion),
            }

            current_game.shrink_subgame(&full_dominion);
            tangles.retain(|tangle| tangle.vertices.is_subset(&current_game.game_vertices));
        }

        (winning_even, winning_odd)
    }

    fn search_dominion(
        &mut self,
        current_game: &impl ParityGraph<u32>,
        mut tangles: Vec<Tangle>,
        strategy: &mut [VertexId<u32>]
    ) -> (Vec<Tangle>, Dominion) {
        loop {
            let mut partial_subgame = current_game.create_subgame([]);
            let mut temp_tangles = Vec::new();

            while partial_subgame.vertex_count() != 0 {
                let d = partial_subgame.priority_max();
                let owner = Owner::from_priority(d);

                let starting_set = AttractionComputer::make_starting_set(&partial_subgame, partial_subgame.vertices_index_by_priority(d));
                let filtered_tangles = tangles.iter().filter(|t| t.owner == owner);
                let tangle_attractor = self.attract.attractor_set_tangle(&partial_subgame, owner, Cow::Owned(starting_set), filtered_tangles, strategy);
                // println!("Strat: {:?}", strategy);
                // panic!();
                let mut tangle_subgame = partial_subgame.clone();
                tangle_subgame.intersect_subgame(&tangle_attractor);
                let new_tangles = self.extract_tangles(&partial_subgame, &tangle_subgame, strategy);

                if let Some(tangle_dominion) = new_tangles.iter().find(|t| t.escapes.is_empty()) {
                    let real_dom = Dominion {
                        dominating_p: d,
                        vertices: tangle_dominion.vertices.clone(),
                    };
                    tangles.extend(new_tangles);
                    tangles.extend(temp_tangles);

                    return (tangles, real_dom)
                } else {
                    temp_tangles.extend(new_tangles);
                    partial_subgame.shrink_subgame(&tangle_attractor);
                }
            }

            tangles.extend(temp_tangles);
        }
    }

    fn extract_tangles<T: ParityGraph<u32>>(&mut self, current_sub_game: &T, tangle_attractor: &SubGame<u32, T::Parent>, strategy: &mut [VertexId<u32>]) -> Vec<Tangle> {
        Vec::new()
    }
}


#[cfg(test)]
pub mod test {
    use crate::explicit::solvers::tangle_learning::TangleSolver;
    use crate::{load_example, tests, Owner};
    use std::time::Instant;

    #[test]
    pub fn verify_correctness() {
        for name in tests::examples_iter() {
            println!("Running test for: {name}...");
            let (game, compare) = tests::load_and_compare_example(&name);
            let mut tl_solver = TangleSolver::new(&game);
            let solution = tl_solver.run();
            compare(solution);
            println!("{name} correct!")
        }
    }

    #[test]
    pub fn test_solve_action_converter() {
        let game = load_example("ActionConverter.tlsf.ehoa.pg");
        let mut solver = TangleSolver::new(&game);

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
