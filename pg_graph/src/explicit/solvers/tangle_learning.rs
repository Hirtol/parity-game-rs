use std::collections::VecDeque;

use ahash::HashMap;
use itertools::Itertools;

use crate::{explicit::solvers::SolverOutput, Owner, ParityGame, ParityGraph, Priority, SubGame, VertexId};

pub struct TangleSolver<'a> {
    pub recursive_calls: usize,
    game: &'a ParityGame,
    attractor: AttractionComputer,
}

struct Tangle {
    owner: Owner,
    priority: Priority,
    vertices: Vec<VertexId>,
    strategy: HashMap<VertexId, VertexId>,
}

impl Tangle {
    /// Calculate the set of nodes which are outside the tangle, but reachable from within by the opposite player of the
    /// tangle's owner.
    ///
    /// TODO: The game might be better as the full set of vertices of the parent game (not sure yet)
    pub fn outgoing_game_edges<'a, T: 'a + ParityGraph>(&'a self, game: &'a T) -> impl Iterator<Item = VertexId> + 'a {
        self.vertices
            .iter()
            .filter(|v| game.get(**v).unwrap().owner != self.owner)
            .flat_map(|u| game.edges(*u).filter(|v| !self.vertices.contains(v)))
    }
}

fn intersect_tangle_game<'a>(
    tangles: &'a [Tangle],
    game: &'a SubGame<'a, u32, ParityGame>,
) -> impl Iterator<Item = &'a Tangle> + 'a {
    tangles
        .iter()
        .filter(|t| t.vertices.iter().all(|v| game.get(*v).is_some()))
}

impl<'a> TangleSolver<'a> {
    pub fn new(game: &'a ParityGame) -> Self {
        TangleSolver {
            game,
            recursive_calls: 0,
            attractor: AttractionComputer::new(),
        }
    }

    #[tracing::instrument(name = "Run Tangle Learning", skip(self))]
    pub fn run(&mut self) -> SolverOutput {
        // let (mut win_even, mut win_odd) = (Vec::new(), Vec::new());
        // let mut tangles = Vec::new();
        // let mut strategy = vec![None; self.game.vertex_count()];

        // let (even, odd) = self.zielonka(ahash::HashSet::default());
        // let mut winners = vec![Owner::Even; self.game.vertex_count()];
        // for idx in odd {
        //     winners[idx.index()] = Owner::Odd;
        // }
        //
        // SolverOutput {
        //     winners,
        //     strategy: None,
        // }
        todo!()
    }

    fn search(
        &mut self,
        game: &SubGame<u32, ParityGame>,
        mut tangles: Vec<Tangle>,
        strategy: &mut Vec<Option<VertexId>>,
    ) -> (Vec<Tangle>, Tangle) {
        // loop {
        //     let mut current_sub = game.create_subgame([]);
        //     let mut region_func = HashMap::default();
        //     let mut new_tangles = Vec::new();
        //
        //     while current_sub.vertex_count() != 0 {
        //         current_sub = game.create_subgame(region_func.keys());
        //         let new_tangles = intersect_tangle_game(&tangles, &current_sub);
        //
        //         let d = current_sub.priority_max();
        //         let owner = Owner::from_priority(d);
        //         let starting_set = current_sub.vertices_by_priority_idx(d).map(|(idx, _)| idx);
        //     }
        // }
        // petgraph::algo::tarjan_scc()

        todo!()
    }

    fn extract_tangles(
        &mut self,
        region_owner: Owner,
        region: Vec<VertexId>,
        game: &SubGame<u32, ParityGame>,
    ) -> Vec<Tangle> {
        //TODO:
        // region.iter().filter(|v| game.get_u(**v));

        vec![]
    }
}

pub struct AttractionComputer {
    queue: VecDeque<VertexId>,
}

impl AttractionComputer {
    pub fn new() -> Self {
        Self {
            queue: Default::default(),
        }
    }

    /// Calculate the attraction set for the given starting set.
    ///
    /// This resulting set will contain all vertices which:
    /// * If a vertex is owned by `player`, then if any edge leads to the attraction set it will be added to the resulting set.
    /// * If a vertex is _not_ owned by `player`, then only if _all_ edges lead to the attraction set will it be added.
    pub fn attractor_set<T: ParityGraph>(
        &mut self,
        game: &T,
        player: Owner,
        starting_set: impl IntoIterator<Item = VertexId>,
        strategy: &mut [Option<VertexId>],
    ) -> ahash::HashSet<VertexId> {
        let mut attract_set = ahash::HashSet::from_iter(starting_set);
        self.queue.extend(&attract_set);

        while let Some(next_item) = self.queue.pop_back() {
            for predecessor in game.predecessors(next_item) {
                let vertex = game.get(predecessor).expect("Invalid predecessor");
                let should_add = if vertex.owner == player {
                    // *any* edge needs to lead to the attraction set, since this is a predecessor of an item already in the attraction set we know that already!
                    true
                } else {
                    game.edges(predecessor).all(|v| attract_set.contains(&v))
                };

                // Only add to the attraction set if we should
                if should_add && attract_set.insert(predecessor) {
                    // Update the strategy to point towards the attraction set.
                    strategy[predecessor.index()] = Some(next_item);
                    self.queue.push_back(predecessor);
                }
            }
        }

        attract_set
    }

    pub fn tangle_attractor_set<T: ParityGraph>(
        &mut self,
        game: &T,
        player: Owner,
        starting_set: impl IntoIterator<Item = VertexId>,
        tangles: &[Tangle],
        strategy: &mut [Option<VertexId>],
    ) -> ahash::HashSet<VertexId> {
        let mut attract_set = ahash::HashSet::from_iter(starting_set);
        self.queue.extend(&attract_set);

        let relevant_tangles = tangles.iter().filter(|t| t.owner == player).collect_vec();
        let outgoing_game_edges = relevant_tangles
            .iter()
            .map(|t| t.outgoing_game_edges(game).collect_vec())
            .filter(|v| !v.is_empty())
            .collect_vec();

        while let Some(next_item) = self.queue.pop_back() {
            for predecessor in game.predecessors(next_item) {
                let vertex = game.get(predecessor).expect("Invalid predecessor");
                let should_add = if vertex.owner == player {
                    // *any* edge needs to lead to the attraction set, since this is a predecessor of an item already in the attraction set we know that already!
                    true
                } else {
                    game.edges(predecessor).all(|v| attract_set.contains(&v))
                };

                // Only add to the attraction set if we should
                if should_add && attract_set.insert(predecessor) {
                    // Update the strategy to point towards the attraction set.
                    strategy[predecessor.index()] = Some(next_item);
                    self.queue.push_back(predecessor);

                    // *** Tangle stuff ***

                    // Can be more efficient, only check for tangles who have this new vertex.
                    let tangles_to_attract = outgoing_game_edges
                        .iter()
                        .zip(&relevant_tangles)
                        .filter(|(t_edges, t)| t_edges.iter().all(|v| attract_set.contains(v)))
                        .collect_vec();

                    // Attract all tangles and their vertices.
                    for (_, tangle) in tangles_to_attract {
                        for v in &tangle.vertices {
                            // TODO: May need to not use the `if` here, and instead add to queue unconditionally.
                            if attract_set.insert(*v) {
                                self.queue.push_back(*v);

                                // Update strategy for this vertex if it wasn't yet in the strategy
                                if game.get(*v).unwrap().owner == player {
                                    if strategy[v.index()] == None {
                                        strategy[v.index()] = tangle.strategy.get(v).copied();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        attract_set
    }
}

#[cfg(test)]
pub mod test {
    use std::time::Instant;

    use pg_parser::parse_pg;

    use crate::{explicit::solvers::zielonka::ZielonkaSolver, Owner, ParityGame, tests::example_dir};

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
