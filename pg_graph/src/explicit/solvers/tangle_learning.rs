use crate::explicit::BitsetExtensions;
use crate::{
    datatypes::Priority,
    explicit::{
        solvers::{AttractionComputer, Dominion, SolverOutput},
        ParityGame, ParityGraph, SubGame, VertexId, VertexSet,
    },
    Owner, VertexVec,
};
use itertools::Itertools;
use std::collections::VecDeque;
use std::{
    borrow::Cow,
    num::{NonZeroU32, NonZeroUsize},
};

pub const NO_STRATEGY: u32 = u32::MAX;

pub struct TangleSolver<'a, Vertex> {
    game: &'a ParityGame<u32, Vertex>,
    attract: AttractionComputer<u32>,
    pearce: PearceTangleScc,
    tangles_found: u32,
    dominions_found: u32,
}

impl<'a, Vertex> TangleSolver<'a, Vertex> {
    pub fn new(game: &'a ParityGame<u32, Vertex>) -> Self {
        TangleSolver {
            pearce: PearceTangleScc::new(game.inverted_vertices.len()),
            game,
            attract: AttractionComputer::new(),
            tangles_found: 0,
            dominions_found: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tangle {
    pub id: u32,
    pub owner: Owner,
    pub priority: Priority,
    pub vertices: VertexSet,
    /// All vertices _to which_ this tangle can escape.
    ///
    /// They will therefor not be in `vertices`.
    pub escape_targets: Vec<VertexId>,
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
            let new_tangles = self.search_dominion(&current_game, tangles, strategy.as_mut_slice());
            tangles = new_tangles;
            
            let (full_dominion, priority) = tangles.iter().filter(|t| t.escape_targets.is_empty()).fold((VertexSet::empty_game(&current_game), 0), |mut acc, tangle| {
                acc.0.union_with(&tangle.vertices);
                (acc.0, tangle.priority)
            });
            let owner = Owner::from_priority(priority);
            
            self.dominions_found += 1;
            let filtered_tangles = tangles.iter().filter(|t| t.owner == owner);
            let full_dominion = self.attract.attractor_set_tangle(
                &current_game,
                Owner::from_priority(priority),
                Cow::Borrowed(&full_dominion),
                filtered_tangles,
                strategy.as_mut_slice()
            );

            tracing::debug!(
                priority = priority,
                size = full_dominion.count_ones(..),
                "Found dominion"
            );

            match owner {
                Owner::Even => winning_even.union_with(&full_dominion),
                Owner::Odd => winning_odd.union_with(&full_dominion),
            }

            current_game.shrink_subgame(&full_dominion);
            
            
            // for dominion in tangles.iter().filter(|t| t.escape_targets.is_empty()) {
            //     self.dominions_found += 1;
            //     let full_dominion = self.attract.attractor_set_tangle(
            //         &current_game,
            //         Owner::from_priority(dominion.priority),
            //         Cow::Borrowed(&dominion.vertices),
            //         tangles.iter(),
            //         strategy.as_mut_slice()
            //     );
            // 
            //     tracing::debug!(
            //         priority = dominion.priority,
            //         size = full_dominion.count_ones(..),
            //         "Found dominion"
            //     );
            // 
            //     match Owner::from_priority(dominion.priority) {
            //         Owner::Even => winning_even.union_with(&full_dominion),
            //         Owner::Odd => winning_odd.union_with(&full_dominion),
            //     }
            // 
            //     current_game.shrink_subgame(&full_dominion);
            // }
            
            tangles.retain(|tangle| tangle.vertices.is_subset(&current_game.game_vertices));
        }

        tracing::info!(tangles_found=?self.tangles_found, "Finished Tangle Learning");

        (winning_even, winning_odd)
    }

    fn search_dominion(
        &mut self,
        current_game: &impl ParityGraph<u32>,
        mut tangles: Vec<Tangle>,
        strategy: &mut [VertexId<u32>],
    ) -> Vec<Tangle> {
        loop {
            let mut partial_subgame = current_game.create_subgame([]);
            let mut temp_tangles = Vec::new();
            strategy.fill(VertexId::new(NO_STRATEGY as usize));

            while partial_subgame.vertex_count() != 0 {
                let d = partial_subgame.priority_max();
                let owner = Owner::from_priority(d);

                let starting_set = AttractionComputer::make_starting_set(
                    &partial_subgame,
                    partial_subgame.vertices_index_by_priority(d),
                );
                let filtered_tangles = tangles.iter().filter(|t| t.owner == owner);
                let tangle_attractor = self.attract.attractor_set_tangle(
                    &partial_subgame,
                    owner,
                    Cow::Owned(starting_set),
                    filtered_tangles,
                    strategy,
                );
                tracing::debug!(attr=?tangle_attractor.ones().collect_vec(), "Tangle Attractor");
                let mut greatest_vertices = tangle_attractor.ones_vertices().filter(|v| current_game.priority(*v) == d);
                // Check if there are any escapes from this region, or if it's locally closed.
                let leaks = greatest_vertices.any(|v| {
                    if current_game.owner(v) == owner {
                        strategy[v.index()].index() == NO_STRATEGY as usize
                    } else {
                        partial_subgame
                            .edges(v)
                            .any(|succ| !tangle_attractor.contains(succ.index()))
                    }
                });
                tracing::debug!(?leaks, "Leaks");

                // If there are any leaks from our region then there's no point in finding tangles
                if leaks {
                    partial_subgame.shrink_subgame(&tangle_attractor);
                    continue;
                }

                let mut tangle_subgame = partial_subgame.clone();
                tangle_subgame.intersect_subgame(&tangle_attractor);
                let (new_tangles, contains_dominion) = self.extract_tangles(current_game, &tangle_attractor, &tangle_subgame, d, strategy);

                temp_tangles.extend(new_tangles);
                if contains_dominion {
                    tangles.extend(temp_tangles);
                    return tangles;
                } else {
                    partial_subgame.shrink_subgame(&tangle_attractor);
                }
            }
            tracing::info!("Finished iteration, next");
            tangles.extend(temp_tangles);
        }
    }

    fn extract_tangles<T: ParityGraph<u32>>(
        &mut self,
        current_game: &T,
        tangle_attractor: &VertexSet,
        tangle_sub_game: &SubGame<u32, T::Parent>,
        region_priority: Priority,
        strategy: &mut [VertexId<u32>],
    ) -> (Vec<Tangle>, bool) {
        let top_vertices = tangle_attractor.ones_vertices().filter(|v| current_game.priority(*v) == region_priority);
        let mut tangles = Vec::new();
        let region_owner = Owner::from_priority(region_priority);
        let mut has_dominion = false;

        self.pearce.run(tangle_sub_game, region_owner, top_vertices, strategy, |tangle, top_vertex| {
            // Ensure non-trivial SCC (more than one vertex OR self-loop
            let tangle_size = tangle.len();
            let id = self.tangles_found;
            let is_tangle = tangle_size != 1 || strategy[top_vertex.index()] == top_vertex || current_game.has_edge(top_vertex, top_vertex);
            if !is_tangle {
                let tangle_data = tangle.collect_vec();
                tracing::warn!(?tangle_data, "Skipping trivial tangle");
                return;
            }
            
            let full_tangle = tangle.collect_vec();
            tracing::info!(id, ?tangle_size, ?region_owner, ?region_priority, ?full_tangle, "Found new tangle");
            
            // Count all escapes that remain in the GREATER unsolved game
            let mut final_tangle = current_game.empty_vertex_set();
            for v in full_tangle {
                final_tangle.insert(v.index());
            }
            let mut target_escapes = Vec::new();

            for v in final_tangle.ones_vertices() {
                // Don't care about alpha-vertices, as they are guaranteed not to be leaks due to our earlier checks,
                // and can simply choose to remain in the tangle (else it wouldn't be an SCC)
                if current_game.owner(v) != region_owner {
                    for target in current_game.edges(v) {
                        // Only mark as escapes those vertices which actually _are_ escapes.
                        // TODO: Make this more efficient.
                        if !final_tangle.contains(target.index()) && !target_escapes.contains(&target) {
                            target_escapes.push(target);
                        }
                    }
                }
            }
            // For faster compute later on
            if target_escapes.is_empty() {
                tracing::info!("Tangle was a dominion");
                self.dominions_found += 1;
                has_dominion = true;
            } else {
                tracing::info!(escapes=?target_escapes.len(), "Tangle has escapes");
                self.tangles_found += 1;
            }

            let new_tangle = Tangle {
                id,
                owner: region_owner,
                priority: region_priority,
                vertices: final_tangle,
                escape_targets: target_escapes,
            };

            tangles.push(new_tangle);
        });

        (tangles, has_dominion)
    }
}

#[derive(Debug, Clone)]
pub struct PearceTangleScc {
    index: u32,
    component_count: u32,
    v_s: VecDeque<VertexId<u32>>,
    i_s: VecDeque<u32>,
    root: VertexSet,
    root_index: Vec<Option<NonZeroU32>>,
}

type PearceSccIter<'a> = std::iter::Skip<std::collections::vec_deque::Iter<'a, VertexId<u32>>>;

impl PearceTangleScc {
    pub fn new(max_size: usize) -> Self {
        Self {
            index: 1,                  // Invariant: index < component_count at all times.
            component_count: u32::MAX, // Will hold if component_count is initialized to number of nodes - 1 or higher.
            v_s: Default::default(),
            i_s: Default::default(),
            root: VertexSet::with_capacity(max_size),
            root_index: vec![None; max_size],
        }
    }

    #[profiling::function]
    pub fn run(&mut self, graph: &impl ParityGraph<u32>, region_owner: Owner, top_level_vs: impl Iterator<Item=VertexId<u32>>, strategy: &[VertexId<u32>], mut scc_found: impl FnMut(PearceSccIter<'_>, VertexId<u32>)) {
        self.root_index.clear();
        self.root.clear();
        self.root_index.resize(graph.original_vertex_count(), None);

        for n in top_level_vs {
            let visited = self.root_index[n.index()].is_some();
            if !visited {
                self.visit(n, region_owner, graph, strategy, &mut scc_found);
            }
        }
    }

    #[profiling::function]
    #[inline]
    fn visit(
        &mut self,
        v_id: VertexId<u32>,
        region_owner: Owner,
        graph: &impl ParityGraph<u32>,
        strategy: &[VertexId<u32>],
        mut scc_found: impl FnMut(PearceSccIter<'_>, VertexId<u32>),
    ) {
        self.begin_visit(v_id);

        while let Some(&to_explore) = self.v_s.front() {
            self.visit_loop(to_explore, region_owner, graph, strategy, &mut scc_found);
        }
    }

    #[profiling::function]
    #[inline]
    fn begin_visit(&mut self, v_id: VertexId<u32>) {
        self.v_s.push_front(v_id);
        self.i_s.push_front(0);
        self.root.insert(v_id.index());
        self.root_index[v_id.index()] = NonZeroU32::new(self.index);
        self.index += 1;
    }

    #[inline]
    fn visit_loop(
        &mut self,
        v_id: VertexId<u32>,
        region_owner: Owner,
        graph: &impl ParityGraph<u32>,
        strategy: &[VertexId<u32>],
        scc_found: impl FnMut(PearceSccIter<'_>, VertexId<u32>),
    ) {
        // Restrict graph to the strategy
        if graph.owner(v_id) == region_owner {
            let edge_i = *self.i_s.front().expect("Impossible Invariant Violation");
            let edge = strategy[v_id.index()];
            
            // We have only one edge, namely the strategy, so we can simply elide the loop.
            if edge_i == 0 && self.root_index[edge.index()].is_none() {
                // Add the current index back in order to `finish_edge` after we finish exploring this new node.
                *self.i_s.front_mut().expect("impossible") = edge_i + 1;
                self.begin_visit(edge);
                return;
            }

            // Finish edge
            if self.root_index(edge) < self.root_index(v_id) {
                self.root_index[v_id.index()] = self.root_index[edge.index()];
                self.root.remove(v_id.index());
            }
        } else {
            let edge_i = *self.i_s.front().expect("Impossible Invariant Violation");

            for (i, edge) in graph.edges(v_id).enumerate().skip(edge_i as usize) {
                // First check if `i` was in range
                // New node to explore, restart visit loop, begin edge
                if self.root_index[edge.index()].is_none() {
                    // Add the current index back in order to `finish_edge` after we finish exploring this new node.
                    *self.i_s.front_mut().expect("impossible") = i as u32;
                    self.begin_visit(edge);
                    return;
                }

                // Finish edge
                if self.root_index(edge) < self.root_index(v_id) {
                    self.root_index[v_id.index()] = self.root_index[edge.index()];
                    self.root.remove(v_id.index());
                }
            }
        }

        self.finish_visit(v_id, scc_found);
    }

    #[profiling::function]
    #[inline]
    fn finish_visit(&mut self, v_id: VertexId<u32>, mut scc_found: impl FnMut(PearceSccIter<'_>, VertexId<u32>)) {
        // Take the current vertex of the stack
        self.v_s.pop_front();
        self.i_s.pop_front();
        // Update a new component
        if self.root.contains(v_id.index()) {
            let c = NonZeroU32::new(self.component_count);
            let mut index_adjust = 1;
            let root_indexes = &mut self.root_index;
            let start = self
                .v_s
                .iter()
                .rposition(|&part_of_scc| {
                    if root_indexes[v_id.index()] > root_indexes[part_of_scc.index()] {
                        true
                    } else {
                        root_indexes[part_of_scc.index()] = c;
                        index_adjust += 1;
                        false
                    }
                })
                .map(|i| i + 1)
                .unwrap_or_default();
            // Ugly way of ensuring this root is in the collection
            self.v_s.push_back(v_id);
            let component_items = self.v_s.iter().skip(start);
            scc_found(component_items, v_id);

            self.v_s.truncate(start);
            self.index -= index_adjust;
            self.root_index[v_id.index()] = c;
            self.component_count -= 1;
        } else {
            self.v_s.push_back(v_id);
        }
    }

    #[inline]
    fn root_index(&self, v_id: VertexId<u32>) -> Option<NonZeroU32> {
        self.root_index[v_id.index()]
    }
}

#[cfg(test)]
pub mod test {
    use crate::{explicit::solvers::tangle_learning::TangleSolver, load_example, tests, Owner};
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
    #[tracing_test::traced_test]
    pub fn test_solve_action_converter() {
        // let game = load_example("ActionConverter.tlsf.ehoa.pg");
        let game = load_example("ltl2dpa13.tlsf.ehoa.pg");
        let mut solver = TangleSolver::new(&game);

        let now = Instant::now();
        let solution = solver.run().winners;

        let (even_wins, odd_wins) = solution.iter().fold((0, 0), |acc, win| match win {
            Owner::Even => (acc.0 + 1, acc.1),
            Owner::Odd => (acc.0, acc.1 + 1),
        });
        tracing::info!(even_wins, odd_wins, "Results");

        // println!("Solution: {:#?}", solution);
        println!("Took: {:?}", now.elapsed());
    }
}
