use crate::{
    datatypes::Priority,
    explicit::{
        solvers::{AttractionComputer, Dominion, SolverOutput},
        ParityGame, ParityGraph, SubGame, VertexId, VertexSet,
    },
    Owner, VertexVec,
};
use std::{
    borrow::Cow,
    collections::VecDeque,
    num::{NonZeroU32, NonZeroUsize},
};

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

            let full_dominion = self.attract.attractor_set_bit(
                &current_game,
                Owner::from_priority(dominion.dominating_p),
                Cow::Borrowed(&dominion.vertices),
            );

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
        strategy: &mut [VertexId<u32>],
    ) -> (Vec<Tangle>, Dominion) {
        loop {
            let mut partial_subgame = current_game.create_subgame([]);
            let mut temp_tangles = Vec::new();

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

                    return (tangles, real_dom);
                } else {
                    temp_tangles.extend(new_tangles);
                    partial_subgame.shrink_subgame(&tangle_attractor);
                }
            }

            tangles.extend(temp_tangles);
        }
    }

    fn extract_tangles<T: ParityGraph<u32>>(
        &mut self,
        current_sub_game: &T,
        tangle_attractor: &SubGame<u32, T::Parent>,
        strategy: &mut [VertexId<u32>],
    ) -> Vec<Tangle> {
        Vec::new()
    }
}

pub struct PearceScc {
    index: u32,
    component_count: u32,
    v_s: VecDeque<VertexId<u32>>,
    i_s: VecDeque<u32>,
    root: VertexSet,
    root_index: Vec<Option<NonZeroU32>>,
}

type PearceSccIter<'a> = std::iter::Skip<std::collections::vec_deque::Iter<'a, VertexId<u32>>>;

impl PearceScc {
    pub fn new(max_size: usize) -> Self {
        Self {
            index: 1,                  // Invariant: index < component_count at all times.
            component_count: u32::MAX, // Will hold if component_count is initialized to number of nodes - 1 or higher.
            v_s: Default::default(),
            i_s: Default::default(),
            root: VertexSet::with_capacity(max_size),
            root_index: Default::default(),
        }
    }

    pub fn run(&mut self, graph: &impl ParityGraph<u32>, mut scc_found: impl FnMut(PearceSccIter<'_>)) {
        self.root_index.clear();
        self.root.clear();
        self.root_index.resize(graph.vertex_count(), None);

        for n in graph.vertices_index() {
            let visited = self.root_index[n.index()].is_some();
            if !visited {
                self.visit(n, graph, &mut scc_found);
            }
        }
    }

    fn visit(&mut self, v_id: VertexId<u32>, graph: &impl ParityGraph<u32>, mut scc_found: impl FnMut(PearceSccIter<'_>)) {
        self.begin_visit(v_id);

        while let Some(&to_explore) = self.v_s.front() {
            self.visit_loop(to_explore, graph, &mut scc_found);
        }
    }

    fn begin_visit(&mut self, v_id: VertexId<u32>) {
        self.v_s.push_front(v_id);
        self.i_s.push_front(0);
        self.root.insert(v_id.index());
        self.root_index[v_id.index()] = NonZeroU32::new(self.index);
        self.index += 1;
    }

    fn visit_loop(&mut self, v_id: VertexId<u32>, graph: &impl ParityGraph<u32>, scc_found: impl FnMut(PearceSccIter<'_>)) {
        let mut edge_i = *self.i_s.front().expect("Impossible Invariant Violation");
        let edge_count = graph.edges(v_id).count();

        for i in edge_i..=edge_count as u32 {
            println!("i: {i} - {edge_count}");
            let edge = graph.edges(v_id).nth(i as usize);
            if i > 0 {
                // TODO: Make this better
                let edge_to_finish = graph.edges(v_id).nth((i - 1) as usize).expect("Invariant Violation");
                self.finish_edge(v_id, edge_to_finish)
            }

            // First check if `i` was in range
            if let Some(edge) = edge {
                let begun_new_edge = {
                    // New node to explore
                    if self.root_index[edge.index()].is_none() {
                        self.i_s.pop_front();
                        self.i_s.push_front(i + 1);
                        self.begin_visit(edge);
                        true
                    } else {
                        false
                    }
                };

                if begun_new_edge {
                    return;
                }
            }
        }

        self.finish_visit(v_id, scc_found);
    }

    fn finish_visit(&mut self, v_id: VertexId<u32>, mut scc_found: impl FnMut(PearceSccIter<'_>)) {
        // Take the current vertex of the stack
        self.v_s.pop_front();
        self.i_s.pop_front();
        // Update a new component
        if self.root[v_id.index()] {
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
            scc_found(component_items);

            self.v_s.truncate(start);
            self.index -= index_adjust;
            self.root_index[v_id.index()] = c;
            self.component_count -= 1;
        } else {
            self.v_s.push_back(v_id);
        }
    }

    fn finish_edge(&mut self, v_id: VertexId<u32>, edge: VertexId<u32>) {
        if self.root_index(edge) < self.root_index(v_id) {
            self.root_index[v_id.index()] = self.root_index[edge.index()];
            self.root.remove(v_id.index());
        }
    }

    #[inline]
    fn root_index(&self, v_id: VertexId<u32>) -> Option<NonZeroU32> {
        self.root_index[v_id.index()]
    }
}

#[cfg(test)]
pub mod test {
    use crate::explicit::ParityGraph;
    use crate::{explicit::solvers::tangle_learning::TangleSolver, load_example, tests, Owner};
    use itertools::Itertools;
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
    pub fn test_pearce() {
        let game = load_example("two_counters_14.pg");
        let mut pearce = super::PearceScc::new(game.original_vertex_count());
        
        let sccs = petgraph::algo::tarjan_scc(&game.graph);

        let mut own_sccs = Vec::new();

        pearce.run(&game, |found| {
            println!("Found SCC");
            own_sccs.push(found.copied().collect_vec())
        });

        println!("Found SCCS: {:?}", sccs);
        println!("\nFound Own SCCS: {:?}", own_sccs);
        
        assert_eq!(sccs, own_sccs);
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
