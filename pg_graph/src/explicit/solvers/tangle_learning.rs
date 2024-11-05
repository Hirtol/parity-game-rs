use crate::{
    datatypes::Priority,
    explicit::{
        solvers::{AttractionComputer, Dominion, SolverOutput},
        BitsetExtensions, ParityGame, ParityGraph, SubGame, VertexId, VertexSet,
    },
    Owner, Vertex,
};
use fixedbitset::{
    sparse::{SparseBitSetCollection, SparseBitSetRef},
    specific::SubBitSet,
};
use itertools::Itertools;
use petgraph::graph::IndexType;
use soa_rs::Soars;
use std::{
    borrow::Cow,
    collections::VecDeque,
    num::{NonZeroU32, NonZeroUsize},
};

pub const NO_STRATEGY: u32 = u32::MAX;

pub struct TangleSolver<'a, Vertex: Soars> {
    game: &'a ParityGame<u32, Vertex>,
    attract: AttractionComputer<u32>,
    pearce: PearceTangleScc,
    tangles: TangleManager,
    pub tangles_found: u32,
    pub dominions_found: u32,
    pub iterations: u32,
}

impl<'a, Vertex: Soars> TangleSolver<'a, Vertex> {
    pub fn new(game: &'a ParityGame<u32, Vertex>) -> Self {
        TangleSolver {
            pearce: PearceTangleScc::new(game.vertices.len()),
            game,
            attract: AttractionComputer::new(game.vertices.len()),
            tangles_found: 0,
            dominions_found: 0,
            iterations: 0,
            tangles: Default::default(),
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
    /// They will therefore not be in `vertices`.
    /// This is an index into the [TangleManager] escape set.
    pub escape_set: usize,
}

impl<'a> TangleSolver<'a, Vertex> {
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
        let mut strategy = vec![VertexId::new(NO_STRATEGY as usize); game.original_vertex_count()];

        while current_game.vertex_count() > 0 {
            let new_dominion = self.search_dominion(&current_game, strategy.as_mut_slice());
            self.iterations += 1;
            let owner = Owner::from_priority(new_dominion.dominating_p);

            let full_dominion = self.attract.attractor_set_tangle(
                &current_game,
                owner,
                Cow::Owned(new_dominion.vertices),
                &self.tangles,
                strategy.as_mut_slice(),
            );

            tracing::debug!(
                priority = new_dominion.dominating_p,
                size = full_dominion.count_ones(..),
                "Found greater dominion"
            );

            match owner {
                Owner::Even => winning_even.union_with(&full_dominion),
                Owner::Odd => winning_odd.union_with(&full_dominion),
            }

            current_game.shrink_subgame(&full_dominion);
            self.tangles.collection.intersect_tangles(&current_game);
        }

        tracing::info!(dominions=?self.dominions_found, tangles_found=?self.tangles_found, "Finished Tangle Learning");

        (winning_even, winning_odd)
    }

    fn search_dominion(&mut self, current_game: &impl ParityGraph<u32>, strategy: &mut [VertexId<u32>]) -> Dominion {
        loop {
            let mut partial_subgame = current_game.create_subgame([]);
            let mut temp_tangles = TangleCollection::default();
            // strategy.fill(VertexId::new(NO_STRATEGY as usize));

            while partial_subgame.vertex_count() != 0 {
                let (d, starting_set) = partial_subgame.vertices_max_priority();
                let owner = Owner::from_priority(d);

                let starting_set = AttractionComputer::make_starting_set(
                    &partial_subgame,
                    starting_set,
                );

                for v in starting_set.ones() {
                    strategy[v] = VertexId::new(NO_STRATEGY as usize);
                }

                let tangle_attractor = self.attract.attractor_set_tangle(
                    &partial_subgame,
                    owner,
                    Cow::Owned(starting_set),
                    &self.tangles,
                    strategy,
                );
                crate::info!(?d, attr=?tangle_attractor.ones().collect_vec(), "Tangle Attractor");
                
                // Check if there are any escapes from this region, or if it's locally closed.
                // We only care about the vertices with the region priority.
                let leaks = tangle_attractor
                    .ones_vertices()
                    .filter(|v| current_game.priority(*v) == d)
                    .any(|v| {
                        if current_game.owner(v) == owner {
                            strategy[v.index()].index() == NO_STRATEGY as usize
                        } else {
                            partial_subgame
                                .edges(v)
                                .any(|succ| !tangle_attractor.contains(succ.index()))
                        }
                    });
                crate::debug!(?leaks, "Leaks");

                // If there are any leaks from our region then it's not worth finding tangles.
                if leaks {
                    partial_subgame.shrink_subgame(&tangle_attractor);
                    continue;
                }

                let tangle_subgame = SubGame::from_vertex_set(partial_subgame.parent, tangle_attractor);
                let possible_dominion =
                    self.extract_tangles(current_game, &tangle_subgame, d, strategy, &mut temp_tangles);

                if let Some(dominion) = possible_dominion {
                    self.tangles.collection.merge(temp_tangles);
                    return dominion;
                } else {
                    partial_subgame.shrink_subgame(&tangle_subgame.game_vertices);
                }
            }

            crate::info!(?self.iterations, "Finished iteration, next");
            self.tangles.collection.merge(temp_tangles);
            self.iterations += 1;
        }
    }

    fn extract_tangles<T: ParityGraph<u32>>(
        &mut self,
        current_game: &T,
        tangle_sub_game: &SubGame<u32, T::Parent>,
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
                    crate::info!(?tangle_size, "Found a tangle which was a dominion");
                    self.dominions_found += 1;

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

#[derive(Default)]
pub struct TangleManager {
    pub escape_list: SparseBitSetCollection,
    pub collection: TangleCollection,
}

impl TangleManager {
    #[inline]
    pub fn tangles_with_escapes(&self, owner: Owner) -> impl Iterator<Item = (&Tangle, SparseBitSetRef<'_>)> {
        self.collection
            .get_matching_set(owner)
            .iter()
            .map(|tangle| (tangle, self.escape_list.get_set_ref(tangle.escape_set)))
    }

    /// Return `true` if the given tangle fully intersects with the given `game`, and has all remaining escapes in the `in_set`.
    #[inline]
    pub fn tangle_attracted_to<Ix: IndexType, Parent: ParityGraph<Ix>>(
        &self,
        tangle: &Tangle,
        game: &SubGame<Ix, Parent>,
        in_set: &VertexSet,
    ) -> bool {
        // We might have (partially) attracted vertices in this tangle to a higher region in `game`, if so skip this tangle.
        // This can happen due to the fact that invalid tangles are only filtered out whenever we find a dominion, not during iterations.
        self.all_escapes_to(tangle, game, in_set) && tangle.vertices.is_subset(&game.game_vertices)
    }

    /// Return `true` if the given tangle has all escapes valid within `game` pointing to the `in_set`.
    #[inline]
    pub fn all_escapes_to<Ix: IndexType, Parent: ParityGraph<Ix>>(
        &self,
        tangle: &Tangle,
        game: &SubGame<Ix, Parent>,
        in_set: &VertexSet,
    ) -> bool {
        let escapes = self.escape_list.get_set_ref(tangle.escape_set);
        let restricted_set = fixedbitset::specific::LazyAnd::new(escapes, &game.game_vertices);
        restricted_set.is_subset(in_set)
    }
}

#[derive(Default)]
pub struct TangleCollection {
    even_tangles: Vec<Tangle>,
    odd_tangles: Vec<Tangle>,
}

impl TangleCollection {
    #[inline]
    pub fn insert_tangle(&mut self, tangle: Tangle) {
        self.get_matching_set_mut(tangle.owner).push(tangle);
    }

    /// Merge with a different set
    pub fn merge(&mut self, other: TangleCollection) {
        self.even_tangles.extend(other.even_tangles);
        self.odd_tangles.extend(other.odd_tangles);
    }

    /// Shrink the current set of tangles to only retain those which are a subset of the given `subgame`.
    pub fn intersect_tangles<Ix: IndexType, Parent: ParityGraph<Ix>>(&mut self, subgame: &SubGame<Ix, Parent>) {
        self.even_tangles
            .retain(|tangle| tangle.vertices.is_subset(&subgame.game_vertices));
        self.odd_tangles
            .retain(|tangle| tangle.vertices.is_subset(&subgame.game_vertices));
    }

    /// Return all tangles where the dominating priority's parity is aligned with `owner`.
    #[inline]
    pub fn get_matching_set(&self, owner: Owner) -> &[Tangle] {
        match owner {
            Owner::Even => &self.even_tangles,
            Owner::Odd => &self.odd_tangles,
        }
    }

    /// Return all tangles where the dominating priority's parity is aligned with `owner`.
    #[inline]
    pub fn get_matching_set_mut(&mut self, owner: Owner) -> &mut Vec<Tangle> {
        match owner {
            Owner::Even => &mut self.even_tangles,
            Owner::Odd => &mut self.odd_tangles,
        }
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
    pub fn run(
        &mut self,
        graph: &impl ParityGraph<u32>,
        region_owner: Owner,
        top_level_vs: impl Iterator<Item = VertexId<u32>>,
        strategy: &[VertexId<u32>],
        mut scc_found: impl FnMut(PearceSccIter<'_>, VertexId<u32>),
    ) {
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
        // Take the current vertex off the stack
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
            if name.contains("two_counters_14") {
                continue;
            }
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
        let game = load_example("amba_decomposed_arbiter_10.tlsf.ehoa.pg");
        // let game = load_example("ltl2dpa13.tlsf.ehoa.pg");
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
