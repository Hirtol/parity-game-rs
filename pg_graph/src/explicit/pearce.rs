//! Iterative implementation of Pearce's algorithm based on the one from Petgraph, and https://whileydave.com/publications/Pea16_IPL_preprint.pdf

use crate::explicit::{ParityGraph, VertexId, VertexSet};
use std::{collections::VecDeque, num::NonZeroU32};

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

    #[profiling::function]
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

    #[profiling::function]
    #[inline]
    fn visit(
        &mut self,
        v_id: VertexId<u32>,
        graph: &impl ParityGraph<u32>,
        mut scc_found: impl FnMut(PearceSccIter<'_>),
    ) {
        self.begin_visit(v_id);

        while let Some(&to_explore) = self.v_s.front() {
            unsafe {
                self.visit_loop(to_explore, graph, &mut scc_found);
            }
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

    #[profiling::function]
    #[inline]
    unsafe fn visit_loop(
        &mut self,
        v_id: VertexId<u32>,
        graph: &impl ParityGraph<u32>,
        scc_found: impl FnMut(PearceSccIter<'_>),
    ) {
        let edge_i = *self.i_s.front().expect("Impossible Invariant Violation");

        for (i, edge) in graph.edges(v_id).enumerate().skip(edge_i as usize) {
            // First check if `i` was in range
            // New node to explore, restart visit loop, begin edge
            if self.root_index.get_unchecked(edge.index()).is_none() {
                // Add the current index back in order to `finish_edge` after we finish exploring this new node.
                *self.i_s.front_mut().expect("impossible") = i as u32;
                self.begin_visit(edge);
                return;
            } else {
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
    unsafe fn finish_visit(&mut self, v_id: VertexId<u32>, mut scc_found: impl FnMut(PearceSccIter<'_>)) {
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
                    if root_indexes.get_unchecked(v_id.index()) > root_indexes.get_unchecked(part_of_scc.index()) {
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
            *self.root_index.get_unchecked_mut(v_id.index()) = c;
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
mod tests {

    // #[test]
    // pub fn test_pearce() {
    //     let game = load_example("basic_paper_example.pg");
    //     let mut pearce = PearceScc::new(game.original_vertex_count());
    //
    //     let sccs = petgraph::algo::tarjan_scc(&game.graph);
    //
    //     let mut own_sccs = Vec::new();
    //
    //     pearce.run(&game, |found| {
    //         let data = found.copied().collect_vec();
    //         println!("Found SCC `{data:?}`");
    //         own_sccs.push(data)
    //     });
    //
    //     println!("Found SCCS: {:?}", sccs);
    //     println!("\nFound Own SCCS: {:?}", own_sccs);
    //
    //     assert_eq!(sccs, own_sccs);
    // }
    //
    // pub fn bench_pearce() {
    //     const ROUNDS: usize = 1000;
    //     let game = load_example("TwoCountersInRangeA6.tlsf.ehoa.pg");
    //     let now = std::time::Instant::now();
    //     let mut sccs = Vec::new();
    //     for i in 0..ROUNDS {
    //         sccs = petgraph::algo::tarjan_scc(&game.graph);
    //     }
    //     let elapsed = now.elapsed();
    //     println!(
    //         "Pet Elapsed: {elapsed:?} - per: {} ms",
    //         elapsed.as_secs_f32() * 1000. / ROUNDS as f32
    //     );
    //
    //     let mut own_sccs = Vec::new();
    //     let now = std::time::Instant::now();
    //     for i in 0..ROUNDS {
    //         let mut pearce = PearceScc::new(game.original_vertex_count());
    //         own_sccs = Vec::new();
    //
    //         pearce.run(&game, |found| {
    //             let data = found.copied().collect_vec();
    //             own_sccs.push(data)
    //         });
    //     }
    //     let elapsed = now.elapsed();
    //     println!(
    //         "Own Elapsed: {elapsed:?} - per: {} ms",
    //         elapsed.as_secs_f32() * 1000. / ROUNDS as f32
    //     );
    //
    //     assert_eq!(sccs, own_sccs);
    // }

    // #[test]
    // pub fn test_pearce_correct() {
    //     for name in tests::examples_iter() {
    //         println!("Running test for: {name}...");
    //         let (game, compare) = tests::load_and_compare_example(&name);
    //
    //         #[cfg(debug_assertions)]
    //         if game.original_vertex_count() > 30_000 {
    //             println!("Skipping test as we're running in debug mode, leading to stack overflows in Petgraph");
    //             continue;
    //         }
    //         let now = std::time::Instant::now();
    //         let sccs = petgraph::algo::tarjan_scc(&game.graph);
    //         println!("Petgraph took: {:?}", now.elapsed());
    //
    //         let mut pearce = PearceScc::new(game.original_vertex_count());
    //         let mut own_sccs = Vec::new();
    //         let now = std::time::Instant::now();
    //         pearce.run(&game, |found| {
    //             let data = found.copied().collect_vec();
    //             own_sccs.push(data)
    //         });
    //         println!("We took: {:?}", now.elapsed());
    //         assert_eq!(sccs, own_sccs);
    //         println!("{name} correct!")
    //     }
    // }
}
