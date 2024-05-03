use std::{collections::HashMap, fmt::Write, ops::Index};

use ahash::HashSet;
use itertools::Itertools;
use petgraph::{
    adj::IndexType,
    graph::{EdgeReference, NodeIndex},
    prelude::EdgeRef,
};

use crate::{Vertex, visualize::VisualVertex};

pub type VertexId<Ix = u32> = NodeIndex<Ix>;
pub type Priority = u32;

pub trait ParityGraph<Ix: IndexType = u32>: Sized {
    type Parent: ParityGraph<Ix>;

    fn vertex_count(&self) -> usize;

    fn vertices_index(&self) -> impl Iterator<Item = NodeIndex<Ix>> + '_;

    fn vertices(&self) -> impl Iterator<Item = &Vertex> + '_;

    fn vertices_and_index(&self) -> impl Iterator<Item = (NodeIndex<Ix>, &Vertex)> + '_ {
        self.vertices_index().zip(self.vertices())
    }

    fn label(&self, vertex_id: NodeIndex<Ix>) -> Option<&str>;

    fn vertices_by_priority(&self, priority: Priority) -> impl Iterator<Item = &Vertex> + '_ {
        self.vertices().filter(move |v| v.priority == priority)
    }

    fn vertices_by_priority_idx(&self, priority: Priority) -> impl Iterator<Item = (NodeIndex<Ix>, &Vertex)> + '_ {
        self.vertices_index()
            .zip(self.vertices())
            .filter(move |v| v.1.priority == priority)
    }

    fn get(&self, id: NodeIndex<Ix>) -> Option<&Vertex>;

    /// Return all predecessors of the given vertex
    ///
    /// Efficiently pre-calculated.
    fn predecessors(&self, id: NodeIndex<Ix>) -> impl Iterator<Item = NodeIndex<Ix>> + '_;

    /// Create a sub-game by excluding all vertices in `exclude`.
    ///
    /// Note that `exclude` should be sorted!
    fn create_subgame<'a>(&'a self, exclude: &'a ahash::HashSet<NodeIndex<Ix>>) -> SubGame<Ix, Self::Parent>;

    /// Return the maximal priority found in the given game.
    fn priority_max(&self) -> Priority {
        self.vertices().map(|v| v.priority).max().expect("No node in graph")
    }

    /// Calculate all the unique priorities present in the given game
    fn priorities_unique(&self) -> impl Iterator<Item = Priority> + '_ {
        self.vertices().map(|v| v.priority).unique()
    }

    /// Count the amount of vertices for each priority
    fn priorities_class_count(&self) -> ahash::HashMap<Priority, u32> {
        self.vertices()
            .fold(HashMap::default(), |mut hash: ahash::HashMap<Priority, u32>, v| {
                hash.entry(v.priority).and_modify(|count| *count += 1).or_insert(1);
                hash
            })
    }

    fn has_edge(&self, from: NodeIndex<Ix>, to: NodeIndex<Ix>) -> bool;

    fn edges(&self, v: NodeIndex<Ix>) -> impl Iterator<Item = NodeIndex<Ix>> + '_;

    fn graph_edges(&self) -> impl Iterator<Item = EdgeReference<'_, (), Ix>>;

    fn to_pg(&self) -> String {
        use std::fmt::Write;
        let mut output = format!("parity {};\n", self.vertex_count());

        for (v_id, v) in self.vertices_index().zip(self.vertices()) {
            writeln!(
                &mut output,
                "{id} {priority} {owner:?} {edges} \"{label}\";",
                id = v_id.index(),
                priority = v.priority,
                owner = v.owner as u8,
                edges = self.edges(v_id).map(|target| target.index()).join(","),
                label = self.label(v_id).unwrap_or_default(),
            )
            .unwrap();
        }

        output
    }
}

#[derive(Debug, Clone)]
pub struct ParityGame<Ix: IndexType = u32> {
    pub graph: petgraph::Graph<Vertex, (), petgraph::Directed, Ix>,
    inverted_vertices: Vec<Vec<VertexId<Ix>>>,
    labels: Vec<Option<String>>,
}

impl<Ix: IndexType> ParityGame<Ix> {
    pub fn empty() -> Self {
        Self {
            graph: petgraph::Graph::with_capacity(20, 20),
            inverted_vertices: vec![],
            labels: Vec::new(),
        }
    }

    pub fn new(parsed_game: Vec<pg_parser::Vertex>) -> eyre::Result<Self> {
        // Ensure the ids are correctly ordered
        if !parsed_game.is_sorted_by(|a, b| a.id < b.id) {
            eyre::bail!("Parsed game is not ordered by ids as expected!");
        }

        let mut out = Self::empty();

        let vertices: Vec<_> = parsed_game
            .iter()
            .map(|v| Vertex {
                priority: v.priority.try_into().expect("Priority too large"),
                owner: v.owner.try_into().expect("Impossible failure"),
            })
            .map(|v| out.graph.add_node(v))
            .collect();

        for (p_v, v_idx) in parsed_game.iter().zip(vertices) {
            for &target_v in &p_v.outgoing_edges {
                out.graph.add_edge(v_idx, NodeIndex::new(target_v), ());
            }
        }

        let mut inverted_graph = vec![Vec::new(); out.graph.node_count()];

        for vertex in out.graph.node_indices() {
            for edge in out.graph.edges(vertex) {
                inverted_graph[edge.target().index()].push(vertex);
            }
        }

        out.inverted_vertices = inverted_graph;
        out.labels = parsed_game.into_iter().map(|v| v.label.map(|v| v.into())).collect();

        Ok(out)
    }
}

impl<Ix: IndexType> ParityGraph<Ix> for ParityGame<Ix> {
    type Parent = Self;

    fn label(&self, vertex_id: NodeIndex<Ix>) -> Option<&str> {
        self.labels[vertex_id.index()].as_deref()
    }

    fn vertex_count(&self) -> usize {
        self.graph.node_count()
    }

    fn vertices_index(&self) -> impl Iterator<Item = NodeIndex<Ix>> + '_ {
        self.graph.node_indices()
    }

    fn vertices(&self) -> impl Iterator<Item = &Vertex> + '_ {
        self.graph.node_weights()
    }

    fn get(&self, id: NodeIndex<Ix>) -> Option<&Vertex> {
        self.graph.node_weight(id)
    }

    fn predecessors(&self, id: NodeIndex<Ix>) -> impl Iterator<Item = NodeIndex<Ix>> + '_ {
        self.inverted_vertices
            .get(id.index())
            .map(|v| v.iter().copied())
            .into_iter()
            .flatten()
    }

    fn has_edge(&self, from: NodeIndex<Ix>, to: NodeIndex<Ix>) -> bool {
        self.graph.contains_edge(from, to)
    }

    fn edges(&self, v: NodeIndex<Ix>) -> impl Iterator<Item = NodeIndex<Ix>> + '_ {
        self.graph.edges(v).map(|e| e.target())
    }

    fn graph_edges(&self) -> impl Iterator<Item = EdgeReference<'_, (), Ix>> {
        self.graph.edge_references()
    }

    fn create_subgame<'a>(&'a self, exclude: &'a HashSet<NodeIndex<Ix>>) -> SubGame<Ix, Self::Parent> {
        SubGame {
            parent: &self,
            ignored: exclude.clone(),
        }
    }
}

impl<T: ParityGraph<u32>> crate::visualize::VisualGraph for T {
    fn vertices(&self) -> Box<dyn Iterator<Item = VisualVertex> + '_> {
        Box::new(
            self.vertices_and_index()
                .map(|(i, v)| VisualVertex { id: i, owner: v.owner }),
        )
    }

    fn edges(&self) -> Box<dyn Iterator<Item = (VertexId, VertexId)> + '_> {
        Box::new(self.graph_edges().map(|e| (e.source(), e.target())))
    }

    fn node_text(&self, node: VertexId, sink: &mut dyn Write) -> std::fmt::Result {
        let v = self.get(node).expect("Invalid vertex id");
        write!(
            sink,
            "{priority},{id}({orig_label})",
            priority = v.priority,
            id = node.index(),
            orig_label = self.label(node).unwrap_or_default(),
        )
    }

    fn edge_text(&self, _edge: (VertexId, VertexId), _sink: &mut dyn Write) -> std::fmt::Result {
        Ok(())
    }
}

impl<Ix: IndexType> Index<VertexId<Ix>> for ParityGame<Ix> {
    type Output = Vertex;

    fn index(&self, index: VertexId<Ix>) -> &Self::Output {
        self.graph.node_weight(index).expect("Invalid Id")
    }
}

pub struct SubGame<'a, Ix: IndexType, Parent: ParityGraph<Ix>> {
    parent: &'a Parent,
    ignored: ahash::HashSet<NodeIndex<Ix>>,
}

impl<'a, Ix: IndexType, Parent: ParityGraph<Ix>> ParityGraph<Ix> for SubGame<'a, Ix, Parent> {
    type Parent = Parent;

    fn vertex_count(&self) -> usize {
        self.vertices_index().count()
    }

    fn vertices_index(&self) -> impl Iterator<Item = NodeIndex<Ix>> + '_ {
        self.parent.vertices_index().filter(|ix| !self.ignored.contains(ix))
    }

    fn vertices(&self) -> impl Iterator<Item = &Vertex> + '_ {
        self.vertices_index().flat_map(|ix| self.parent.get(ix))
    }

    fn label(&self, vertex_id: NodeIndex<Ix>) -> Option<&str> {
        if !self.ignored.contains(&vertex_id) {
            self.parent.label(vertex_id)
        } else {
            None
        }
    }

    fn get(&self, id: NodeIndex<Ix>) -> Option<&Vertex> {
        self.parent.get(id)
    }

    fn predecessors(&self, id: NodeIndex<Ix>) -> impl Iterator<Item = NodeIndex<Ix>> + '_ {
        self.parent.predecessors(id).filter(|idx| !self.ignored.contains(idx))
    }

    fn create_subgame<'b>(&'b self, exclude: &'b HashSet<NodeIndex<Ix>>) -> SubGame<Ix, Self::Parent> {
        let mut new_ignore = self.ignored.clone();
        new_ignore.extend(exclude);
        SubGame {
            parent: self.parent,
            ignored: new_ignore,
        }
    }

    fn has_edge(&self, from: NodeIndex<Ix>, to: NodeIndex<Ix>) -> bool {
        !self.ignored.contains(&from) && !self.ignored.contains(&to) && self.parent.has_edge(from, to)
    }

    fn edges(&self, v: NodeIndex<Ix>) -> impl Iterator<Item = NodeIndex<Ix>> + '_ {
        self.parent.edges(v).filter(|idx| !self.ignored.contains(idx))
    }

    fn graph_edges(&self) -> impl Iterator<Item = EdgeReference<'_, (), Ix>> {
        self.parent
            .graph_edges()
            .filter(|edge| !self.ignored.contains(&edge.source()) && !self.ignored.contains(&edge.target()))
    }
}
