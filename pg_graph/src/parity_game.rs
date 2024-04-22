use std::collections::HashMap;
use std::ops::Index;
use itertools::Itertools;
use petgraph::adj::IndexType;
use petgraph::graph::NodeIndex;
use petgraph::prelude::EdgeRef;
use crate::{Owner, Vertex};

pub type VertexId<Ix = u32> = NodeIndex<Ix>;
pub type Priority = u32;

#[derive(Debug, Clone)]
pub struct ParityGame<V = Vertex, Ix: IndexType = u32> {
    pub graph: petgraph::Graph<V, (), petgraph::Directed, Ix>,
    inverted_vertices: Vec<Vec<VertexId<Ix>>>,
    labels: Vec<Option<String>>,
}

impl<Ix: IndexType> ParityGame<Vertex, Ix> {
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

        let vertices: Vec<_> = parsed_game.iter().map(|v| Vertex {
            priority: v.priority.try_into().expect("Priority too large"),
            owner: v.owner.try_into().expect("Impossible failure"),
        }).map(|v| out.graph.add_node(v)).collect();

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
    
    pub fn label(&self, vertex_id: NodeIndex<Ix>) -> Option<&str> {
        self.labels[vertex_id.index()].as_deref()
    }

    pub fn vertex_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn vertices_index(&self) -> impl Iterator<Item=NodeIndex<Ix>> + '_ {
        self.graph.node_indices()
    }
    
    pub fn vertices(&self) -> impl Iterator<Item=&Vertex> + '_ {
        self.graph.node_weights()
    }

    pub fn vertices_by_priority(&self, priority: Priority) -> impl Iterator<Item=&Vertex> + '_{
        self.vertices().filter(move |v| v.priority == priority)
    }

    pub fn vertices_by_priority_idx(&self, priority: Priority) -> impl Iterator<Item=(NodeIndex<Ix>, &Vertex)> + '_{
        self.vertices_index().zip(self.vertices()).filter(move |v| v.1.priority == priority)
    }

    pub fn get(&self, id: NodeIndex<Ix>) -> Option<&Vertex> {
        self.graph.node_weight(id)
    }

    /// Return all predecessors of the given vertex
    ///
    /// Efficiently pre-calculated.
    pub fn predecessors(&self, id: NodeIndex<Ix>) -> impl Iterator<Item=NodeIndex<Ix>> + '_ {
        // self.graph.edges_directed(id, Direction::Incoming).map(|e| e.source())
        self.inverted_vertices.get(id.index()).map(|v| v.iter().copied()).into_iter().flatten()
    }

    /// Return the maximal priority found in the given game.
    pub fn priority_max(&self) -> Priority {
        self.vertices().map(|v| v.priority).max().unwrap_or_default()
    }

    /// Calculate all the unique priorities present in the given game
    pub fn priorities_unique(&self) -> impl Iterator<Item=Priority> + '_ {
        self.vertices().map(|v| v.priority).unique()
    }

    /// Count the amount of vertices for each priority
    pub fn priorities_class_count(&self) -> ahash::HashMap<Priority, u32> {
        self.vertices().fold(HashMap::default(), |mut hash: ahash::HashMap<Priority, u32>, v| {
            hash.entry(v.priority).and_modify(|count| *count += 1).or_insert(1);
            hash
        })
    }

    pub fn has_edge(&self, from: NodeIndex<Ix>, to: NodeIndex<Ix>) -> bool {
        self.graph.contains_edge(from, to)
    }
    
    pub fn edges(&self, v: NodeIndex<Ix>) -> impl Iterator<Item = NodeIndex<Ix>> + '_ {
        self.graph.edges(v).map(|e| e.target())
    }
    
    pub fn graph_edges(&self) -> petgraph::graph::EdgeReferences<'_, (), Ix> {
        self.graph.edge_references()
    }

    pub fn to_mermaid(&self) -> String {
        use std::fmt::Write;
        let mut output = String::from("flowchart TD\n");

        for (v_id, v) in self.graph.node_indices().zip(self.graph.node_weights()) {
            let (open_token, close_token) = match v.owner {
                Owner::Even => ("(", ")"),
                Owner::Odd => ("{{", "}}"),
            };

            writeln!(
                &mut output,
                "{}{}\"{priority},{id}({orig_label})\"{close}",
                v_id.index(),
                open_token,
                priority = v.priority,
                id = v_id.index(),
                orig_label = self.label(v_id).unwrap_or_default(),
                close = close_token
            )
                .unwrap();

            for edge in self.graph.edges(v_id) {
                writeln!(&mut output, "{} --> {}", v_id.index(), edge.target().index()).unwrap();
            }
        }

        output
    }

    pub fn to_pg(&self) -> String {
        use std::fmt::Write;
        let mut output = format!("parity {};\n", self.vertex_count());

        for (v_id, v) in self.graph.node_indices().zip(self.graph.node_weights()) {
            writeln!(
                &mut output,
                "{id} {priority} {owner:?} {edges} \"{label}\";",
                id=v_id.index(),
                priority = v.priority,
                owner=v.owner as u8,
                edges=self.graph.edges(v_id).map(|v| v.target().index()).join(","),
                label = self.label(v_id.into()).unwrap_or_default(),
            )
                .unwrap();
        }

        output
    }
}

impl<Ix: IndexType> Index<VertexId<Ix>> for ParityGame<Vertex, Ix> {
    type Output = Vertex;

    fn index(&self, index: VertexId<Ix>) -> &Self::Output {
        self.graph.node_weight(index).expect("Invalid Id")
    }
}