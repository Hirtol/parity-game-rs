use crate::{datatypes::Priority, visualize::VisualVertex, Owner, ParityVertexSoa, Vertex, VertexVec};
use fixedbitset::FixedBitSet;
use itertools::Itertools;
use petgraph::{
    adj::IndexType,
    graph::{EdgeReference, NodeIndex},
    prelude::EdgeRef,
};
use pg_parser::PgBuilder;
use std::marker::PhantomData;
use std::{
    collections::HashMap,
    fmt::{Debug, Write},
};

pub type VertexId<Ix = u32> = NodeIndex<Ix>;

pub trait ParityGraph<Ix: IndexType = u32>: Sized {
    type Parent: ParityGraph<Ix>;
    
    /// Query the size of the original ParityGraph.
    /// 
    /// Useful in case one may have a SubGame.
    fn original_vertex_count(&self) -> usize {
        self.vertex_count()
    }

    fn vertex_count(&self) -> usize;

    fn edge_count(&self) -> usize;

    fn vertices_index(&self) -> impl Iterator<Item = NodeIndex<Ix>> + '_;

    fn label(&self, vertex_id: NodeIndex<Ix>) -> Option<&str>;

    #[inline(always)]
    fn vertices_index_by_priority(&self, priority: Priority) -> impl Iterator<Item = NodeIndex<Ix>> + '_ {
        self.vertices_index().filter(move |&v| self.priority(v) == priority)
    }

    /// Get the full vertex
    #[inline(always)]
    fn get_vertex(&self, id: NodeIndex<Ix>) -> Option<Vertex> {
        Some(
            Vertex {
                priority: self.get_priority(id)?,
                owner: self.get_owner(id)?,
            }
        )
    }

    /// Get the full vertex
    #[inline(always)]
    fn vertex(&self, id: NodeIndex<Ix>) -> Vertex {
        Vertex {
            priority: self.priority(id),
            owner: self.owner(id),
        }
    }

    /// Get the priority for the given vertex
    fn get_priority(&self, id: NodeIndex<Ix>) -> Option<Priority>;

    /// Get the priority for the given vertex
    #[inline(always)]
    fn priority(&self, id: NodeIndex<Ix>) -> Priority {
        self.get_priority(id).unwrap()
    }

    /// Get the owner of the given vertex
    fn get_owner(&self, id: NodeIndex<Ix>) -> Option<Owner>;

    #[inline(always)]
    /// Get the owner of the given vertex
    fn owner(&self, id: NodeIndex<Ix>) -> Owner {
        self.get_owner(id).unwrap()
    }

    /// Return all predecessors of the given vertex
    ///
    /// Efficiently pre-calculated.
    fn predecessors(&self, id: NodeIndex<Ix>) -> impl Iterator<Item = NodeIndex<Ix>> + '_;

    /// Create a sub-game by excluding all vertices in `exclude`.
    ///
    /// Note that `exclude` should be sorted!
    fn create_subgame(&self, exclude: impl IntoIterator<Item = NodeIndex<Ix>>) -> SubGame<Ix, Self::Parent>;

    fn create_subgame_bit(&self, exclude: &FixedBitSet) -> SubGame<Ix, Self::Parent> {
        todo!()
    }

    /// Return the maximal priority found in the given game.
    #[inline(always)]
    fn priority_max(&self) -> Priority {
        self.vertices_index().map(|v| self.priority(v)).max().expect("No node in graph")
    }

    /// Calculate all the unique priorities present in the given game
    #[inline(always)]
    fn priorities_unique(&self) -> impl Iterator<Item = Priority> + '_ {
        self.vertices_index().map(|v| self.priority(v)).unique()
    }

    /// Count the amount of vertices for each priority
    #[inline(always)]
    fn priorities_class_count(&self) -> ahash::HashMap<Priority, u32> {
        self.vertices_index()
            .fold(HashMap::default(), |mut hash: ahash::HashMap<Priority, u32>, v| {
                hash.entry(self.priority(v)).and_modify(|count| *count += 1).or_insert(1);
                hash
            })
    }

    fn has_edge(&self, from: NodeIndex<Ix>, to: NodeIndex<Ix>) -> bool;

    fn edges(&self, v: NodeIndex<Ix>) -> impl Iterator<Item = NodeIndex<Ix>> + '_;

    fn graph_edges(&self) -> impl Iterator<Item = EdgeReference<'_, (), Ix>>;

    fn to_pg(&self) -> String {
        use std::fmt::Write;
        let mut output = format!("parity {};\n", self.vertex_count());

        for v_id in self.vertices_index() {
            writeln!(
                &mut output,
                "{id} {priority} {owner:?} {edges} \"{label}\";",
                id = v_id.index(),
                priority = self.priority(v_id),
                owner = self.owner(v_id) as u8,
                edges = self.edges(v_id).map(|target| target.index()).join(","),
                label = self.label(v_id).unwrap_or_default(),
            )
            .unwrap();
        }

        output
    }
}

#[derive(Default)]
pub struct ParityGameBuilder<Ix: IndexType = u32, VertexSoa = VertexVec> {
    vertices: VertexSoa,
    edge_indexes: Vec<usize>,
    edges: Vec<VertexId<Ix>>,
    inverted_edges: Vec<Vec<VertexId<Ix>>>,
    labels: Vec<Option<String>>,
}

impl<Ix: IndexType, Vert: Default> ParityGameBuilder<Ix, Vert> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn build(self) -> ParityGame<Ix, Vert> {
        // Construct the petgraph first
        let mut graph = petgraph::Graph::with_capacity(self.inverted_edges.len(), self.edges.len());
        for _ in 0..self.edge_indexes.len() {
            graph.add_node(());
        }
        for (source_vertex, window) in self.edge_indexes.windows(2).enumerate() {
            let source = VertexId::new(source_vertex);
            for &edge in &self.edges[window[0]..window[1]] {
                graph.add_edge(source, edge, ());
            }
        }
        // The last edge can never be in the window
        if let Some(&last) = self.edge_indexes.last() { 
            let source = VertexId::new(self.edge_indexes.len() - 1);
            for &edge in &self.edges[last..] {
                graph.add_edge(source, edge, ());
            }
        }

        ParityGame {
            graph,
            vertices: self.vertices,
            inverted_vertices: self.inverted_edges,
            labels: self.labels,
        }
    }
}

impl<'a, Ix: IndexType> PgBuilder<'a> for ParityGameBuilder<Ix> {
    fn set_header(&mut self, vertex_count: usize) -> eyre::Result<()> {
        self.labels = Vec::with_capacity(vertex_count);
        self.inverted_edges = vec![Vec::new(); vertex_count];
        self.edges = Vec::with_capacity(vertex_count);
        self.edge_indexes = Vec::with_capacity(vertex_count);
        Ok(())
    }

    #[inline(always)]
    fn add_vertex(&mut self, id: usize, vertex: pg_parser::Vertex<'a>) -> eyre::Result<()> {
        if self.vertices.len() != id {
            eyre::bail!("Non-contiguous vertex IDs!")
        }

        self.vertices.push(Vertex {
            priority: vertex.priority as Priority,
            owner: vertex.owner.try_into()?,
        });
        self.labels.push(vertex.label.map(|v| v.into()));
        
        // Add edges
        self.edge_indexes.push(self.edges.len());
        for edge in vertex.outgoing_edges {
            // The header is not guaranteed to be correct
            while self.inverted_edges.len() <= edge {
                self.inverted_edges.push(Vec::new());
            }
            self.edges.push(VertexId::new(edge));
            self.inverted_edges[edge].push(VertexId::new(id));
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ParityGame<Ix: IndexType = u32, VertexSoa = VertexVec> {
    pub graph: petgraph::Graph<(), (), petgraph::Directed, Ix>,
    pub vertices: VertexSoa,
    pub(crate) inverted_vertices: Vec<Vec<VertexId<Ix>>>,
    pub(crate) labels: Vec<Option<String>>,
}

impl<Ix: IndexType, Vert: Default> ParityGame<Ix, Vert> {
    pub fn empty() -> Self {
        Self {
            graph: petgraph::Graph::with_capacity(20, 20),
            vertices: Vert::default(),
            inverted_vertices: vec![],
            labels: Vec::new(),
        }
    }
}

impl ParityGame {
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
            .map(|v| {
                out.vertices.push(v);
                out.graph.add_node(())
            })
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

impl<Ix: IndexType, VertexSoa: ParityVertexSoa<Ix>> ParityGraph<Ix> for ParityGame<Ix, VertexSoa> {
    type Parent = Self;

    #[inline(always)]
    fn vertex_count(&self) -> usize {
        self.graph.node_count()
    }

    fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    #[inline(always)]
    fn vertices_index(&self) -> impl Iterator<Item = NodeIndex<Ix>> + '_ {
        self.graph.node_indices()
    }

    #[inline(always)]
    fn label(&self, vertex_id: NodeIndex<Ix>) -> Option<&str> {
        self.labels.get(vertex_id.index()).and_then(|i| i.as_deref())
    }

    #[inline(always)]
    fn get_priority(&self, id: NodeIndex<Ix>) -> Option<Priority> {
        self.vertices.get_priority(id)
    }

    #[inline(always)]
    fn priority(&self, id: NodeIndex<Ix>) -> Priority {
        self.vertices.priority(id)
    }

    #[inline(always)]
    fn get_owner(&self, id: NodeIndex<Ix>) -> Option<Owner> {
        self.vertices.get_owner(id)
    }

    #[inline(always)]
    fn owner(&self, id: NodeIndex<Ix>) -> Owner {
        self.vertices.owner(id)
    }

    #[inline(always)]
    fn predecessors(&self, id: NodeIndex<Ix>) -> impl Iterator<Item = NodeIndex<Ix>> + '_ {
        self.inverted_vertices
            .get(id.index())
            .map(|v| v.iter().copied())
            .into_iter()
            .flatten()
    }

    fn create_subgame(&self, exclude: impl IntoIterator<Item = NodeIndex<Ix>>) -> SubGame<Ix, Self::Parent> {
        let mut all_ones_subgame = FixedBitSet::with_capacity(self.vertex_count());
        all_ones_subgame.insert_range(..);
        for v in exclude {
            all_ones_subgame.set(v.index(), false);
        }
        
        SubGame {
            parent: self,
            len: all_ones_subgame.count_ones(..),
            game_vertices: all_ones_subgame,
            _phant: Default::default(),
        }
    }

    fn create_subgame_bit(&self, exclude: &FixedBitSet) -> SubGame<Ix, Self::Parent> {
        // let mut all_ones_subgame = FixedBitSet::with_capacity_and_blocks(self.vertex_count(), std::iter::repeat(!0));
        let mut all_ones_subgame = FixedBitSet::with_capacity(self.vertex_count());
        all_ones_subgame.insert_range(..);
        all_ones_subgame.difference_with(exclude);
        SubGame {
            parent: self,
            len: all_ones_subgame.count_ones(..),
            game_vertices: all_ones_subgame,
            _phant: Default::default(),
        }
    }

    #[inline(always)]
    fn has_edge(&self, from: NodeIndex<Ix>, to: NodeIndex<Ix>) -> bool {
        self.graph.contains_edge(from, to)
    }

    #[inline(always)]
    fn edges(&self, v: NodeIndex<Ix>) -> impl Iterator<Item = NodeIndex<Ix>> + '_ {
        self.graph.edges(v).map(|e| e.target())
    }

    #[inline(always)]
    fn graph_edges(&self) -> impl Iterator<Item = EdgeReference<'_, (), Ix>> {
        self.graph.edge_references()
    }
}

impl<Ix: IndexType, T: ParityGraph<Ix>> crate::visualize::VisualGraph<Ix> for T {
    fn vertices(&self) -> Box<dyn Iterator<Item = VisualVertex<Ix>> + '_> {
        Box::new(
            self.vertices_index()
                .map(|i| VisualVertex { id: i, owner: self.owner(i) }),
        )
    }

    fn edges(&self) -> Box<dyn Iterator<Item = (VertexId<Ix>, VertexId<Ix>)> + '_> {
        Box::new(self.graph_edges().map(|e| (e.source(), e.target())))
    }

    fn node_text(&self, node: VertexId<Ix>, sink: &mut dyn Write) -> std::fmt::Result {
        write!(
            sink,
            "{priority},{id}({orig_label})",
            priority = self.priority(node),
            id = node.index(),
            orig_label = self.label(node).unwrap_or_default(),
        )
    }

    fn edge_text(&self, _edge: (VertexId<Ix>, VertexId<Ix>), _sink: &mut dyn Write) -> std::fmt::Result {
        write!(_sink, " ")
    }
}

pub struct SubGame<'a, Ix: IndexType, Parent: ParityGraph<Ix>> {
    pub(crate) parent: &'a Parent,
    pub(crate) game_vertices: fixedbitset::FixedBitSet,
    pub(crate) _phant: PhantomData<Ix>,
    pub(crate) len: usize,
    // pub(crate) ignored: ahash::HashSet<NodeIndex<Ix>>,
}

impl<'a, Ix: IndexType, Parent: ParityGraph<Ix>> SubGame<'a, Ix, Parent> {
    pub fn parent(&self) -> &'a Parent {
        self.parent
    }
    
    pub fn vertex_in_subgame(&self, vertex_id: VertexId<Ix>) -> bool {
        self.game_vertices.contains(vertex_id.index())
    }
}

impl<'a, Ix: IndexType, Parent: ParityGraph<Ix>> ParityGraph<Ix> for SubGame<'a, Ix, Parent>{
    type Parent = Parent;

    fn original_vertex_count(&self) -> usize {
        self.parent.vertex_count()
    }

    #[inline(always)]
    fn vertex_count(&self) -> usize {
        // More efficient than doing a `.count()` call on `vertices_index()`
        // self.game_vertices.count_ones(..)
        // self.parent.vertex_count() - self.ignored.len()
        self.len
    }

    fn edge_count(&self) -> usize {
        // TODO: This is not correct, but also doesn't really matter... so eh..
        // self.parent.edge_count() - self.ignored.len()
        self.parent.edge_count() - self.len 
    }

    #[inline(always)]
    fn vertices_index(&self) -> impl Iterator<Item = NodeIndex<Ix>> + '_ {
        // self.parent.vertices_index().filter(|ix| !self.ignored.contains(ix.index()))
        // let mut out = self.game_vertices.ones().map(VertexId::new).collect_vec();
        // tracing::warn!(?out, "Hey");
        self.game_vertices.ones().map(VertexId::new)
        // out.into_iter()
    }

    #[inline(always)]
    fn label(&self, vertex_id: NodeIndex<Ix>) -> Option<&str> {
        if self.game_vertices.contains(vertex_id.index()) {
            self.parent.label(vertex_id)
        } else {
            None
        }
    }

    #[inline(always)]
    fn get_priority(&self, id: NodeIndex<Ix>) -> Option<Priority> {
        self.parent.get_priority(id)
    }

    #[inline(always)]
    fn priority(&self, id: NodeIndex<Ix>) -> Priority {
        self.parent.priority(id)
    }

    #[inline(always)]
    fn get_owner(&self, id: NodeIndex<Ix>) -> Option<Owner> {
        self.parent.get_owner(id)
    }

    #[inline(always)]
    fn owner(&self, id: NodeIndex<Ix>) -> Owner {
        self.parent.owner(id)
    }

    #[inline(always)]
    fn predecessors(&self, id: NodeIndex<Ix>) -> impl Iterator<Item = NodeIndex<Ix>> + '_ {
        self.parent.predecessors(id).filter(|idx| self.game_vertices.contains(idx.index()))
    }

    #[inline(always)]
    fn create_subgame(&self, exclude: impl IntoIterator<Item = NodeIndex<Ix>>) -> SubGame<Ix, Self::Parent> {
        let mut new_game = self.game_vertices.clone();
        
        // new_game.extend(exclude.into_iter().map(|v| v.index()));
        for v in exclude {
            new_game.set(v.index(), false);
        }
        SubGame {
            parent: self.parent,
            len: new_game.count_ones(..),
            game_vertices: new_game,
            _phant: Default::default(),
        }
    }

    fn create_subgame_bit(&self, exclude: &FixedBitSet) -> SubGame<Ix, Self::Parent> {
        let mut new_game = self.game_vertices.clone();
        
        new_game.difference_with(exclude);
        SubGame {
            parent: self.parent,
            len: new_game.count_ones(..),
            game_vertices: new_game,
            _phant: Default::default(),
        }
    }


    #[inline(always)]
    fn has_edge(&self, from: NodeIndex<Ix>, to: NodeIndex<Ix>) -> bool {
        self.game_vertices.contains(from.index()) && self.game_vertices.contains(to.index()) && self.parent.has_edge(from, to)
    }

    #[inline(always)]
    fn edges(&self, v: NodeIndex<Ix>) -> impl Iterator<Item = NodeIndex<Ix>> + '_ {
        self.parent.edges(v).filter(|idx| self.game_vertices.contains(idx.index()))
    }

    #[inline(always)]
    fn graph_edges(&self) -> impl Iterator<Item = EdgeReference<'_, (), Ix>> {
        self.parent
            .graph_edges()
            .filter(|edge| self.game_vertices.contains(edge.source().index()) && self.game_vertices.contains(edge.target().index()))
    }
}
