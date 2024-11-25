use crate::{
    datatypes::Priority,
    explicit::{BitsetExtensions, VertexSet},
    visualize::VisualVertex,
    Owner, ParityVertexSoa, Vertex,
};
use fixedbitset::FixedBitSet;
use itertools::Itertools;
use petgraph::{adj::IndexType, graph::NodeIndex};
use pg_parser::PgBuilder;
use soa_rs::{Soa, Soars};
use std::{
    collections::HashMap,
    fmt::{Debug, Write},
    marker::PhantomData,
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

    fn original_game(&self) -> &Self::Parent;

    fn empty_vertex_set(&self) -> VertexSet {
        VertexSet::empty_game(self)
    }

    fn vertex_count(&self) -> usize;

    fn edge_count(&self) -> usize;

    fn vertices_index(&self) -> impl Iterator<Item = NodeIndex<Ix>> + '_;

    fn label(&self, vertex_id: NodeIndex<Ix>) -> Option<&str>;

    #[inline(always)]
    fn vertices_index_by_priority(&self, priority: Priority) -> impl Iterator<Item = NodeIndex<Ix>> + '_ {
        self.vertices_index().filter(move |&v| self.priority(v) == priority)
    }

    /// List all vertices with the given priority, and any vertices with a lower priority with the same parity, so long as
    /// no other priorities of the opposing parity are in between these values.
    #[inline(always)]
    fn vertices_by_compressed_priority(&self, priority: Priority) -> impl Iterator<Item = NodeIndex<Ix>> + '_ {
        let mut current_priority = priority;
        let mut next_highest_priority = None;
        let mut current_itr = self.vertices_index();
        std::iter::from_fn(move || {
            loop {
                for v_idx in current_itr.by_ref() {
                    let p = self.priority(v_idx);

                    if p == current_priority {
                        return Some(v_idx)
                    } else if next_highest_priority.map(|next| p > next).unwrap_or(true) && p < current_priority {
                        next_highest_priority = Some(p);
                    }
                }

                if let Some(next) = next_highest_priority {
                    if Owner::from_priority(next) == Owner::from_priority(current_priority) {
                        current_itr = self.vertices_index();
                        current_priority = next;
                        next_highest_priority = None;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            None
        })
    }

    /// Get the full vertex
    #[inline(always)]
    fn get_vertex(&self, id: NodeIndex<Ix>) -> Option<Vertex> {
        Some(Vertex {
            priority: self.get_priority(id)?,
            owner: self.get_owner(id)?,
        })
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

    /// Create a sub-game by excluding all vertices in `exclude`.
    ///
    /// Note that `exclude` should have a length equal to [Self::original_vertex_count]
    fn create_subgame_bit(&self, exclude: &FixedBitSet) -> SubGame<Ix, Self::Parent>;

    /// Return the maximal priority found in the given game.
    #[inline(always)]
    fn priority_max(&self) -> Priority {
        self.vertices_index()
            .map(|v| self.priority(v))
            .max()
            .expect("No node in graph")
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
                hash.entry(self.priority(v))
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
                hash
            })
    }

    fn has_edge(&self, from: NodeIndex<Ix>, to: NodeIndex<Ix>) -> bool;

    fn edges(&self, v: NodeIndex<Ix>) -> impl Iterator<Item = NodeIndex<Ix>> + '_;

    fn vertex_edge_count(&self, v: NodeIndex<Ix>) -> usize {
        self.edges(v).count()
    }

    fn graph_edges(&self) -> impl Iterator<Item = (VertexId<Ix>, VertexId<Ix>)>;

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
                label = v_id.index(),
            )
            .unwrap();
        }

        output
    }
}

pub struct ParityGameBuilder<V: Soars, Ix: IndexType = u32> {
    vertices: Soa<V>,
    /// Will always have the last element indicate the end `len` of `edges`
    edge_indexes: Vec<usize>,
    edges: Vec<VertexId<Ix>>,
    in_degrees: Vec<u32>,
    labels: Vec<Option<String>>,
}

impl<Ix: IndexType, Vert: Soars> ParityGameBuilder<Vert, Ix> {
    pub fn new() -> Self {
        Self {
            vertices: Soa::default(),
            edge_indexes: vec![0],
            edges: vec![],
            in_degrees: vec![],
            labels: vec![],
        }
    }

    pub fn build(self) -> ParityGame<Ix, Vert> {
        let num_vertices = self.vertices.len();
        // Compute the start indexes for our inverted edges
        let mut inverted_indexes = vec![0; num_vertices + 1];
        for i in 0..num_vertices {
            inverted_indexes[i + 1] = inverted_indexes[i] + self.in_degrees[i] as usize;
        }

        // Ensure we have a fully populated CSR inverted indexes array.
        let mut inverted_edges = vec![VertexId::default(); self.edges.len()];
        let mut current_position = inverted_indexes.clone();
        for u in 0..num_vertices {
            let start = self.edge_indexes[u];
            let end = self.edge_indexes[u + 1];
            for v in &self.edges[start..end] {
                let pos = current_position[v.index()];
                inverted_edges[pos] = VertexId::new(u);
                current_position[v.index()] += 1;
            }
        }

        ParityGame {
            vertices: self.vertices,
            edge_indexes: self.edge_indexes,
            edges: self.edges,
            inverted_indexes,
            inverted_edges,
            labels: self.labels,
        }
    }

    /// Allocate enough room to accommodate `vertex_count` vertices without having to do intermediate re-allocations.
    pub fn preallocate(&mut self, vertex_count: usize) {
        self.labels = Vec::with_capacity(vertex_count);
        self.in_degrees = vec![0; vertex_count];
        self.edges = Vec::with_capacity(vertex_count);
        self.edge_indexes.reserve(vertex_count);
    }

    /// Add a vertex to the end of the list.
    ///
    /// Any outgoing edges from this vertex need to be added before the next `add_vertex` call.
    #[inline]
    pub fn push_vertex(&mut self, id: usize, vertex: Vert) -> eyre::Result<VertexId<Ix>> {
        if self.vertices.len() != id {
            eyre::bail!("Non-contiguous vertex IDs!")
        }

        self.vertices.push(vertex);

        Ok(VertexId::new(id))
    }

    /// These edges will be added to the most recently added vertex.
    #[inline]
    pub fn push_edges(&mut self, edges: impl Iterator<Item = VertexId<Ix>>) {
        for edge in edges {
            // The header is not guaranteed to be correct
            while self.in_degrees.len() <= edge.index() {
                self.in_degrees.push(0);
            }
            self.edges.push(edge);
            self.in_degrees[edge.index()] += 1;
        }

        // Maintain invariant that the last element in `self.edge_indexes` is always the len of `edges`.
        self.edge_indexes.push(self.edges.len());
    }
}

impl<'a, Ix: IndexType> PgBuilder<'a> for ParityGameBuilder<Vertex, Ix> {
    fn set_header(&mut self, vertex_count: usize) -> eyre::Result<()> {
        self.preallocate(vertex_count);
        Ok(())
    }

    #[inline(always)]
    fn add_vertex(&mut self, id: usize, vertex: pg_parser::Vertex<'a>) -> eyre::Result<()> {
        self.push_vertex(
            id,
            Vertex {
                priority: vertex.priority as Priority,
                owner: vertex.owner.try_into()?,
            },
        )?;
        self.labels.push(vertex.label.map(|v| v.into()));

        // Add edges
        self.push_edges(vertex.outgoing_edges.into_iter().map(VertexId::new));

        Ok(())
    }
}

pub struct ParityGame<Ix: IndexType = u32, V: Soars = Vertex> {
    pub vertices: Soa<V>,
    /// Stored in CSR format, where the last element of edge_indexes is not an actual vertex, thus it is always safe to
    /// index with `[v_id, v_id + 1]`.
    pub edge_indexes: Vec<usize>,
    pub edges: Vec<VertexId<Ix>>,
    pub inverted_indexes: Vec<usize>,
    pub inverted_edges: Vec<VertexId<Ix>>,
    pub(crate) labels: Vec<Option<String>>,
}

impl<Ix: IndexType, Vert: Soars> ParityGame<Ix, Vert> {
    pub fn empty() -> Self {
        Self {
            vertices: Soa::default(),
            edge_indexes: vec![0],
            edges: vec![],
            // inverted_vertices: vec![],
            inverted_indexes: vec![0],
            inverted_edges: vec![],
            labels: Vec::new(),
        }
    }
}

impl<Ix: IndexType, Vert: Soars> ParityGraph<Ix> for ParityGame<Ix, Vert>
where
    Soa<Vert>: ParityVertexSoa<Ix>,
{
    type Parent = Self;

    fn original_game(&self) -> &Self::Parent {
        self
    }

    #[inline(always)]
    fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    #[inline]
    fn edge_count(&self) -> usize {
        self.edges.len()
    }

    #[inline(always)]
    fn vertices_index(&self) -> impl Iterator<Item = VertexId<Ix>> + '_ {
        (0..self.vertex_count()).map(VertexId::new)
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
        self.vertices.priority_of(id)
    }

    #[inline(always)]
    fn get_owner(&self, id: NodeIndex<Ix>) -> Option<Owner> {
        self.vertices.get_owner(id)
    }

    #[inline(always)]
    fn owner(&self, id: NodeIndex<Ix>) -> Owner {
        self.vertices.owner_of(id)
    }

    #[inline(always)]
    fn predecessors(&self, id: NodeIndex<Ix>) -> impl Iterator<Item = NodeIndex<Ix>> + '_ {
        let (edges_start, edges_end) = (self.inverted_indexes[id.index()], self.inverted_indexes[id.index() + 1]);
        self.inverted_edges[edges_start..edges_end].iter().copied()
        // self.inverted_vertices
        //     .get(id.index())
        //     .map(|v| v.iter().copied())
        //     .into_iter()
        //     .flatten()
    }

    fn create_subgame(&self, exclude: impl IntoIterator<Item = NodeIndex<Ix>>) -> SubGame<Ix, Self::Parent> {
        create_subgame(self, exclude)
    }

    fn create_subgame_bit(&self, exclude: &FixedBitSet) -> SubGame<Ix, Self::Parent> {
        create_subgame_bit(self, exclude)
    }

    #[inline(always)]
    fn has_edge(&self, from: NodeIndex<Ix>, to: NodeIndex<Ix>) -> bool {
        self.edges(from).contains(&to)
    }

    #[inline(always)]
    fn edges(&self, v: NodeIndex<Ix>) -> impl Iterator<Item = NodeIndex<Ix>> + '_ {
        let (edges_start, edges_end) = (self.edge_indexes[v.index()], self.edge_indexes[v.index() + 1]);
        self.edges[edges_start..edges_end].iter().copied()
    }

    #[inline(always)]
    fn graph_edges(&self) -> impl Iterator<Item = (VertexId<Ix>, VertexId<Ix>)> {
        self.edge_indexes.windows(2).enumerate().flat_map(|(source, targets)| {
            let source = VertexId::new(source);

            self.edges[targets[0]..targets[1]]
                .iter()
                .map(move |target| (source, *target))
        })
    }
}

impl<Ix: IndexType, T: ParityGraph<Ix>> crate::visualize::VisualGraph<Ix> for T {
    fn vertices(&self) -> Box<dyn Iterator<Item = VisualVertex<Ix>> + '_> {
        Box::new(self.vertices_index().map(|i| VisualVertex {
            id: i,
            owner: self.owner(i),
        }))
    }

    fn edges(&self) -> Box<dyn Iterator<Item = (VertexId<Ix>, VertexId<Ix>)> + '_> {
        Box::new(self.graph_edges())
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

#[derive(Debug)]
pub struct SubGame<'a, Ix: IndexType, Parent> {
    pub(crate) parent: &'a Parent,
    pub(crate) game_vertices: VertexSet,
    pub(crate) len: usize,
    pub(crate) _phant: PhantomData<Ix>,
}

impl<'a, Ix: IndexType, Parent> Clone for SubGame<'a, Ix, Parent> {
    fn clone(&self) -> Self {
        Self {
            parent: &self.parent,
            game_vertices: self.game_vertices.clone(),
            len: self.len,
            _phant: Default::default(),
        }
    }
}

impl<'a, Ix: IndexType, Parent: ParityGraph<Ix>> SubGame<'a, Ix, Parent> {
    pub fn from_vertex_set(parent: &'a Parent, set: VertexSet) -> Self {
        SubGame {
            parent,
            len: set.count_ones(..),
            game_vertices: set,
            _phant: Default::default(),
        }
    }

    pub fn parent(&self) -> &'a Parent {
        self.parent
    }

    #[inline]
    pub fn vertex_in_subgame(&self, vertex_id: VertexId<Ix>) -> bool {
        self.game_vertices.contains(vertex_id.index())
    }

    /// Similar to [Self::create_subgame()], but `self` is not borrowed for the new subgame, and instead modified in place.
    pub fn shrink_subgame(&mut self, exclude: &VertexSet) {
        self.game_vertices.difference_with(exclude);
        self.len = self.game_vertices.count_ones(..);
    }

    /// Create a new [SubGame] by intersecting self with the given [VertexSet]
    pub fn intersect_subgame(&mut self, intersect: &VertexSet) {
        self.game_vertices.intersect_with(intersect);
        self.len = self.game_vertices.count_ones(..);
    }
}

impl<'a, Ix: IndexType, Parent: ParityGraph<Ix>> ParityGraph<Ix> for SubGame<'a, Ix, Parent> {
    type Parent = Parent;

    fn original_vertex_count(&self) -> usize {
        self.parent.vertex_count()
    }

    fn original_game(&self) -> &Self::Parent {
        self.parent
    }

    #[inline(always)]
    fn vertex_count(&self) -> usize {
        self.len
    }

    fn edge_count(&self) -> usize {
        self.parent
            .graph_edges()
            .filter(|(s, t)| self.game_vertices.contains(s.index()) && self.game_vertices.contains(t.index()))
            .count()
    }

    #[inline(always)]
    fn vertices_index(&self) -> impl Iterator<Item = NodeIndex<Ix>> + '_ {
        self.game_vertices.ones_vertices()
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
        self.parent
            .predecessors(id)
            .filter(|idx| self.game_vertices.contains(idx.index()))
    }

    #[inline(always)]
    fn create_subgame(&self, exclude: impl IntoIterator<Item = NodeIndex<Ix>>) -> SubGame<'a, Ix, Self::Parent> {
        let mut new_game = self.game_vertices.clone();
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

    fn create_subgame_bit(&self, exclude: &FixedBitSet) -> SubGame<'a, Ix, Self::Parent> {
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
        self.parent.has_edge(from, to)
            && self.game_vertices.contains(from.index())
            && self.game_vertices.contains(to.index())
    }

    #[inline(always)]
    fn edges(&self, v: NodeIndex<Ix>) -> impl Iterator<Item = NodeIndex<Ix>> + '_ {
        self.parent
            .edges(v)
            .filter(|idx| self.game_vertices.contains(idx.index()))
    }

    fn graph_edges(&self) -> impl Iterator<Item = (VertexId<Ix>, VertexId<Ix>)> {
        self.parent
            .graph_edges()
            .filter(|edge| self.game_vertices.contains(edge.0.index()) && self.game_vertices.contains(edge.1.index()))
    }
}

/// Create a generic subgame from the given `parent`.
#[inline]
pub fn create_subgame<Ix: IndexType, PG: ParityGraph<Ix>>(
    parent: &PG,
    exclude: impl IntoIterator<Item = VertexId<Ix>>,
) -> SubGame<Ix, PG> {
    let mut all_ones_subgame = FixedBitSet::with_capacity(parent.original_vertex_count());
    all_ones_subgame.insert_range(..);
    for v in exclude {
        all_ones_subgame.set(v.index(), false);
    }

    SubGame {
        parent,
        len: all_ones_subgame.count_ones(..),
        game_vertices: all_ones_subgame,
        _phant: Default::default(),
    }
}

/// Create a generic subgame from the given `parent` and exclusion bitset.
#[inline]
pub fn create_subgame_bit<'a, Ix: IndexType, PG: ParityGraph<Ix>>(
    parent: &'a PG,
    exclude: &FixedBitSet,
) -> SubGame<'a, Ix, PG> {
    let mut all_ones_subgame = FixedBitSet::with_capacity(parent.vertex_count());
    all_ones_subgame.insert_range(..);
    all_ones_subgame.difference_with(exclude);

    SubGame {
        parent,
        len: all_ones_subgame.count_ones(..),
        game_vertices: all_ones_subgame,
        _phant: Default::default(),
    }
}

pub trait OptimisedGraph<Ix: IndexType> {
    /// Return all edges starting in this vertex in the root game.
    /// 
    /// One still has to check that this edge target is valid in the current game.
    fn root_edges(&self, v_id: VertexId<Ix>) -> &[VertexId<Ix>];
    
    /// Check whether the given `v_id` is part of our graph
    fn part_of_graph(&self, v_id: VertexId<Ix>) -> bool;
}

impl<Ix: IndexType> OptimisedGraph<Ix> for ParityGame<Ix> {
    #[inline(always)]
    fn root_edges(&self, v_id: VertexId<Ix>) -> &[VertexId<Ix>] {
        let (edges_start, edges_end) = (self.edge_indexes[v_id.index()], self.edge_indexes[v_id.index() + 1]);
        &self.edges[edges_start..edges_end]
    }

    #[inline(always)]
    fn part_of_graph(&self, _v_id: VertexId<Ix>) -> bool {
        true
    }
}

impl<'a, Ix: IndexType, Parent: OptimisedGraph<Ix>> OptimisedGraph<Ix> for SubGame<'a, Ix, Parent> {
    #[inline(always)]
    fn root_edges(&self, v_id: VertexId<Ix>) -> &[VertexId<Ix>] {
        self.parent.root_edges(v_id)
    }

    #[inline(always)]
    fn part_of_graph(&self, v_id: VertexId<Ix>) -> bool {
        self.game_vertices.contains(v_id.index())
    }
}