//! See [ReducedRegisterGame]
use crate::{
    datatypes::Priority,
    explicit::{ParityGame, ParityGraph, SubGame, VertexId},
    Owner, ParityVertex,
};
use ahash::HashSet;
use ecow::{eco_vec, EcoVec};
use itertools::Itertools;
use petgraph::graph::IndexType;
use petgraph::graph::{EdgeReference, NodeIndex};
use std::{cmp::Ordering, collections::VecDeque};

/// The value type for `k` in a register game.
///
/// Practically this doesn't need more than
pub type Rank = u8;

pub struct ReducedRegisterGame<'a> {
    pub original_game: &'a crate::explicit::ParityGame,
    pub controller: Owner,
    pub vertices: Vec<RegisterVertex>,
    pub reg_v_index: ahash::HashMap<RegisterVertex, VertexId>,
    /// Mapping from the parity game vertex to all register game vertices with this vertex as root.
    pub original_to_reg_v: ahash::HashMap<VertexId, ahash::HashSet<VertexId>>,
    pub reg_quantity: usize,
}

#[derive(Debug, Clone)]
pub struct GameRegisterVertex {
    pub priority: Priority,
    pub owner: Owner,
    pub original_v: VertexId,
}

impl ParityVertex for GameRegisterVertex {
    fn priority(&self) -> Priority {
        self.priority
    }

    fn owner(&self) -> Owner {
        self.owner
    }
}

enum ToExpand {
    OriginalVertex(VertexId),
    /// Contains the register vertex id, and the register state for this particular case.
    RegisterVertex(VertexId),
}

impl<'a> ReducedRegisterGame<'a> {
    pub fn construct_2021_reduced(game: &'a crate::explicit::ParityGame, k: Rank, controller: Owner) -> Self {
        let reg_quantity = (k + 1) as usize;
        let base_registers = game
            .priorities_unique()
            .chain([0])
            .map(|pr| (pr, eco_vec!(pr; reg_quantity)))
            .collect::<ahash::HashMap<_, _>>();

        let mut to_expand: VecDeque<VertexId> = VecDeque::with_capacity(game.vertex_count());

        let mut reg_v_index: ahash::HashMap<RegisterVertex, VertexId> = ahash::HashMap::default();
        let mut original_to_reg_v = ahash::HashMap::default();
        let mut final_graph = Vec::new();

        // Only expand the new register vertex if it's actually unique.
        // Done as a macro as we mutably borrow `to_expand`
        macro_rules! add_new_register_vertex {
            ($reg:expr) => {{
                let reg = $reg;

                if let Some(&r_id) = reg_v_index.get(&reg) {
                    r_id
                } else {
                    final_graph.push(reg.clone());
                    let r_id = VertexId::new(final_graph.len() - 1);
                    reg_v_index.insert(reg, r_id);
                    // Ensure this one is listed for further expansion
                    to_expand.push_back(r_id);
                    r_id
                }
            }};
        }

        // First add all the original vertices as starting vertices.
        for v_id in game.vertices_index() {
            let v = &game[v_id];
            let register_values = base_registers
                .get(&v.priority)
                .expect("Priority register wasn't initialised");
            let new_priority = reset_to_priority_2021(0, register_values[0], v.priority, controller);

            // We assume we start with the next action being a register change instead of a move,
            // thus the owner is always the controller.
            let register_vertex = RegisterVertex {
                original_graph_id: v_id,
                priority: new_priority,
                owner: v.owner,
                register_state: register_values.clone(),
                rank_reset: 0,
            };

            let new_id: VertexId = add_new_register_vertex!(register_vertex);

            original_to_reg_v
                .entry(v_id)
                .and_modify(|v: &mut HashSet<_>| {v.insert(new_id);})
                .or_insert(HashSet::from_iter([new_id]));
        }

        // Operate on the remaining expansion queue.
        while let Some(reg_id) = to_expand.pop_back() {
            let reg_v = &final_graph[reg_id.index()];

            // NOTE: It is currently assumed that the edges are added such that the edges with the same
            // `edge_node` original vertex are consecutive. See `register_attractor`.
            for edge_node in game.edges(reg_v.original_graph_id) {
                for r in 0..reg_quantity {
                    let reg_v = &final_graph[reg_id.index()];
                    let next_v = &game[edge_node];
                    let new_priority =
                        reset_to_priority_2021(r as Rank, reg_v.register_state[r], next_v.priority, controller);
                    let mut new_registers = reg_v.register_state.clone();

                    let new_r = new_registers.make_mut();
                    next_registers_2021(new_r, next_v.priority, reg_quantity, r);

                    let e_r = RegisterVertex {
                        original_graph_id: edge_node,
                        priority: new_priority,
                        owner: next_v.owner,
                        register_state: new_registers,
                        rank_reset: r as Rank,
                    };

                    let new_id: VertexId = add_new_register_vertex!(e_r);

                    original_to_reg_v
                        .entry(edge_node)
                        .and_modify(|v: &mut HashSet<_>| {v.insert(new_id);})
                        .or_insert(HashSet::from_iter([new_id]));
                }
            }
        }

        Self {
            original_game: game,
            controller,
            vertices: final_graph,
            reg_v_index,
            original_to_reg_v,
            reg_quantity,
        }
    }

    /// Calculate the max `k` which assures the validity of calculated results on a `k`-register game.
    ///
    /// Note that register games of `j < k` might also be valid, but it's not guaranteed.
    /// This only assures that the register-index of the particular given `pg` is not _more_ than the result.
    pub fn max_register_index(pg: &ParityGame) -> Rank {
        // Ideally we'd use the result of https://faculty.runi.ac.il/udiboker/files/ToWeak.pdf
        // Where the max register-index would be 1 + log(z), where z is the maximal number of vertex-disjoint cycles.
        // Calculating that seems(?) to be an NP-hard problem though.
        // Just calculating the strongly connected components isn't enough, as everyone uses the (maximal) SCC definition,
        // but we don't want the maximal SCCs!
        (1 + pg.vertex_count().ilog2()) as Rank
    }

    /// Project the given register game winners back to the original game's vertices.
    pub fn project_winners_original(&self, game_winners: &[Owner]) -> Vec<Owner> {
        let mut output = vec![None; self.original_game.vertex_count()];

        for (reg_id, &winner) in game_winners.iter().enumerate() {
            let reg_v = &self.vertices[reg_id];

            let original_id = reg_v.original_graph_id;
            let current_winner = &mut output[original_id.index()];

            if let Some(curr_win) = current_winner {
                if *curr_win != winner {
                    // panic!("Game winner ({curr_win:?}) for Reg Idx `{reg_id}` did not equal existing winner ({winner:?}) for original Idx `{original_id:?}`");
                }
                *curr_win = winner;
            } else {
                *current_winner = Some(winner);
            }
        }

        output.into_iter().flatten().collect()
    }

    #[inline]
    fn edges_for_root_vertex_rg<'b>(&'b self, reg_v: &'b RegisterVertex, root_vertex: VertexId<u32>) -> impl Iterator<Item=VertexId<u32>> + 'b {
        (0..self.reg_quantity).flat_map(move |r| {
            let next_v = &self.original_game[root_vertex];
            let new_priority =
                reset_to_priority_2021(r as Rank, reg_v.register_state[r], next_v.priority, self.controller);
            let mut new_registers = reg_v.register_state.clone();

            let new_r = new_registers.make_mut();
            next_registers_2021(new_r, next_v.priority, self.reg_quantity, r);

            let e_r = RegisterVertex {
                original_graph_id: root_vertex,
                priority: new_priority,
                owner: next_v.owner,
                register_state: new_registers,
                rank_reset: r as Rank,
            };

            self.reg_v_index.get(&e_r).copied()
        })
    }
}

impl<'a> ParityGraph<u32, RegisterVertex> for ReducedRegisterGame<'a> {
    type Parent = Self;

    fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    fn edge_count(&self) -> usize {
        self.original_game.edge_count()
    }

    fn vertices_index(&self) -> impl Iterator<Item = NodeIndex<u32>> + '_ {
        (0..self.vertices.len()).map(NodeIndex::new)
    }

    fn vertices(&self) -> impl Iterator<Item = &RegisterVertex> + '_ {
        self.vertices.iter()
    }

    fn label(&self, vertex_id: NodeIndex<u32>) -> Option<&str> {
        None
    }

    fn get(&self, id: NodeIndex<u32>) -> Option<&RegisterVertex> {
        self.vertices.get(id.index())
    }

    fn predecessors(&self, id: NodeIndex<u32>) -> impl Iterator<Item = NodeIndex<u32>> + '_ {
        let reg_v = &self.vertices[id.index()];
        let original_v = &self.original_game[reg_v.original_graph_id];
        self.original_game.predecessors(reg_v.original_graph_id).flat_map(move |v| {
            let register_game_vs = self.original_to_reg_v.get(&v).unwrap();
            // let old_method = register_game_vs
            //     .iter()
            //     .filter(move |reg_v| self.edges(**reg_v).any(|reg_v_edge| reg_v_edge == id))
            //     .copied().collect_vec();
            register_game_vs.iter().filter(|v| {
                let rg = &self.vertices[v.index()];
                can_be_next_registers(rg, reg_v, original_v.priority, self.controller)
            }).copied()
        })
    }

    fn create_subgame(
        &self,
        exclude: impl IntoIterator<Item = NodeIndex<u32>>,
    ) -> SubGame<u32, RegisterVertex, Self::Parent> {
        SubGame {
            parent: self,
            ignored: HashSet::from_iter(exclude),
            _phantom: Default::default(),
        }
    }

    fn has_edge(&self, from: NodeIndex<u32>, to: NodeIndex<u32>) -> bool {
        // This can be done more efficiently without constructing the edges every time... but not needed right now.
        self.edges(from).contains(&to)
    }

    fn edges(&self, v: NodeIndex<u32>) -> impl Iterator<Item = NodeIndex<u32>> + '_ {
        self.grouped_edges(v).flatten()
    }

    fn graph_edges(&self) -> impl Iterator<Item = EdgeReference<'_, (), u32>> {
        std::iter::empty()
    }
}

pub trait RegisterParityGraph<Ix: IndexType = u32, Vert: ParityVertex + 'static = RegisterVertex>: ParityGraph<Ix, Vert> {
    /// Return all edges from a register vertex `v`, but grouped by the underlying parity game vertex id of the target vertices.
    fn grouped_edges(&self, v: VertexId<Ix>) -> impl Iterator<Item = impl Iterator<Item = VertexId<Ix>> + '_> + '_;

    /// Return all `E_i` edges from a particular `register_v`, where the underlying parity game vertex is is `root_vertex`. 
    fn edges_for_root_vertex(&self, register_v: VertexId<Ix>, root_vertex: VertexId) -> impl Iterator<Item = VertexId<Ix>> + '_;
}

impl<'a> RegisterParityGraph<u32, RegisterVertex> for ReducedRegisterGame<'a> {
    #[inline]
    fn grouped_edges(&self, v: NodeIndex<u32>) -> impl Iterator<Item = impl Iterator<Item = NodeIndex<u32>> + '_> + '_ {
        let reg_v = &self.vertices[v.index()];
        self.original_game
            .edges(reg_v.original_graph_id)
            .map(move |original| {
                self.edges_for_root_vertex_rg(reg_v, original)
            })
    }

    fn edges_for_root_vertex(&self, register_v: VertexId<u32>, root_vertex: VertexId<u32>) -> impl Iterator<Item=VertexId<u32>> + '_ {
        let reg_v = &self.vertices[register_v.index()];
        self.edges_for_root_vertex_rg(reg_v, root_vertex)
    }
}

impl<'a, Parent: RegisterParityGraph<u32, RegisterVertex>> RegisterParityGraph<u32, RegisterVertex> for SubGame<'a, u32, RegisterVertex, Parent> {
    #[inline]
    fn grouped_edges(&self, v: NodeIndex<u32>) -> impl Iterator<Item = impl Iterator<Item = NodeIndex<u32>> + '_> + '_ {
        self.parent.grouped_edges(v)
            .flat_map(|r_itr| {
                let mut itr = r_itr.filter(|rv| !self.ignored.contains(rv)).peekable();
        
                if itr.peek().is_some() {
                    Some(itr)
                } else {
                    None
                }
            })
    }

    fn edges_for_root_vertex(&self, register_v: VertexId<u32>, root_vertex: VertexId<u32>) -> impl Iterator<Item=VertexId<u32>> + '_ {
        self.parent.edges_for_root_vertex(register_v, root_vertex).filter(|rv| !self.ignored.contains(rv))
    }
}

/// Convert a reset of a particular register to a priority to be used in the register game.
#[inline]
pub fn reset_to_priority_2021(
    rank: Rank,
    saved_priority: Priority,
    vertex_priority: Priority,
    controller: Owner,
) -> Priority {
    let is_aligned = controller.priority_aligned(saved_priority.max(vertex_priority));
    let out = 2 * rank as Priority;
    match controller {
        Owner::Even => {
            if is_aligned {
                out
            } else {
                out + 1
            }
        }
        Owner::Odd => {
            if is_aligned {
                out + 1
            } else {
                out + 2
            }
        }
    }
}

/// Calculate the next set of registers for a particular set of register contents.
pub fn next_registers_2021(
    current: &mut [Priority],
    vertex_priority: Priority,
    n_registers: usize,
    reset_register: usize,
) {
    for i in 0..n_registers {
        match i.cmp(&reset_register) {
            Ordering::Less => current[i] = 0,
            Ordering::Equal => current[i] = vertex_priority,
            Ordering::Greater => current[i] = current[i].max(vertex_priority),
        }
    }
}

pub fn can_be_next_registers(
    current: &RegisterVertex,
    next: &RegisterVertex,
    vertex_priority: Priority,
    controller: Owner,
) -> bool {
    let reset = next.rank_reset as usize;

    if reset_to_priority_2021(next.rank_reset, current.register_state[reset], vertex_priority, controller) != next.priority {
        return false;
    }

    for (&p, &next_p) in current.register_state[reset + 1..].iter().zip(&next.register_state[reset + 1..]) {
        if next_p != vertex_priority && next_p > p {
            return false;
        }

        if p > next_p {
            return false;
        }
    }

    true
}

#[derive(Clone, Debug, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub struct RegisterVertex {
    pub original_graph_id: VertexId,
    pub priority: Priority,
    pub owner: Owner,
    pub register_state: EcoVec<Priority>,
    pub rank_reset: Rank,
}

impl ParityVertex for RegisterVertex {
    fn priority(&self) -> Priority {
        self.priority
    }

    fn owner(&self) -> Owner {
        self.owner
    }
}
