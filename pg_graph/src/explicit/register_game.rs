//! See [RegisterGame]
use std::{cmp::Ordering, collections::VecDeque, fmt::Write};

use ecow::{eco_vec, EcoVec};

use crate::{
    datatypes::Priority,
    explicit::{ParityGame, ParityGraph, VertexId},
    Owner,
    visualize::VisualVertex,
};

/// The value type for `k` in a register game.
///
/// Practically this doesn't need more than
pub type Rank = u8;

#[cfg_attr(doc, aquamarine::aquamarine)]
/// A Register Game is an expansion of the original parity game with a certain `k`-amount of registers.
///
/// An expansion works as follows, given the below parity game (Diamond == [Owner::Odd], Round == [Owner::Even]):
/// include_mmd!("../example_diagrams/rg_example.mmd")
///
/// Would be represented as the 1-Register Game:
/// include_mmd!("../example_diagrams/rg_pg_example.mmd")
///
/// Note that in the above example, whilst the original parity game has [Owner::Even] as the winner, the Register Game
/// has [Owner::Odd]. You need at least a 2-Register Game (`1 + log_2(n=2) = 2`) to ensure the correct result.
pub struct RegisterGame<'a> {
    pub original_game: &'a crate::explicit::ParityGame,
    pub vertices: Vec<RegisterVertex>,
    pub edges: ahash::HashMap<VertexId, Vec<VertexId>>,
}

enum ToExpand {
    OriginalVertex(VertexId),
    /// Contains the register vertex id, and the register state for this particular case.
    RegisterVertex(VertexId),
}

impl<'a> RegisterGame<'a> {
    /// Construct a `k`-register based parity game derived from the given `game`.
    ///
    /// The registers will be controlled by `controller`.
    ///
    /// TODO: Consider making `k` a const-generic
    #[tracing::instrument(name = "Construct Register Game", skip(game))]
    pub fn construct(game: &'a crate::explicit::ParityGame, k: Rank, controller: Owner) -> Self {
        let base_registers = game
            .priorities_unique()
            .chain([0])
            .map(|pr| (pr, eco_vec!(pr; k as usize)))
            .collect::<ahash::HashMap<_, _>>();

        let mut to_expand = game
            .vertices_index()
            .map(ToExpand::OriginalVertex)
            .collect::<VecDeque<_>>();

        let mut reg_v_index = ahash::HashMap::default();
        let mut final_graph = Vec::new();
        let mut final_edges = ahash::HashMap::default();

        // Only expand the new register vertex if it's actually unique.
        // Done as a macro as we mutably borrow `to_expand`
        macro_rules! add_new_register_vertex {
            ($reg:expr) => {{
                let reg = $reg;

                if let Some(&r_id) = reg_v_index.get(&reg) {
                    VertexId::new(r_id)
                } else {
                    final_graph.push(reg.clone());
                    let r_id = final_graph.len() - 1;
                    reg_v_index.insert(reg, r_id);
                    // Ensure this one is listed for further expansion
                    to_expand.push_back(ToExpand::RegisterVertex(VertexId::new(r_id)));
                    VertexId::new(r_id)
                }
            }};
        }

        while let Some(expanding) = to_expand.pop_back() {
            match expanding {
                // Expand the given original vertex as a starting node, aka, assuming fresh registers
                ToExpand::OriginalVertex(v_id) => {
                    let register_values = base_registers.get(&0).expect("Priority register wasn't initialised");

                    // We assume we start with the next action being a register change instead of a move,
                    // thus the owner is always the controller.
                    let register_vertex = RegisterVertex {
                        original_graph_id: v_id,
                        priority: neutral_priority(controller),
                        owner: controller,
                        register_state: register_values.clone(),
                        next_action: ChosenAction::RegisterChange,
                    };

                    let _: VertexId = add_new_register_vertex!(register_vertex);
                }
                ToExpand::RegisterVertex(reg_id) => {
                    let reg_v = &final_graph[reg_id.index()];
                    let edges = final_edges.entry(reg_id).or_insert_with(Vec::new);
                    let original_vertex = &game[reg_v.original_graph_id];

                    match reg_v.next_action {
                        ChosenAction::RegisterChange => {
                            // Now create its edges to expand (E_skip and E_r)
                            for r in 0..k {
                                let reg_v = &final_graph[reg_id.index()];
                                let new_priority = rank_to_priority(r, reg_v.register_state[r as usize], controller);
                                let mut new_registers = reg_v.register_state.clone();
                                new_registers.make_mut().rotate_right(r as usize);
                                new_registers.make_mut()[0] = 0;

                                let e_r = RegisterVertex {
                                    original_graph_id: reg_v.original_graph_id,
                                    priority: new_priority,
                                    owner: original_vertex.owner,
                                    register_state: new_registers,
                                    next_action: ChosenAction::Move,
                                };

                                edges.push(add_new_register_vertex!(e_r));
                            }

                            let reg_v = &final_graph[reg_id.index()];
                            let e_skip = RegisterVertex {
                                original_graph_id: reg_v.original_graph_id,
                                priority: neutral_priority(controller),
                                owner: original_vertex.owner,
                                register_state: reg_v.register_state.clone(),
                                next_action: ChosenAction::Move,
                            };
                            edges.push(add_new_register_vertex!(e_skip));
                        }
                        ChosenAction::Move => {
                            for edge in game.edges(reg_v.original_graph_id) {
                                let reg_v = &final_graph[reg_id.index()];
                                let next = game.get(edge).unwrap();
                                let mut r_next = RegisterVertex {
                                    original_graph_id: edge,
                                    priority: neutral_priority(controller),
                                    register_state: reg_v.register_state.clone(),
                                    next_action: ChosenAction::RegisterChange,
                                    // After a move it'll always be the controller's turn again
                                    owner: controller,
                                };

                                for reg in r_next.register_state.make_mut() {
                                    if next.priority > *reg {
                                        *reg = next.priority;
                                    }
                                }

                                edges.push(add_new_register_vertex!(r_next));
                            }
                        }
                    }
                }
            }
        }

        Self {
            original_game: game,
            vertices: final_graph,
            edges: final_edges,
        }
    }

    #[tracing::instrument(name = "Construct Register Game 2021", skip(game))]
    pub fn construct_2021(game: &'a crate::explicit::ParityGame, k: Rank, controller: Owner) -> Self {
        let reg_quantity = k as usize + 1;
        let base_registers = game
            .priorities_unique()
            .chain([0])
            .map(|pr| (pr, eco_vec!(pr; reg_quantity)))
            .collect::<ahash::HashMap<_, _>>();

        let mut to_expand = game
            .vertices_index()
            .map(ToExpand::OriginalVertex)
            .collect::<VecDeque<_>>();

        let mut reg_v_index = ahash::HashMap::default();
        let mut final_graph = Vec::new();
        let mut final_edges = ahash::HashMap::default();

        // Only expand the new register vertex if it's actually unique.
        // Done as a macro as we mutably borrow `to_expand`
        macro_rules! add_new_register_vertex {
            ($reg:expr) => {{
                let reg = $reg;

                if let Some(&r_id) = reg_v_index.get(&reg) {
                    VertexId::new(r_id)
                } else {
                    final_graph.push(reg.clone());
                    let r_id = final_graph.len() - 1;
                    reg_v_index.insert(reg, r_id);
                    // Ensure this one is listed for further expansion
                    to_expand.push_back(ToExpand::RegisterVertex(VertexId::new(r_id)));
                    VertexId::new(r_id)
                }
            }};
        }

        while let Some(expanding) = to_expand.pop_back() {
            match expanding {
                // Expand the given original vertex as a starting node, aka, assuming fresh registers
                ToExpand::OriginalVertex(v_id) => {
                    let register_values = base_registers.get(&0).expect("Priority register wasn't initialised");

                    // We assume we start with the next action being a register change instead of a move,
                    // thus the owner is always the controller.
                    let register_vertex = RegisterVertex {
                        original_graph_id: v_id,
                        priority: neutral_priority_2021(controller),
                        owner: controller,
                        register_state: register_values.clone(),
                        next_action: ChosenAction::RegisterChange,
                    };

                    let _: VertexId = add_new_register_vertex!(register_vertex);
                }
                ToExpand::RegisterVertex(reg_id) => {
                    let reg_v = &final_graph[reg_id.index()];
                    let edges = final_edges.entry(reg_id).or_insert_with(Vec::new);
                    let original_vertex = &game[reg_v.original_graph_id];

                    match reg_v.next_action {
                        ChosenAction::RegisterChange => {
                            // Now create its edges to expand (E_r)
                            for r in 0..reg_quantity {
                                let reg_v = &final_graph[reg_id.index()];
                                let new_priority = reset_to_priority_2021(
                                    r as Rank,
                                    reg_v.register_state[r],
                                    original_vertex.priority,
                                    controller,
                                );
                                let mut new_registers = reg_v.register_state.clone();

                                let new_r = new_registers.make_mut();
                                next_registers_2021(new_r, original_vertex.priority, reg_quantity, r);

                                let e_r = RegisterVertex {
                                    original_graph_id: reg_v.original_graph_id,
                                    priority: new_priority,
                                    owner: original_vertex.owner,
                                    register_state: new_registers,
                                    next_action: ChosenAction::Move,
                                };

                                edges.push(add_new_register_vertex!(e_r));
                            }
                        }
                        ChosenAction::Move => {
                            for edge_node in game.edges(reg_v.original_graph_id) {
                                let reg_v = &final_graph[reg_id.index()];
                                let r_next = RegisterVertex {
                                    original_graph_id: edge_node,
                                    priority: neutral_priority_2021(controller),
                                    register_state: reg_v.register_state.clone(),
                                    next_action: ChosenAction::RegisterChange,
                                    // After a move it'll always be the controller's turn again
                                    owner: controller,
                                };

                                edges.push(add_new_register_vertex!(r_next));
                            }
                        }
                    }
                }
            }
        }

        Self {
            original_game: game,
            vertices: final_graph,
            edges: final_edges,
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
            // Only persist the result of the 0-initialised registers, as they're the starting registers
            let reg_v = &self.vertices[reg_id];
            if reg_v.register_state.iter().any(|r| *r != 0) {
                continue;
            }

            let original_id = reg_v.original_graph_id;
            let current_winner = &mut output[original_id.index()];

            if let Some(curr_win) = current_winner {
                if *curr_win != winner {
                    panic!("Game winner ({curr_win:?}) for Reg Idx `{reg_id}` did not equal existing winner ({winner:?}) for original Idx `{original_id:?}`");
                }
                *curr_win = winner;
            } else {
                *current_winner = Some(winner);
            }
        }

        output.into_iter().flatten().collect()
    }

    #[tracing::instrument(name = "Convert to Parity Game", skip(self))]
    pub fn to_game(&self) -> eyre::Result<crate::explicit::ParityGame> {
        let mut parsed_game = vec![];

        for (v_id, v, edges) in self
            .vertices
            .iter()
            .enumerate()
            .flat_map(|(v_id, v)| self.edges.get(&VertexId::new(v_id)).map(|edg| (v_id, v, edg)))
        {
            parsed_game.push(pg_parser::Vertex {
                id: v_id,
                priority: v.priority as usize,
                owner: v.owner as u8,
                outgoing_edges: edges.iter().copied().map(VertexId::index).collect(),
                label: self.original_game.label(VertexId::from(v.original_graph_id)),
            });
        }

        crate::explicit::ParityGame::new(parsed_game)
    }
}

impl<'a> crate::visualize::VisualGraph for RegisterGame<'a> {
    fn vertices(&self) -> Box<dyn Iterator<Item = VisualVertex> + '_> {
        Box::new(self.vertices.iter().enumerate().map(|(i, v)| VisualVertex {
            id: VertexId::new(i),
            owner: v.owner,
        }))
    }

    fn edges(&self) -> Box<dyn Iterator<Item = (VertexId, VertexId)> + '_> {
        Box::new(
            self.edges
                .iter()
                .flat_map(|(v, targets)| targets.iter().map(|u| (*v, *u))),
        )
    }

    fn node_text(&self, node: VertexId, sink: &mut dyn Write) -> std::fmt::Result {
        let v = &self.vertices[node.index()];
        write!(
            sink,
            "{priority},{orig}({orig_label},{orig_priority}),{regs:?},{next_move:?}",
            priority = v.priority,
            orig = v.original_graph_id.index(),
            orig_label = self.original_game.label(v.original_graph_id).unwrap_or_default(),
            orig_priority = self.original_game[v.original_graph_id].priority,
            regs = v.register_state,
            next_move = v.next_action,
        )
    }

    fn edge_text(&self, edge: (VertexId, VertexId), sink: &mut dyn Write) -> std::fmt::Result {
        match self.vertices[edge.0.index()].next_action {
            ChosenAction::RegisterChange => write!(sink, "E_i"),
            ChosenAction::Move => write!(sink, "E_move"),
        }
    }
}

/// Return the neutral priority for the given owner, used for constructing `E_move` and `E_skip` priorities.
fn neutral_priority(owner: Owner) -> Priority {
    match owner {
        Owner::Even => 1,
        Owner::Odd => 0,
    }
}

fn neutral_priority_2021(_owner: Owner) -> Priority {
    0
}

/// Convert a reset of a particular rank to a priority to be used in the register game.
fn rank_to_priority(rank: Rank, saved_priority: Priority, controller: Owner) -> Priority {
    // Rank is 0-indexed for us, but 1-indexed in the original paper
    let out = 2 * (rank + 1) as Priority;

    match controller {
        Owner::Even => {
            if controller.priority_aligned(saved_priority) {
                out
            } else {
                out + 1
            }
        }
        Owner::Odd => {
            if controller.priority_aligned(saved_priority) {
                out - 1
            } else {
                out
            }
        }
    }
}

/// Convert a reset of a particular register to a priority to be used in the register game.
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

#[derive(Clone, Debug, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub struct RegisterVertex {
    pub original_graph_id: VertexId,
    pub priority: Priority,
    pub owner: Owner,
    pub register_state: EcoVec<Priority>,
    pub next_action: ChosenAction,
}

#[derive(Clone, Copy, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum ChosenAction {
    RegisterChange = 0,
    Move = 1,
}

impl ChosenAction {
    pub fn from_num(val: u32) -> ChosenAction {
        match val {
            0 => ChosenAction::RegisterChange,
            1 => ChosenAction::Move,
            _ => panic!("Invalid"),
        }
    }
}
