use std::collections::VecDeque;

use oxidd::bdd::BDDFunction;
use oxidd_core::{
    Edge,
    function::{EdgeOfFunc, Function},
    HasLevel, InnerNode, Manager, Node, util::{Borrowed, OptBool},
};
use oxidd_rules_bdd::simple::BDDTerminal;

use crate::explicit::VertexId;

#[inline]
#[must_use]
fn collect_children<E: Edge, N: InnerNode<E>>(node: &N) -> (Borrowed<E>, Borrowed<E>) {
    let mut it = node.children();
    let f_then = it.next().unwrap();
    let f_else = it.next().unwrap();
    (f_then, f_else)
}

pub struct MidAssignment<'a, F: Function> {
    next_edge: EdgeOfFunc<'a, F>,
    valuation_progress: Vec<OptBool>,
}

pub struct TruthAssignmentsIterator<'b, 'a, F: Function> {
    manager: &'b F::Manager<'a>,
    queue: VecDeque<MidAssignment<'a, F>>,
    counter: usize,
}

impl<'b, 'a> TruthAssignmentsIterator<'b, 'a, BDDFunction> {
    pub fn new(manager: &'b <BDDFunction as Function>::Manager<'a>, bdd: &BDDFunction) -> Self {
        let base_edge = bdd.as_edge(manager);
        let begin = match manager.get_node(base_edge) {
            Node::Inner(_) => Some(vec![OptBool::None; manager.num_levels() as usize]),
            Node::Terminal(t) => match t {
                BDDTerminal::False => None,
                BDDTerminal::True => Some(vec![OptBool::None; manager.num_levels() as usize]),
            },
        }
        .map(|start_valuation| MidAssignment {
            next_edge: manager.clone_edge(base_edge),
            valuation_progress: start_valuation,
        });

        Self {
            manager,
            queue: begin.into_iter().collect(),
            counter: 0,
        }
    }
}

impl<'b, 'a> Iterator for TruthAssignmentsIterator<'b, 'a, BDDFunction> {
    type Item = Vec<OptBool>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut next_evaluate = self.queue.pop_back()?;

        let mut next_cube = Some(next_evaluate.next_edge.borrowed());

        while let Some(next) = &next_cube {
            let Node::Inner(node) = self.manager.get_node(next) else {
                break;
            };

            let (true_edge, false_edge) = collect_children(node);
            let next_edge = if self.manager.get_node(&true_edge).is_terminal(&BDDTerminal::False) {
                false
            } else if self.manager.get_node(&false_edge).is_terminal(&BDDTerminal::False) {
                true
            } else {
                // First queue up the next item (`true`)
                let mut queued_val = MidAssignment {
                    next_edge: self.manager.clone_edge(&true_edge),
                    valuation_progress: next_evaluate.valuation_progress.clone(),
                };
                queued_val.valuation_progress[node.level() as usize] = OptBool::True;
                self.queue.push_back(queued_val);

                // Always take the `false` edge first
                false
            };

            next_evaluate.valuation_progress[node.level() as usize] = OptBool::from(next_edge);
            next_cube = Some(if next_edge { true_edge } else { false_edge })
        }

        self.manager.drop_edge(next_evaluate.next_edge);

        Some(next_evaluate.valuation_progress)
    }
}

/// Turn a sequence of booleans into a sequence of `VertexId`s based on a binary encoding.
///
/// # Arguments
/// * `values` - The binary encoding
/// * `first` - The amount of `bools` to take from each individual `value in values`, this assumes that all vertex
/// variables are first.
pub fn decode_assignments<'a>(values: impl IntoIterator<Item = impl AsRef<[OptBool]>>, first: usize) -> Vec<VertexId> {
    let mut output = Vec::new();

    fn inner(current_value: u32, idx: usize, assignments: &[OptBool], out: &mut Vec<VertexId>) {
        let Some(next) = assignments.get(idx) else {
            out.push(current_value.into());
            return;
        };

        match next {
            OptBool::None => {
                // True
                inner(current_value | (1 << idx), idx + 1, assignments, out);
                // False
                inner(current_value, idx + 1, assignments, out);
            }
            OptBool::False => {
                inner(current_value, idx + 1, assignments, out);
            }
            OptBool::True => {
                inner(current_value | (1 << idx), idx + 1, assignments, out);
            }
        }
    }

    for vertex_assigment in values {
        inner(0, 0, &vertex_assigment.as_ref()[0..first], &mut output)
    }

    output
}
