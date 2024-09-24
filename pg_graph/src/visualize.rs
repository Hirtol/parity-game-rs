use std::fmt::{Debug, Display, Write};

use oxidd_core::function::Function;
use oxidd_core::{Edge, HasLevel, Manager};
use oxidd_dump::dot::DotStyle;
use petgraph::graph::IndexType;

use crate::symbolic::oxidd_extensions::GeneralBooleanFunction;
use crate::{
    explicit::{
        register_game::{ChosenAction, RegisterGame},
        VertexId,
    },
    Owner
    ,
};

/// An abstraction to allow for generic writing of the underlying graphs.
///
/// See [DotWriter] and [MermaidWriter]
pub trait VisualGraph<Ix = u32> {
    fn vertices(&self) -> Box<dyn Iterator<Item = VisualVertex<Ix>> + '_>;
    fn edges(&self) -> Box<dyn Iterator<Item = (VertexId<Ix>, VertexId<Ix>)> + '_>;

    fn node_text(&self, node: VertexId<Ix>, sink: &mut dyn Write) -> std::fmt::Result;

    fn edge_text(&self, edge: (VertexId<Ix>, VertexId<Ix>), sink: &mut dyn Write) -> std::fmt::Result;
}

pub struct VisualVertex<T = u32> {
    pub id: VertexId<T>,
    pub owner: Owner,
}

pub struct DotWriter;

impl DotWriter {
    pub fn write_dot<T: Copy + IndexType>(graph: &dyn VisualGraph<T>) -> eyre::Result<String> {
        let mut output = String::from("digraph DD {\n");
        // Write defaults
        writeln!(
            &mut output,
            "node[style=filled,fillcolor=lightgrey,margin=0.05,height=0.05,width=0.05]edge[arrowsize=0.4]"
        )?;
        writeln!(&mut output, "rankdir=LR")?;

        for v in graph.vertices() {
            let shape = match v.owner {
                Owner::Even => "circle",
                Owner::Odd => "square",
            };

            write!(&mut output, "{}[label=<", v.id.index())?;
            graph.node_text(v.id, &mut output)?;
            writeln!(&mut output, ">,shape={shape}]")?;
        }

        for (v_id, edge) in graph.edges() {
            write!(&mut output, "{} -> {} [label=<", v_id.index(), edge.index())?;
            graph.edge_text((v_id, edge), &mut output)?;
            writeln!(&mut output, ">]")?;
        }

        write!(&mut output, "}}")?;

        Ok(output)
    }

    pub fn write_dot_symbolic<'a, F>(
        graph: &'a crate::symbolic::SymbolicParityGame<F>,
        additional_funcs: impl IntoIterator<Item = (&'a F, String)>,
    ) -> eyre::Result<String>
        where for<'id> F: 'a + Function + DotStyle<<<F::Manager<'id> as Manager>::Edge as Edge>::Tag>,
              for<'id> <F::Manager<'id> as Manager>::InnerNode: HasLevel,
              for<'id> <<F::Manager<'id> as Manager>::Edge as Edge>::Tag: Debug,
              for<'id> <F::Manager<'id> as Manager>::Terminal: Display, {
        use oxidd_core::ManagerRef;

        let mut out = Vec::new();

        graph
            .manager
            .with_manager_exclusive(|man| {
                let variables = graph
                    .variables
                    .iter()
                    .chain(graph.variables_edges.iter())
                    .enumerate()
                    .map(|(i, v)| {
                        (
                            v,
                            if i >= graph.variables.len() {
                                format!("x_{}", i - graph.variables.len())
                            } else {
                                format!("x{i}")
                            },
                        )
                    });

                let functions = graph
                    .priorities
                    .iter()
                    .map(|(p, bdd)| (bdd, format!("Priority {p}")))
                    .chain([
                        (&graph.vertices, "vertices".into()),
                        (&graph.vertices_even, "vertices_even".into()),
                        (&graph.vertices_odd, "vertices_odd".into()),
                        (&graph.edges, "edges".into()),
                    ])
                    .chain(additional_funcs);

                oxidd_dump::dot::dump_all(&mut out, man, variables, functions)
            })
            .expect("Failed to lock");

        Ok(String::from_utf8(out).expect("Invalid UTF-8"))
    }

    pub fn write_dot_symbolic_register<'a, F>(
        graph: &'a crate::symbolic::register_game::SymbolicRegisterGame<F>,
        additional_funcs: impl IntoIterator<Item = (&'a F, String)>,
    ) -> eyre::Result<String>
        where for<'id> F: 'a + GeneralBooleanFunction + DotStyle<<<F::Manager<'id> as Manager>::Edge as Edge>::Tag>,
              for<'id> <F::Manager<'id> as Manager>::InnerNode: HasLevel,
              for<'id> <<F::Manager<'id> as Manager>::Edge as Edge>::Tag: Debug,
              for<'id> <F::Manager<'id> as Manager>::Terminal: Display,{
        use oxidd_core::ManagerRef;

        let mut out = Vec::new();

        graph
            .manager
            .with_manager_exclusive(|man| {
                let variables = graph
                    .variables
                    .iter_names("")
                    .chain(graph.variables_edges.iter_names("_"));

                let prio = graph.priorities.iter().map(|(p, bdd)| (bdd, format!("Priority {p}")));

                let functions = [
                    (&graph.vertices, "vertices".into()),
                    (&graph.v_even, "vertices_even".into()),
                    (&graph.v_odd, "vertices_odd".into()),
                    (&graph.e_move, "E_move".into()),
                    (&graph.e_i_all, "E_i_all".into())
                ]
                .into_iter()
                .chain(prio)
                .chain(additional_funcs);

                oxidd_dump::dot::dump_all(&mut out, man, variables, functions)
            })
            .expect("Failed to lock");

        Ok(String::from_utf8(out).expect("Invalid UTF-8"))
    }

    pub fn write_dot_symbolic_register_hot<'a, F>(
        graph: &'a crate::symbolic::register_game_one_hot::OneHotRegisterGame<F>,
        additional_funcs: impl IntoIterator<Item = (&'a F, String)>,
    ) -> eyre::Result<String>
        where for<'id> F: 'a + GeneralBooleanFunction + DotStyle<<<F::Manager<'id> as Manager>::Edge as Edge>::Tag>,
              for<'id> <F::Manager<'id> as Manager>::InnerNode: HasLevel,
              for<'id> <<F::Manager<'id> as Manager>::Edge as Edge>::Tag: Debug,
              for<'id> <F::Manager<'id> as Manager>::Terminal: Display,{
        use oxidd_core::ManagerRef;

        let mut out = Vec::new();

        graph
            .manager
            .with_manager_exclusive(|man| {
                let variables = graph
                    .variables
                    .iter_names("")
                    .chain(graph.variables_edges.iter_names("_"));

                let prio = graph.priorities.iter().map(|(p, bdd)| (bdd, format!("Priority {p}")));
                let e_is = graph.e_i.iter().enumerate().map(|(i, bdd)| (bdd, format!("E_{i}")));

                let functions = [
                    (&graph.vertices, "vertices".into()),
                    (&graph.v_even, "vertices_even".into()),
                    (&graph.v_odd, "vertices_odd".into()),
                    (&graph.e_move, "E_move".into()),
                    (&graph.e_i_all, "E_i_all".into())
                ]
                    .into_iter()
                    .chain(prio)
                    .chain(e_is)
                    .chain(additional_funcs);

                oxidd_dump::dot::dump_all(&mut out, man, variables, functions)
            })
            .expect("Failed to lock");

        Ok(String::from_utf8(out).expect("Invalid UTF-8"))
    }
}

pub struct MermaidWriter;

impl MermaidWriter {
    pub fn write_mermaid<T: Copy + IndexType>(graph: &dyn VisualGraph<T>) -> eyre::Result<String> {
        let mut output = String::from("flowchart TD\n");

        for v in graph.vertices() {
            let (open_token, close_token) = match v.owner {
                Owner::Even => ("(", ")"),
                Owner::Odd => ("{{", "}}"),
            };

            write!(&mut output, "{}{}\"", v.id.index(), open_token,)?;
            graph.node_text(v.id, &mut output)?;
            writeln!(&mut output, "\"{close_token}")?;
        }

        for (v_id, edge) in graph.edges() {
            write!(&mut output, "{} -->|", v_id.index())?;
            graph.edge_text((v_id, edge), &mut output)?;
            writeln!(&mut output, "| {}", edge.index())?;
        }

        Ok(output)
    }
}

pub struct VisualRegisterGame<'a, 'b>(pub &'a RegisterGame<'b>);

impl<'a, 'b> VisualGraph for VisualRegisterGame<'a, 'b> {
    fn vertices(&self) -> Box<dyn Iterator<Item = VisualVertex<u32>> + '_> {
        self.0.vertices()
    }

    fn edges(&self) -> Box<dyn Iterator<Item = (VertexId, VertexId)> + '_> {
        self.0.edges()
    }

    fn node_text(&self, node: VertexId, sink: &mut dyn Write) -> std::fmt::Result {
        let v = &self.0.vertices[node.index()];
        write!(
            sink,
            "{priority} ({node})<br/>{orig_priority},{regs:?}",
            node = node.index(),
            priority = v.priority,
            orig_priority = self.0.original_game[v.original_graph_id].priority,
            regs = v.register_state,
        )
    }

    fn edge_text(&self, edge: (VertexId, VertexId), sink: &mut dyn Write) -> std::fmt::Result {
        match self.0.vertices[edge.0.index()].next_action {
            ChosenAction::RegisterChange => write!(sink, "E<sub>i</sub>"),
            ChosenAction::Move => write!(sink, "E<sub>move</sub>"),
        }
    }
}
