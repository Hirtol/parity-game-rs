use std::fmt::Write;
use crate::{Owner, VertexId};
use crate::register_game::{ChosenAction, RegisterGame};

/// An abstraction to allow for generic writing of the underlying graphs.
/// 
/// See [DotWriter] and [MermaidWriter]
pub trait VisualGraph {
    fn vertices(&self) -> Box<dyn Iterator<Item=VisualVertex> + '_>;
    fn edges(&self) -> Box<dyn Iterator<Item=(VertexId, VertexId)> + '_>;
    
    fn node_text(&self, node: VertexId, sink: &mut dyn Write) -> std::fmt::Result;
    
    fn edge_text(&self, edge: (VertexId, VertexId), sink: &mut dyn Write) -> std::fmt::Result;
}

pub struct VisualVertex {
    pub id: VertexId,
    pub owner: Owner,
}

pub struct DotWriter;

impl DotWriter {
    pub fn write_dot(graph: &dyn VisualGraph) -> eyre::Result<String> {
        let mut output = String::from("digraph DD {\n");
        // Write defaults
        writeln!(&mut output, "node[style=filled,fillcolor=lightgrey,margin=0.05,height=0.05,width=0.05]edge[arrowsize=0.4]")?;
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

        writeln!(&mut output, "}}")?;

        Ok(output)
    }
}

pub struct MermaidWriter;

impl MermaidWriter {
    pub fn write_mermaid(graph: &dyn VisualGraph) -> eyre::Result<String> {
        let mut output = String::from("flowchart TD\n");

        for v in graph.vertices() {
            let (open_token, close_token) = match v.owner {
                Owner::Even => ("(", ")"),
                Owner::Odd => ("{{", "}}"),
            };

            write!(&mut output, "{}{}\"", v.id.index(), open_token, )?;
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
    fn vertices(&self) -> Box<dyn Iterator<Item=VisualVertex> + '_> {
        self.0.vertices()
    }

    fn edges(&self) -> Box<dyn Iterator<Item=(VertexId, VertexId)> + '_> {
        self.0.edges()
    }

    fn node_text(&self, node: VertexId, sink: &mut dyn Write) -> std::fmt::Result {
        let v = &self.0.vertices[node.index()];
        write!(sink, "{priority}<br/>{orig_priority},{regs:?}",
               priority = v.priority,
               orig_priority = self.0.original_game[v.original_graph_id].priority,
               regs = v.register_state,)
    }

    fn edge_text(&self, edge: (VertexId, VertexId), sink: &mut dyn Write) -> std::fmt::Result {
        match self.0.vertices[edge.0.index()].next_action {
            ChosenAction::RegisterChange => write!(sink, "E<sub>i</sub>"),
            ChosenAction::Move => write!(sink, "E<sub>move</sub>")
        }
    }
}