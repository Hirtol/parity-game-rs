use std::collections::HashMap;

use itertools::Itertools;
use oxidd::{bdd::BDDFunction, BooleanFunction, BooleanFunctionQuant, ManagerRef};
use oxidd::bdd::BDDManagerRef;
use petgraph::prelude::EdgeRef;

use crate::{Owner, ParityGame, Priority};

type BDD = BDDFunction;

pub struct SymbolicParityGame {
    manager: BDDManagerRef,
    variables: Vec<BDD>,
    variables_edges: Vec<BDD>,
    vertices: BDD,
    vertices_even: BDD,
    vertices_odd: BDD,
    edges: BDD,
    priorities: HashMap<Priority, BDD>,
}

impl SymbolicParityGame {
    #[tracing::instrument(name = "Build Symbolic Parity Game")]
    /// For now, translated from [https://github.com/olijzenga/bdd-parity-game-solver/blob/master/src/pg.py].
    pub fn from_explicit(explicit: &ParityGame) -> eyre::Result<Self> {
        Self::from_explicit_impl(explicit).map_err(|e| eyre::eyre!("Could not construct BDD due to: {e:?}"))
    }

    fn from_explicit_impl(explicit: &ParityGame) -> oxidd::util::AllocResult<Self> {
        // Need: BDD for set V, BDDs for sets V_even and V_odd, BDD^p for every p in P of the PG.
        // Alternatively, for BDD^p we could use a multi-terminal BDD (according to https://arxiv.org/pdf/2009.10876.pdf), and oxidd provides it!
        // Lastly need a duplication of the variable set (which is distinct!) to then create a BDD representing the edge relation E.

        let n_variables = (explicit.vertex_count() as f64).log2().ceil() as usize;
        let manager = oxidd::bdd::new_manager(1024, 1024, 1);
        let variables = manager
            .with_manager_exclusive(|man| (0..n_variables).flat_map(|x| BDDFunction::new_var(man)).collect_vec());
        let edge_variables = manager
            .with_manager_exclusive(|man| (0..n_variables).flat_map(|x| BDDFunction::new_var(man)).collect_vec());

        let v_to_expr = |v_idx: u32, successor: bool| {
            let variable_set = if successor { &edge_variables } else { &variables };
            // Could also use BDDFunction::t(), but more effort, just check the first bit
            let mut expr = if v_idx & 1 != 0 {
                variable_set[0].clone()
            } else {
                variable_set[0].not()?
            };

            for i in 1..n_variables {
                // Check if bit is set
                if v_idx & (1 << i) != 0 {
                    expr = expr.and(&variable_set[i])?;
                } else {
                    expr = expr.and(&variable_set[i].not()?)?;
                }
            }

            Ok::<_, oxidd::util::OutOfMemory>(expr)
        };

        // Contains all vertices in the graph
        let mut s_vertices = manager.with_manager_exclusive(|man| oxidd::bdd::BDDFunction::t(man));
        // Declare Vertices, exclude ones which are not part of the statespace
        if ((explicit.vertex_count() as f64).log2() - n_variables as f64) < f64::EPSILON {
            tracing::trace!(
                excluded = (explicit.vertex_count() as f64).log2() - n_variables as f64,
                "Excluding (inefficiently?) vertices"
            );
            for v_idx in explicit.vertex_count()..2_usize.pow(n_variables as u32) {
                s_vertices = s_vertices.and(&v_to_expr(v_idx as u32, false)?.not_owned()?)?;
            }
        }

        // Edges
        let mut s_edges = manager.with_manager_exclusive(|man| oxidd::bdd::BDDFunction::f(man));
        for edge in explicit.graph_edges() {
            let s_edge =
                v_to_expr(edge.source().index() as u32, false)?.and(&v_to_expr(edge.target().index() as u32, true)?)?;
            s_edges = s_edges.or(&s_edge)?;
        }

        // Priorities
        let mut s_priorities = explicit
            .priorities_unique()
            .map(|p| (p, manager.with_manager_exclusive(|man| oxidd::bdd::BDDFunction::f(man))))
            .collect::<HashMap<_, _>>();
        let mut s_even = manager.with_manager_exclusive(|man| oxidd::bdd::BDDFunction::f(man));
        let mut s_odd = manager.with_manager_exclusive(|man| oxidd::bdd::BDDFunction::f(man));

        for v_idx in explicit.vertices_index() {
            let vertex = explicit.get(v_idx).expect("Impossible");
            let expr = v_to_expr(v_idx.index() as u32, false)?;
            let priority = vertex.priority;

            // Fill priority
            s_priorities
                .entry(priority)
                .and_modify(|bdd| *bdd = bdd.or(&expr).expect("Out of memory"));

            // Set owners
            match vertex.owner {
                Owner::Even => s_even = s_even.or(&expr)?,
                Owner::Odd => s_odd = s_odd.or(&expr)?,
            }
        }

        Ok(Self {
            manager,
            variables,
            variables_edges: edge_variables,
            vertices: s_vertices,
            vertices_even: s_even,
            vertices_odd: s_odd,
            edges: s_edges,
            priorities: s_priorities,
        })
    }
}


#[cfg(test)]
mod tests {
    use oxidd::ManagerRef;
    use crate::{ParityGame};

    #[test]
    pub fn test_symbolic() -> eyre::Result<()> {
        let mut pg = r#"parity 3;
0 1 1 0,1 "0";
1 1 0 2 "1";
2 2 0 2 "2";"#;
        let pg = pg_parser::parse_pg(&mut pg).unwrap();
        let pg = ParityGame::new(pg)?;
        let s_pg = super::SymbolicParityGame::from_explicit(&pg)?;
        let mut out = std::fs::OpenOptions::new().write(true).create(true).truncate(true).open("out.dot")?;
        s_pg.manager.with_manager_exclusive(move |man| {
            let variables = s_pg.variables.iter().chain(s_pg.variables_edges.iter()).enumerate().map(|(i, v)| (v, format!("x{i}")));
            oxidd_dump::dot::dump_all(out, man, variables, [
                (&s_pg.vertices, "vertices"),
                (&s_pg.vertices_even, "vertices_even"),
                (&s_pg.vertices_odd, "vertices_odd"),
                (&s_pg.edges, "edges"),
            ])
        })?;

        Ok(())
    }
}