use std::{
    collections::{hash_map::Entry, HashMap},
    hash::Hash,
};

use itertools::Itertools;
use oxidd::{
    bdd::{BDDFunction, BDDManagerRef},
    BooleanFunction, Manager, ManagerRef,
};
use oxidd_core::{
    function::{BooleanFunctionQuant, Function},
    util::AllocResult,
};
use oxidd_core::util::OptBool;
use petgraph::prelude::EdgeRef;

use helpers::CachedSymbolicEncoder;

use crate::{symbolic::helpers::BddExtensions, Owner, ParityGame, ParityGraph, Priority, VertexId};

pub mod helpers;

type BDD = BDDFunction;
type Result<T> = std::result::Result<T, helpers::BddError>;

pub struct SymbolicParityGame {
    pub manager: BDDManagerRef,
    pub variables: Vec<BDD>,
    pub variables_edges: Vec<BDD>,
    pub vertices: BDD,
    pub vertices_even: BDD,
    pub vertices_odd: BDD,
    pub edges: BDD,
    pub priorities: HashMap<Priority, BDD>,
}

impl SymbolicParityGame {
    #[tracing::instrument(name = "Build Symbolic Parity Game", skip_all)]
    /// For now, translated from [https://github.com/olijzenga/bdd-parity-game-solver/blob/master/src/pg.py].
    pub fn from_explicit(explicit: &ParityGame) -> eyre::Result<Self> {
        Self::from_explicit_impl(explicit).map_err(|e| eyre::eyre!("Could not construct BDD due to: {e:?}"))
    }

    fn from_explicit_impl(explicit: &ParityGame) -> Result<Self> {
        // Need: BDD for set V, BDDs for sets V_even and V_odd, BDD^p for every p in P of the PG.
        // Alternatively, for BDD^p we could use a multi-terminal BDD (according to https://arxiv.org/pdf/2009.10876.pdf), and oxidd provides it!
        // Lastly need a duplication of the variable set (which is distinct!) to then create a BDD representing the edge relation E.

        let n_variables = (explicit.vertex_count() as f64).log2().ceil() as usize;
        let manager = oxidd::bdd::new_manager(explicit.vertex_count(), explicit.vertex_count(), 12);

        // Construct base building blocks for the BDD
        let base_true = manager.with_manager_exclusive(|man| BDDFunction::t(man));
        let base_false = manager.with_manager_exclusive(|man| BDDFunction::f(man));
        let variables = manager
            .with_manager_exclusive(|man| (0..n_variables).flat_map(|_| BDDFunction::new_var(man)).collect_vec());
        let edge_variables = manager
            .with_manager_exclusive(|man| (0..n_variables).flat_map(|_| BDDFunction::new_var(man)).collect_vec());

        let mut var_encoder = CachedSymbolicEncoder::new(variables.clone());
        let mut e_var_encoder = CachedSymbolicEncoder::new(edge_variables.clone());

        tracing::trace!("Starting vertex BDD construction");
        // Contains all vertices in the graph
        let mut s_vertices = base_true.clone();

        // Declare Vertices, exclude ones which are not part of the statespace
        if ((explicit.vertex_count() as f64).log2() - n_variables as f64) < f64::EPSILON {
            // We either exclude all items explicitly (inefficient if we're really close to the next power of two),
            // or explicitly include all valid vertices with their last bit == 1, while blacklisting everything else
            let items_to_exclude = 2_usize.pow(n_variables as u32) - explicit.vertex_count();
            let include_threshold = 2_usize.pow((n_variables - 1) as u32) / 2;

            if items_to_exclude > include_threshold {
                let start = 2_usize.pow((n_variables - 1) as u32);

                tracing::trace!(
                    n_included = explicit.vertex_count() - start,
                    items_to_exclude,
                    include_threshold,
                    "Explicitly including additional vertices"
                );
                let mut blacklist = variables.last().expect("Impossible").not()?;

                for v_idx in start..explicit.vertex_count() {
                    blacklist = blacklist.or(var_encoder.encode(v_idx)?)?;
                }

                s_vertices = s_vertices.and(&blacklist)?;
            } else {
                tracing::trace!(n_excluded = items_to_exclude, "Explicitly excluding vertices");
                for v_idx in explicit.vertex_count()..2_usize.pow(n_variables as u32) {
                    s_vertices = s_vertices.and(&var_encoder.encode(v_idx)?.not()?)?;
                }
            }
        }

        tracing::trace!("Starting edge BDD construction");
        // Edges
        let mut s_edges = base_false.clone();
        for edge in explicit.graph_edges() {
            let (source, target) = (
                var_encoder.encode(edge.source().index())?,
                e_var_encoder.encode(edge.target().index())?,
            );
            s_edges = s_edges.or(&source.and(target)?)?;
        }

        // Priorities
        let mut s_priorities = explicit
            .priorities_unique()
            .map(|p| (p, base_false.clone()))
            .collect::<HashMap<_, _>>();
        let mut s_even = base_false.clone();
        let mut s_odd = base_false;

        tracing::trace!("Starting priority/owner BDD construction");
        for v_idx in explicit.vertices_index() {
            let vertex = explicit.get(v_idx).expect("Impossible");
            let expr = var_encoder.encode(v_idx.index())?;
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

    /// Remove unreferenced nodes.
    pub fn gc(&self) -> usize {
        self.manager.with_manager_exclusive(|man| man.gc())
    }

    /// Get the amount of BDD nodes.
    pub fn bdd_node_count(&self) -> usize {
        self.manager.with_manager_shared(|man| man.num_inner_nodes())
    }
    
    /// Count of all variables used in the BDD.
    pub fn bdd_variable_count(&self) -> usize {
        self.variables.len() + self.variables_edges.len()
    }

    pub fn encode_vertex(&self, v_idx: VertexId) -> Result<BDDFunction> {
        CachedSymbolicEncoder::encode_impl(&self.variables, v_idx.index())
    }

    pub fn encode_edge_vertex(&self, v_idx: VertexId) -> Result<BDDFunction> {
        CachedSymbolicEncoder::encode_impl(&self.variables_edges, v_idx.index())
    }

    /// Calculate the predecessors of the set of vertices `of`.
    pub fn predecessors(&self, of: &BDDFunction) -> Result<BDDFunction> {
        let of_subs = of.bulk_substitute(&self.variables, &self.variables_edges)?;

        Ok(self.edges.and(&of_subs)?)
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use oxidd::bdd::BDDFunction;
    use oxidd_core::{function::BooleanFunction, util::SatCountCache};
    use oxidd_core::util::OptBool;

    use crate::{
        symbolic::{helpers::BddExtensions, SymbolicParityGame},
        visualize::DotWriter,
        ParityGame, ParityGraph,
    };

    fn small_pg() -> eyre::Result<ParityGame> {
        let mut pg = r#"parity 3;
0 1 1 0,1 "0";
1 1 0 2 "1";
2 2 0 2 "2";"#;
        let pg = pg_parser::parse_pg(&mut pg).unwrap();
        ParityGame::new(pg)
    }

    #[test]
    pub fn test_symbolic() -> eyre::Result<()> {
        let s = symbolic_pg()?;

        // Ensure that non-existent vertices are not represented
        let v_3 = s.pg.encode_vertex(3.into()).unwrap();
        assert!(!s.pg.vertices.and(&v_3).unwrap().satisfiable());

        // Ensure edges exist between nodes that we'd expect.
        let edge_relation = s.vertices[0].and(&s.edge_vertices[1]).unwrap();
        assert!(s.pg.edges.and(&edge_relation).unwrap().satisfiable());
        // And others don't exist
        let edge_relation = s.vertices[0].and(&s.edge_vertices[2]).unwrap();
        assert!(!s.pg.edges.and(&edge_relation).unwrap().satisfiable());

        Ok(())
    }

    #[test]
    pub fn test_predecessors() -> eyre::Result<()> {
        let s = symbolic_pg()?;
        
        let predecessor = s.pg.predecessors(&s.vertices[2])?;

        let count = predecessor.sat_quick_count(s.pg.bdd_variable_count() as u32);
        assert_eq!(count, 2);
        
        let valuations = [true, false].into_iter().flat_map(|v| predecessor.pick_cube(None, |_, _| v)).collect_vec();
        
        assert_eq!(valuations, vec![
            vec![OptBool::True, OptBool::False, OptBool::False, OptBool::True],
            vec![OptBool::False, OptBool::True, OptBool::False, OptBool::True]
        ]);

        Ok(())
    }

    #[test]
    pub fn test_symbolic_substitution() -> eyre::Result<()> {
        let s = symbolic_pg()?;
        let s_pg = &s.pg;

        let final_subs_bdd = s_pg.vertices.bulk_substitute(&s.pg.variables, &s.pg.variables_edges).unwrap();

        // Ensure that the variables were substituted correctly
        let edge_nodes_exist = &s.edge_vertices[0];
        for edge_vertex in s.edge_vertices {
            assert!(final_subs_bdd.and(&edge_vertex).unwrap().satisfiable());
        }

        s.pg.gc();
        let dot = DotWriter::write_dot_symbolic(
            &s_pg,
            [
                (&final_subs_bdd, "substitution".into()),
            ],
        )?;
        std::fs::write("out_sym.dot", dot)?;

        Ok(())
    }

    struct SymbolicTest {
        pg: SymbolicParityGame,
        vertices: Vec<BDDFunction>,
        edge_vertices: Vec<BDDFunction>,
    }

    fn symbolic_pg() -> eyre::Result<SymbolicTest> {
        let pg = small_pg()?;
        let s_pg = super::SymbolicParityGame::from_explicit(&pg)?;
        let nodes = pg
            .vertices_index()
            .flat_map(|idx| s_pg.encode_vertex(idx))
            .collect_vec();
        let edge_nodes = pg
            .vertices_index()
            .flat_map(|idx| s_pg.encode_edge_vertex(idx))
            .collect_vec();

        Ok(SymbolicTest {
            pg: s_pg,
            vertices: nodes,
            edge_vertices: edge_nodes,
        })
    }
}
