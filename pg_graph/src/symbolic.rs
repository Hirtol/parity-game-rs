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
use petgraph::prelude::EdgeRef;

use crate::{Owner, ParityGame, ParityGraph, Priority, VertexId};

type BDD = BDDFunction;

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

    fn from_explicit_impl(explicit: &ParityGame) -> oxidd::util::AllocResult<Self> {
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
    pub fn vertex_count(&self) -> usize {
        self.manager.with_manager_shared(|man| man.num_inner_nodes())
    }
    
    pub fn encode_vertex(&self, v_idx: VertexId) -> AllocResult<BDDFunction> {
        CachedSymbolicEncoder::encode_impl(&self.variables, v_idx.index())
    }

    pub fn encode_edge_vertex(&self, v_idx: VertexId) -> AllocResult<BDDFunction> {
        CachedSymbolicEncoder::encode_impl(&self.variables_edges, v_idx.index())
    }

    pub fn to_dot(&self) -> String {
        self.to_dot_extra([])
    }
    
    pub fn to_dot_extra<'a>(&'a self, additional_funcs: impl IntoIterator<Item=(&'a BDDFunction, String)>) -> String {
        let mut out = Vec::new();

        self.manager
            .with_manager_exclusive(|man| {
                let variables = self
                    .variables
                    .iter()
                    .chain(self.variables_edges.iter())
                    .enumerate()
                    .map(|(i, v)| {
                        (
                            v,
                            if i >= self.variables.len() {
                                format!("x_{}", i - self.variables.len())
                            } else {
                                format!("x{i}")
                            },
                        )
                    });
                let functions = self
                    .priorities
                    .iter()
                    .map(|(p, bdd)| (bdd, format!("Priority {p}")))
                    .chain([
                        (&self.vertices, "vertices".into()),
                        (&self.vertices_even, "vertices_even".into()),
                        (&self.vertices_odd, "vertices_odd".into()),
                        (&self.edges, "edges".into()),
                    ])
                    .chain(additional_funcs);

                oxidd_dump::dot::dump_all(&mut out, man, variables, functions)
            })
            .expect("Failed to lock");

        String::from_utf8(out).expect("Invalid UTF-8")
    }
}

pub struct CachedSymbolicEncoder<T> {
    cache: ahash::HashMap<T, BDDFunction>,
    variables: Vec<BDDFunction>,
}

impl<T> CachedSymbolicEncoder<T>
where
    T: std::ops::BitAnd + std::ops::Shl<Output = T> + Copy + From<u8>,
    T: Eq + Hash,
    <T as std::ops::BitAnd>::Output: PartialEq<T>,
{
    pub fn new(variables: Vec<BDDFunction>) -> Self {
        Self {
            cache: ahash::HashMap::default(),
            variables,
        }
    }

    /// Encode the given value as a [BDDFunction], caching it for future use.
    ///
    /// If `value` was already provided once the previously created [BDDFunction] will be returned.
    pub fn encode(&mut self, value: T) -> AllocResult<&BDDFunction> {
        let out = match self.cache.entry(value) {
            Entry::Occupied(val) => val.into_mut(),
            Entry::Vacant(val) => val.insert(Self::encode_impl(&self.variables, value)?),
        };

        Ok(out)
    }

    fn encode_impl(variables: &[BDDFunction], value: T) -> AllocResult<BDDFunction> {
        let mut expr = if value & 1.into() != 0.into() {
            variables[0].clone()
        } else {
            variables[0].not()?
        };

        for i in 1..variables.len() {
            // Check if bit is set
            if value & (T::from(1) << T::from(i as u8)) != 0.into() {
                expr = expr.and(&variables[i])?;
            } else {
                expr = expr.and(&variables[i].not()?)?;
            }
        }

        Ok(expr)
    }
}

pub trait BddExtensions {
    /// Conceptually substitute the given `vars` in `self` for `replace_with`.
    ///
    /// `vars` and `replace_with` can be conjunctions of variables (e.g., `x1 ^ x2 ^ x3 ^...`) to substitute multiple in one go.
    ///
    /// In practice this (inefficiently) creates a new BDD with: `exists vars. (replace_with <=> vars) && self`.
    fn substitute(&self, vars: &BDDFunction, replace_with: &BDDFunction) -> AllocResult<BDDFunction>;
}

impl BddExtensions for BDDFunction {
    fn substitute(&self, vars: &BDDFunction, replace_with: &BDDFunction) -> AllocResult<BDDFunction> {
        vars.with_manager_shared(|manager, vars| {
            let iff = Self::equiv_edge(manager, vars, replace_with.as_edge(manager))?;
            let and = Self::and_edge(manager, self.as_edge(manager), &iff)?;
            let exists = Self::exist_edge(manager, &and, vars)?;

            manager.drop_edge(iff);
            manager.drop_edge(and);

            Ok(Self::from_edge(manager, exists))
        })
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use oxidd::bdd::BDDFunction;
    use oxidd_core::{function::BooleanFunction, ManagerRef};

    use crate::{ParityGame, ParityGraph, symbolic::BddExtensions, tests::example_dir};
    use crate::symbolic::SymbolicParityGame;

    pub fn load_pg(game: &str) -> ParityGame {
        let path = example_dir().join(game);
        let data = std::fs::read_to_string(&path).unwrap();
        let graph = pg_parser::parse_pg(&mut data.as_str()).unwrap();
        let parity_game = crate::ParityGame::new(graph).unwrap();
        parity_game
    }
    
    fn small_pg() -> eyre::Result<ParityGame> {
        let mut pg = r#"parity 3;
0 1 1 0,1 "0";
1 1 0 2 "1";
2 2 0 2 "2";"#;
        let pg = pg_parser::parse_pg(&mut pg).unwrap();
        ParityGame::new(pg)
    }
    
    fn symbolic_pg() -> eyre::Result<SymbolicTest> {
        let pg = small_pg()?;
        let s_pg = super::SymbolicParityGame::from_explicit(&pg)?;
        let nodes = pg.vertices_index().flat_map(|idx| s_pg.encode_vertex(idx)).collect_vec();
        let edge_nodes = pg.vertices_index().flat_map(|idx| s_pg.encode_edge_vertex(idx)).collect_vec();
        
        Ok(SymbolicTest {
            s_pg,
            vertices: nodes,
            edge_vertices: edge_nodes,
        })
    }
    
    struct SymbolicTest {
        s_pg: SymbolicParityGame,
        vertices: Vec<BDDFunction>,
        edge_vertices: Vec<BDDFunction>,
    }

    #[test]
    pub fn test_symbolic() -> eyre::Result<()> {
        let s = symbolic_pg()?;
        
        // Ensure that non-existent vertices are not represented
        let v_3 = s.s_pg.encode_vertex(3.into()).unwrap();
        assert!(!s.s_pg.vertices.and(&v_3).unwrap().satisfiable());
        
        // Ensure edges exist between nodes that we'd expect.
        let edge_relation = s.vertices[0].and(&s.edge_vertices[1]).unwrap();
        assert!(s.s_pg.edges.and(&edge_relation).unwrap().satisfiable());
        // And others don't exist
        let edge_relation = s.vertices[0].and(&s.edge_vertices[2]).unwrap();
        assert!(!s.s_pg.edges.and(&edge_relation).unwrap().satisfiable());
        
        Ok(())
    }

    #[test]
    pub fn test_symbolic_substitution() -> eyre::Result<()> {
        let s = symbolic_pg()?;
        let s_pg = &s.s_pg;

        let mut true_base = s_pg.manager.with_manager_exclusive(|man| BDDFunction::t(man));
        let start_var = s_pg
            .variables
            .iter()
            .fold(true_base.clone(), |acc, var| acc.and(var).unwrap());
        let other_vars = s_pg
            .variables_edges
            .iter()
            .fold(true_base, |acc, var| acc.and(var).unwrap());
        let final_subs_bdd = s_pg.vertices.substitute(&start_var, &other_vars).unwrap();
        
        // Ensure that the variables were substituted correctly
        let edge_nodes_exist = &s.edge_vertices[0];
        for edge_vertex in s.edge_vertices {
            assert!(final_subs_bdd.and(&edge_vertex).unwrap().satisfiable());
        }
        
        std::fs::write("out_sym.dot", s_pg.to_dot_extra([(&final_subs_bdd, "substitution".into())]))?;
        
        Ok(())
    }
}
