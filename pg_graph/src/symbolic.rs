use std::collections::{hash_map::Entry, HashMap};

use itertools::Itertools;
use oxidd::{
    bdd::{BDDFunction, BDDManagerRef},
    BooleanFunction, Manager, ManagerRef,
};
use oxidd_core::function::{BooleanFunctionQuant, Function};
use oxidd_core::util::AllocResult;
use petgraph::prelude::EdgeRef;

use crate::{Owner, ParityGame, ParityGraph, Priority};

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
        let variables = manager
            .with_manager_exclusive(|man| (0..n_variables).flat_map(|x| BDDFunction::new_var(man)).collect_vec());
        let edge_variables = manager
            .with_manager_exclusive(|man| (0..n_variables).flat_map(|x| BDDFunction::new_var(man)).collect_vec());

        let mut v_bdd_cache: HashMap<u32, BDDFunction, ahash::RandomState> = HashMap::default();
        let mut v_bdd_cache_2: HashMap<u32, BDDFunction, ahash::RandomState> = HashMap::default();

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
        macro_rules! v_to_expr_cached {
            ($v_idx:expr, false) => {
                match v_bdd_cache.entry($v_idx) {
                    Entry::Occupied(val) => val.into_mut(),
                    Entry::Vacant(val) => {
                        let symbolic_v = v_to_expr($v_idx, false)?;
                        val.insert(symbolic_v)
                    }
                }
            };
            ($v_idx:expr, true) => {
                match v_bdd_cache_2.entry($v_idx) {
                    Entry::Occupied(val) => val.into_mut(),
                    Entry::Vacant(val) => {
                        let symbolic_v = v_to_expr($v_idx, true)?;
                        val.insert(symbolic_v)
                    }
                }
            };
        }

        tracing::trace!("Starting vertex BDD construction");
        // Contains all vertices in the graph
        let mut s_vertices = manager.with_manager_exclusive(|man| oxidd::bdd::BDDFunction::t(man));
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
                    blacklist = blacklist.or(v_to_expr_cached!(v_idx as u32, false))?;
                }

                s_vertices = s_vertices.and(&blacklist)?;
            } else {
                tracing::trace!(n_excluded = items_to_exclude, "Explicitly excluding vertices");
                for v_idx in explicit.vertex_count()..2_usize.pow(n_variables as u32) {
                    s_vertices = s_vertices.and(&v_to_expr_cached!(v_idx as u32, false).not()?)?;
                }
            }
        }

        tracing::trace!("Starting edge BDD construction");
        // Edges
        let mut s_edges = manager.with_manager_exclusive(|man| oxidd::bdd::BDDFunction::f(man));
        for edge in explicit.graph_edges() {
            let s_edge = v_to_expr_cached!(edge.source().index() as u32, false)
                .and(v_to_expr_cached!(edge.target().index() as u32, true))?;
            s_edges = s_edges.or(&s_edge)?;
        }

        // Priorities
        let mut s_priorities = explicit
            .priorities_unique()
            .map(|p| (p, manager.with_manager_exclusive(|man| oxidd::bdd::BDDFunction::f(man))))
            .collect::<HashMap<_, _>>();
        let mut s_even = manager.with_manager_exclusive(|man| oxidd::bdd::BDDFunction::f(man));
        let mut s_odd = manager.with_manager_exclusive(|man| oxidd::bdd::BDDFunction::f(man));
        
        tracing::trace!("Starting priority/owner BDD construction");
        for v_idx in explicit.vertices_index() {
            let vertex = explicit.get(v_idx).expect("Impossible");
            let expr = v_to_expr_cached!(v_idx.index() as u32, false);
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

    pub fn to_dot(&self) -> String {
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
                    ]);

                oxidd_dump::dot::dump_all(&mut out, man, variables, functions)
            })
            .expect("Failed to lock");

        String::from_utf8(out).expect("Invalid UTF-8")
    }
}

pub trait BddExtensions {
    /// Conceptually substitute the given `var` in `self` for `replace_with`.
    ///
    /// In practice this (inefficiently) creates a new BDD with: `exists var. (replace_with <=> var) && self`.
    fn substitute(&self, var: &BDDFunction, replace_with: &BDDFunction) -> AllocResult<BDDFunction>;
}

impl BddExtensions for BDDFunction {
    fn substitute(&self, var: &BDDFunction, replace_with: &BDDFunction) -> AllocResult<BDDFunction> {
        var.with_manager_shared(|manager, root| {
            let iff = Self::equiv_edge(manager, root, replace_with.as_edge(manager))?;
            let and = Self::and_edge(manager, self.as_edge(manager), &iff)?;
            let exists = Self::exist_edge(manager, &and, root)?;
            Ok(Self::from_edge(manager, exists))
        })
        // let iff = var.equiv(replace_with)?;
        // let and = self.and(&iff)?;
        // and.exist(var)
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use oxidd::bdd::BDDFunction;
    use oxidd_core::function::BooleanFunction;
    use oxidd_core::ManagerRef;
    use crate::ParityGame;
    use crate::symbolic::BddExtensions;
    use crate::tests::example_dir;

    #[test]
    pub fn test_symbolic() -> eyre::Result<()> {
        let mut pg = r#"parity 3;
0 1 1 0,1 "0";
1 1 0 2 "1";
2 2 0 2 "2";"#;
        let pg = pg_parser::parse_pg(&mut pg).unwrap();
        let pg = ParityGame::new(pg)?;
        let s_pg = super::SymbolicParityGame::from_explicit(&pg)?;
        std::fs::write("out.dot", s_pg.to_dot())?;
        // let mut final_subs_bdd = s_pg.manager.with_manager_exclusive(|man| BDDFunction::t(man));
        let mut true_base = s_pg.manager.with_manager_exclusive(|man| BDDFunction::t(man));
        let start_var = s_pg.variables.iter().fold(true_base.clone(), |acc, v_2| acc.and(v_2).unwrap());
        let other_vars = s_pg.variables_edges.iter().fold(true_base, |acc, v_2| acc.and(v_2).unwrap());

        let final_subs_bdd = s_pg.vertices.substitute(&start_var, &other_vars).unwrap();

        // for (var, var_x) in s_pg.variables.iter().zip(&s_pg.variables_edges) {
        //     final_subs_bdd = final_subs_bdd.and(&s_pg.vertices.substitute(var ,var_x).unwrap()).unwrap()
        // }
        s_pg.gc();

        let mut out = Vec::new();

        s_pg.manager
            .with_manager_exclusive(|man| {
                let variables = s_pg
                    .variables
                    .iter()
                    .chain(s_pg.variables_edges.iter())
                    .enumerate()
                    .map(|(i, v)| {
                        (
                            v,
                            if i >= s_pg.variables.len() {
                                format!("x_{}", i - s_pg.variables.len())
                            } else {
                                format!("x{i}")
                            },
                        )
                    });
                let functions = s_pg
                    .priorities
                    .iter()
                    .map(|(p, bdd)| (bdd, format!("Priority {p}")))
                    .chain([
                        (&s_pg.vertices, "vertices".into()),
                        (&s_pg.vertices_even, "vertices_even".into()),
                        (&s_pg.vertices_odd, "vertices_odd".into()),
                        (&s_pg.edges, "edges".into()),
                        (&final_subs_bdd, "substituted".into())
                    ]);

                oxidd_dump::dot::dump_all(&mut out, man, variables, functions)
            })
            .expect("Failed to lock");


        // let subs = s_pg.vertices.substitute(s_pg.)
        std::fs::write("out_subs.dot", String::from_utf8(out).expect("Invalid UTF-8"))?;
        Ok(())
    }

    #[test]
    pub fn test_symbolic_tue() -> eyre::Result<()> {
        let pg = std::fs::read_to_string(example_dir().join("tue_example.pg")).unwrap();
        let pg = pg_parser::parse_pg(&mut pg.as_str()).unwrap();
        let pg = ParityGame::new(pg)?;
        let s_pg = super::SymbolicParityGame::from_explicit(&pg)?;
        std::fs::write("out.dot", s_pg.to_dot())?;
        // let mut final_subs_bdd = s_pg.manager.with_manager_exclusive(|man| BDDFunction::t(man));
        let mut true_base = s_pg.manager.with_manager_exclusive(|man| BDDFunction::t(man));
        let start_var = s_pg.variables.iter().fold(true_base.clone(), |acc, v_2| acc.and(v_2).unwrap());
        let other_vars = s_pg.variables_edges.iter().fold(true_base, |acc, v_2| acc.and(v_2).unwrap());

        let final_subs_bdd = s_pg.vertices.substitute(&start_var, &other_vars).unwrap();

        // for (var, var_x) in s_pg.variables.iter().zip(&s_pg.variables_edges) {
        //     final_subs_bdd = final_subs_bdd.and(&s_pg.vertices.substitute(var ,var_x).unwrap()).unwrap()
        // }
        s_pg.gc();

        let mut out = Vec::new();

        s_pg.manager
            .with_manager_exclusive(|man| {
                let variables = s_pg
                    .variables
                    .iter()
                    .chain(s_pg.variables_edges.iter())
                    .enumerate()
                    .map(|(i, v)| {
                        (
                            v,
                            if i >= s_pg.variables.len() {
                                format!("x_{}", i - s_pg.variables.len())
                            } else {
                                format!("x{i}")
                            },
                        )
                    });
                let functions = s_pg
                    .priorities
                    .iter()
                    .map(|(p, bdd)| (bdd, format!("Priority {p}")))
                    .chain([
                        (&s_pg.vertices, "vertices".into()),
                        (&s_pg.vertices_even, "vertices_even".into()),
                        (&s_pg.vertices_odd, "vertices_odd".into()),
                        (&s_pg.edges, "edges".into()),
                        (&final_subs_bdd, "substituted".into())
                    ]);

                oxidd_dump::dot::dump_all(&mut out, man, variables, functions)
            })
            .expect("Failed to lock");


        // let subs = s_pg.vertices.substitute(s_pg.)
        std::fs::write("out_subs.dot", String::from_utf8(out).expect("Invalid UTF-8"))?;
        Ok(())
    }
}
