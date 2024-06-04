use std::{collections::HashMap, marker::PhantomData};

use ecow::EcoVec;
use oxidd::bdd::BDDManagerRef;
use oxidd_core::{
    function::{BooleanFunctionQuant, Function},
    ManagerRef,
    util::AllocResult,
};
use petgraph::prelude::EdgeRef;

use crate::{
    explicit::{ParityGame, ParityGraph, register_game::Rank},
    Owner,
    symbolic,
    symbolic::{
        BDD,
        helpers::CachedSymbolicEncoder,
        oxidd_extensions::{BooleanFunctionExtensions, FunctionManagerExtension},
    },
};

pub struct SymbolicRegisterGame<F: Function> {
    pub manager: F::ManagerRef,
    pub variables: RegisterVertexVars<F>,
    pub variables_edges: RegisterVertexVars<F>,
    pub conjugated_variables: F,
    pub conjugated_v_edges: F,

    pub v_even: F,
    pub v_odd: F,

    pub base_true: F,
    pub base_false: F,
}

impl<F> SymbolicRegisterGame<F>
where
    F: Function + BooleanFunctionExtensions + BooleanFunctionQuant + FunctionManagerExtension,
{
    /// Construct a new symbolic register game straight from an explicit parity game.
    ///
    /// See [crate::register_game::RegisterGame::construct]
    #[tracing::instrument(name = "Build Symbolic Register Game", skip_all)]
    pub fn from_explicit(explicit: &ParityGame, k: Rank, controller: Owner) -> symbolic::Result<Self> {
        let register_bits_needed = (explicit.priority_max() as f64).log2().ceil() as usize;
        // Calculate the amount of variables we'll need for the binary encodings of the registers and vertices.
        let n_regis_vars = register_bits_needed * (k as usize + 1);
        let n_variables = (explicit.vertex_count() as f64).log2().ceil() as usize;

        let manager = F::new_manager(explicit.vertex_count(), explicit.vertex_count(), 12);

        // Construct base building blocks for the BDD
        let base_true = manager.with_manager_exclusive(|man| F::t(man));
        let base_false = manager.with_manager_exclusive(|man| F::f(man));

        let (variables, edge_variables) = manager.with_manager_exclusive(|man| {
            Ok::<_, symbolic::BddError>((
                RegisterVertexVars {
                    next_move_var: F::new_var(man)?,
                    register_vars: (0..n_regis_vars).flat_map(|_| F::new_var(man)).collect::<EcoVec<_>>(),
                    vertex_vars: (0..n_variables).flat_map(|_| F::new_var(man)).collect::<EcoVec<_>>(),
                },
                RegisterVertexVars {
                    next_move_var: F::new_var(man)?,
                    register_vars: (0..n_regis_vars).flat_map(|_| F::new_var(man)).collect::<EcoVec<_>>(),
                    vertex_vars: (0..n_variables).flat_map(|_| F::new_var(man)).collect::<EcoVec<_>>(),
                },
            ))
        })?;

        let mut var_encoder = CachedSymbolicEncoder::new(&manager, variables.vertex_vars.clone());
        // let mut e_var_encoder = CachedSymbolicEncoder::new(&manager, edge_variables.clone());

        tracing::debug!("Starting player vertex set construction");
        let mut s_even = base_false.clone();
        let mut s_odd = base_false.clone();
        for v_idx in explicit.vertices_index() {
            let vertex = explicit.get(v_idx).expect("Impossible");
            let mut v_expr = var_encoder.encode(v_idx.index())?.clone();

            // The opposing player can only play with E_move
            // Thus, `next_move_var` should be `1/true`
            if controller != vertex.owner {
                v_expr = v_expr.and(&variables.next_move_var)?;
            }

            // Set owners
            // Note that possible register contents are not constrained here, this will be done in the edge relation.
            match vertex.owner {
                Owner::Even => s_even = s_even.or(&v_expr)?,
                Owner::Odd => s_odd = s_odd.or(&v_expr)?,
            }
        }

        // tracing::debug!("Starting edge BDD construction");
        // // Edges
        // let mut s_edges = base_false.clone();
        // for edge in explicit.graph_edges() {
        //     let (source, target) = (
        //         var_encoder.encode(edge.source().index())?,
        //         e_var_encoder.encode(edge.target().index())?,
        //     );
        //     s_edges = s_edges.or(&source.and(target)?)?;
        // }
        //
        // // Priorities
        // let mut s_priorities = explicit
        //     .priorities_unique()
        //     .map(|p| (p, base_false.clone()))
        //     .collect::<HashMap<_, _>>();
        // let mut s_even = base_false.clone();
        // let mut s_odd = base_false.clone();
        //
        // tracing::debug!("Starting priority/owner BDD construction");
        // for v_idx in explicit.vertices_index() {
        //     let vertex = explicit.get(v_idx).expect("Impossible");
        //     let expr = var_encoder.encode(v_idx.index())?;
        //     let priority = vertex.priority;
        //
        //     // Fill priority
        //     s_priorities
        //         .entry(priority)
        //         .and_modify(|bdd| *bdd = bdd.or(&expr).expect("Out of memory"));
        //
        //     // Set owners
        //     match vertex.owner {
        //         Owner::Even => s_even = s_even.or(&expr)?,
        //         Owner::Odd => s_odd = s_odd.or(&expr)?,
        //     }
        // }

        let conj_v = variables.conjugated(&base_true)?;
        let conj_e = edge_variables.conjugated(&base_true)?;

        Ok(Self {
            manager,
            variables,
            variables_edges: edge_variables,
            conjugated_variables: conj_v,
            conjugated_v_edges: conj_e,
            v_even: s_even,
            v_odd: s_odd,
            base_true,
            base_false,
        })
    }
}

pub struct RegisterVertexVars<F: Function> {
    pub next_move_var: F,
    pub register_vars: EcoVec<F>,
    pub vertex_vars: EcoVec<F>,
}

impl<F: BooleanFunctionExtensions> RegisterVertexVars<F> {
    fn conjugated(&self, base_true: &F) -> AllocResult<F> {
        self.next_move_var
            .and(
                &self
                    .register_vars
                    .iter()
                    .fold(base_true.clone(), |acc: F, next: &F| acc.and(next).unwrap()),
            )?
            .and(
                &self
                    .vertex_vars
                    .as_slice()
                    .iter()
                    .fold(base_true.clone(), |acc: F, next: &F| acc.and(next).unwrap()),
            )
    }

    pub fn iter(&self) -> impl Iterator<Item = &F> {
        [&self.next_move_var]
            .into_iter()
            .chain(&self.register_vars)
            .chain(&self.vertex_vars)
    }

    pub fn iter_names<'a>(&'a self, suffix: &'a str) -> impl Iterator<Item = (&F, String)> + 'a {
        [(&self.next_move_var, format!("t{suffix}"))]
            .into_iter()
            .chain(
                self.register_vars
                    .iter()
                    .enumerate()
                    .map(move |(i, r)| (r, format!("r{i}{suffix}"))),
            )
            .chain(
                self.vertex_vars
                    .iter()
                    .enumerate()
                    .map(move |(i, r)| (r, format!("x{i}{suffix}"))),
            )
    }
}

#[cfg(test)]
mod tests {
    use oxidd_core::{Manager, ManagerRef};

    use pg_parser::parse_pg;

    use crate::{
        explicit::ParityGame,
        Owner,
        symbolic::{BDD, register_game::SymbolicRegisterGame},
        tests::example_dir,
        visualize::DotWriter,
    };

    #[test]
    pub fn test() {
        let input = std::fs::read_to_string(example_dir().join("tue_example.pg")).unwrap();
        let pg = parse_pg(&mut input.as_str()).unwrap();
        let game = ParityGame::new(pg).unwrap();
        let s_pg: SymbolicRegisterGame<BDD> = SymbolicRegisterGame::from_explicit(&game, 1, Owner::Even).unwrap();
        s_pg.manager.with_manager_exclusive(|man| man.gc());

        std::fs::write("out.dot", DotWriter::write_dot_symbolic_register(&s_pg, []).unwrap()).unwrap();
    }
}
