use std::{collections::HashMap, marker::PhantomData};

use ecow::EcoVec;
use oxidd::bdd::BDDManagerRef;
use oxidd_core::{
    function::{BooleanFunctionQuant, Function},
    ManagerRef,
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
    pub variables: EcoVec<F>,
    pub variables_edges: EcoVec<F>,
    pub conjugated_variables: F,
    pub conjugated_v_edges: F,
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

        let (next_move_var, reg_vars, variables, edge_next_move_var, edge_reg_vars, edge_variables) = manager
            .with_manager_exclusive(|man| {
                Ok::<_, symbolic::BddError>((
                    F::new_var(man)?,
                    (0..n_regis_vars).flat_map(|_| F::new_var(man)).collect::<EcoVec<_>>(),
                    (0..n_variables).flat_map(|_| F::new_var(man)).collect::<EcoVec<_>>(),
                    F::new_var(man)?,
                    (0..n_regis_vars).flat_map(|_| F::new_var(man)).collect::<EcoVec<_>>(),
                    (0..n_variables).flat_map(|_| F::new_var(man)).collect::<EcoVec<_>>(),
                ))
            })?;

        let mut var_encoder = CachedSymbolicEncoder::new(&manager, variables.clone());
        let mut e_var_encoder = CachedSymbolicEncoder::new(&manager, edge_variables.clone());

        tracing::debug!("Starting edge BDD construction");
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
        let mut s_odd = base_false.clone();

        tracing::debug!("Starting priority/owner BDD construction");
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

        let conj_v = variables
            .iter()
            .fold(base_true.clone(), |acc, next| acc.and(next).unwrap());
        let conj_e = edge_variables
            .iter()
            .fold(base_true.clone(), |acc, next| acc.and(next).unwrap());

        Ok(Self {
            manager,
            variables,
            variables_edges: edge_variables,
            conjugated_variables: conj_v,
            conjugated_v_edges: conj_e,
            base_true,
            base_false,
        })
    }
}
