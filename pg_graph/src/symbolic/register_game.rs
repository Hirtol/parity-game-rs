use std::{collections::HashMap, marker::PhantomData};

use ecow::EcoVec;
use itertools::Itertools;
use oxidd::bdd::BDDManagerRef;
use oxidd_core::{
    function::{BooleanFunctionQuant, Function, FunctionSubst},
    ManagerRef,
    util::{AllocResult, Subst},
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
    pub e_move: F,

    pub base_true: F,
    pub base_false: F,
}

impl<F> SymbolicRegisterGame<F>
where
    F: Function + BooleanFunctionExtensions + BooleanFunctionQuant + FunctionManagerExtension + FunctionSubst,
{
    /// Construct a new symbolic register game straight from an explicit parity game.
    ///
    /// See [crate::register_game::RegisterGame::construct]
    #[tracing::instrument(name = "Build Symbolic Register Game", skip_all)]
    pub fn from_explicit(explicit: &ParityGame, k: Rank, controller: Owner) -> symbolic::Result<Self> {
        let k = k as usize;
        let register_bits_needed = (explicit.priority_max() as f64).log2().ceil() as usize;
        // Calculate the amount of variables we'll need for the binary encodings of the registers and vertices.
        let n_regis_vars = register_bits_needed * (k + 1);
        let n_variables = (explicit.vertex_count() as f64).log2().ceil() as usize;

        let manager = F::new_manager(explicit.vertex_count(), explicit.vertex_count(), 12);

        // Construct base building blocks for the BDD
        let base_true = manager.with_manager_exclusive(|man| F::t(man));
        let base_false = manager.with_manager_exclusive(|man| F::f(man));

        let (variables, edge_variables) = manager.with_manager_exclusive(|man| {
            let next_move = F::new_var(man)?;
            let next_move_edge = F::new_var(man)?;
            Ok::<_, symbolic::BddError>((
                RegisterVertexVars::new(
                    next_move,
                    (0..n_regis_vars)
                        .flat_map(|_| F::new_var(man))
                        .collect_vec()
                        .into_iter(),
                    (0..n_variables).flat_map(|_| F::new_var(man)),
                ),
                RegisterVertexVars::new(
                    next_move_edge,
                    (0..n_regis_vars)
                        .flat_map(|_| F::new_var(man))
                        .collect_vec()
                        .into_iter(),
                    (0..n_variables).flat_map(|_| F::new_var(man)),
                ),
            ))
        })?;

        let mut var_encoder = CachedSymbolicEncoder::new(&manager, variables.vertex_vars().into());
        let mut e_var_encoder = CachedSymbolicEncoder::new(&manager, edge_variables.vertex_vars().into());

        tracing::debug!("Starting player vertex set construction");

        let mut s_even = base_false.clone();
        let mut s_odd = base_false.clone();
        for v_idx in explicit.vertices_index() {
            let vertex = explicit.get(v_idx).expect("Impossible");
            let mut v_expr = var_encoder.encode(v_idx.index())?.clone();

            // The opposing player can only play with E_move
            // Thus, `next_move_var` should be `1/true`
            if controller != vertex.owner {
                v_expr = v_expr.and(variables.next_move_var())?;
            }

            // Set owners
            // Note that possible register contents are not constrained here, this will be done in the edge relation.
            match vertex.owner {
                Owner::Even => s_even = s_even.or(&v_expr)?,
                Owner::Odd => s_odd = s_odd.or(&v_expr)?,
            }
        }

        tracing::debug!("Starting E_move construction");

        let mut e_move = base_false.clone();
        let iff_register_condition = variables
            .register_vars()
            .iter()
            .zip(edge_variables.register_vars())
            .fold(base_true.clone(), |acc, (r, r_next)| {
                acc.and(&r.equiv(r_next).unwrap()).unwrap()
            });

        for edge in explicit.graph_edges() {
            let (source, target) = (
                var_encoder
                    .encode(edge.source().index())?
                    .and(&variables.next_move_var())?,
                e_var_encoder
                    .encode(edge.target().index())?
                    .diff(&edge_variables.next_move_var())?,
            );
            e_move = e_move.or(&source.and(&target)?.and(&iff_register_condition)?)?;
        }

        tracing::debug!("Starting E_i construction");
        let n_priorities = if controller == Owner::Even { k + 1 } else { k + 2 };
        let mut priorities = (0..=2 * n_priorities)
            .into_iter()
            .map(|val| (val, base_false.clone()))
            .collect::<ahash::HashMap<_, _>>();
        // All E_move edges get a priority of `0` by default.
        let _ = priorities
            .entry(0)
            .and_modify(|pri| *pri = variables.next_move_var().not().unwrap());

        let mut e_i_edges = vec![base_false.clone(); k + 1];

        for i in 0..=k {
            let e_i = &mut e_i_edges[i];
            // From t=0 -> t=1
            let base_edge = edge_variables.next_move_var().diff(&variables.next_move_var())?;

            for (v_idx, vertex) in explicit.vertices_and_index() {}
        }

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
            e_move,
            base_true,
            base_false,
        })
    }

    #[inline]
    fn edge_substitute(&self, bdd: &F) -> symbolic::Result<F> {
        let subs = Subst::new(&self.variables.all_variables, &self.variables_edges.all_variables);
        Ok(bdd.substitute(subs)?)
    }

    #[inline]
    fn rev_edge_substitute(&self, bdd: &F) -> symbolic::Result<F> {
        let subs = Subst::new(&self.variables_edges.all_variables, &self.variables.all_variables);
        Ok(bdd.substitute(subs)?)
    }
}

pub struct RegisterVertexVars<F: Function> {
    pub all_variables: EcoVec<F>,
    n_register_vars: usize,
}

impl<F: BooleanFunctionExtensions> RegisterVertexVars<F> {
    pub fn new(
        next_move: F,
        register_vars: impl ExactSizeIterator<Item = F>,
        vertex_vars: impl Iterator<Item = F>,
    ) -> Self {
        Self {
            n_register_vars: register_vars.len(),
            all_variables: [next_move]
                .into_iter()
                .chain(register_vars)
                .chain(vertex_vars)
                .collect(),
        }
    }

    #[inline(always)]
    pub fn next_move_var(&self) -> &F {
        &self.all_variables[0]
    }

    #[inline(always)]
    pub fn register_vars(&self) -> &[F] {
        &self.all_variables[1..self.n_register_vars + 1]
    }

    #[inline(always)]
    pub fn vertex_vars(&self) -> &[F] {
        &self.all_variables[self.n_register_vars + 1..]
    }

    fn conjugated(&self, base_true: &F) -> AllocResult<F> {
        Ok(self
            .all_variables
            .iter()
            .fold(base_true.clone(), |acc: F, next: &F| acc.and(next).unwrap()))
    }

    pub fn iter_names<'a>(&'a self, suffix: &'a str) -> impl Iterator<Item = (&F, String)> + 'a {
        [(self.next_move_var(), format!("t{suffix}"))]
            .into_iter()
            .chain(
                self.register_vars()
                    .iter()
                    .enumerate()
                    .map(move |(i, r)| (r, format!("r{i}{suffix}"))),
            )
            .chain(
                self.vertex_vars()
                    .iter()
                    .enumerate()
                    .map(move |(i, r)| (r, format!("x{i}{suffix}"))),
            )
    }
}

#[cfg(test)]
mod tests {
    use oxidd_core::{
        function::{BooleanFunction, BooleanFunctionQuant},
        Manager, ManagerRef,
    };

    use pg_parser::parse_pg;

    use crate::{
        explicit::ParityGame,
        Owner,
        symbolic::{BDD, helpers::CachedSymbolicEncoder, register_game::SymbolicRegisterGame},
        tests::example_dir,
        visualize::DotWriter,
    };

    #[test]
    pub fn test_tue() {
        let input = std::fs::read_to_string(example_dir().join("tue_example.pg")).unwrap();
        let pg = parse_pg(&mut input.as_str()).unwrap();
        let game = ParityGame::new(pg).unwrap();
        let s_pg: SymbolicRegisterGame<BDD> = SymbolicRegisterGame::from_explicit(&game, 1, Owner::Even).unwrap();
        s_pg.manager.with_manager_exclusive(|man| man.gc());

        std::fs::write("out.dot", DotWriter::write_dot_symbolic_register(&s_pg, []).unwrap()).unwrap();
    }

    #[test]
    pub fn test_small() {
        let game = small_pg().unwrap();
        let s_pg: SymbolicRegisterGame<BDD> = SymbolicRegisterGame::from_explicit(&game, 1, Owner::Even).unwrap();
        s_pg.manager.with_manager_exclusive(|man| man.gc());
        let vert = CachedSymbolicEncoder::encode_impl(&s_pg.variables.vertex_vars(), 0usize).unwrap();
        let next_move = vert.and(&s_pg.variables.next_move_var()).unwrap();
        let register_contents = CachedSymbolicEncoder::encode_impl(&s_pg.variables.register_vars(), 3usize).unwrap();
        let full_vert = next_move.and(&register_contents).unwrap();

        let total_next_v = s_pg
            .e_move
            .and(&full_vert)
            .unwrap()
            .exist(&s_pg.conjugated_variables)
            .unwrap();
        let real = s_pg.rev_edge_substitute(&total_next_v).unwrap();
        s_pg.manager.with_manager_exclusive(|man| man.gc());

        std::fs::write(
            "out.dot",
            DotWriter::write_dot_symbolic_register(&s_pg, [(&real, "Next V".to_string())]).unwrap(),
        )
        .unwrap();
    }

    fn small_pg() -> eyre::Result<ParityGame> {
        let mut pg = r#"parity 3;
0 1 1 0,1 "0";
1 1 0 2 "1";
2 2 0 2 "2";"#;
        let pg = pg_parser::parse_pg(&mut pg).unwrap();
        ParityGame::new(pg)
    }
}
