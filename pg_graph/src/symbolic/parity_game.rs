use ecow::EcoVec;
use oxidd_core::{
    function::{Function},
    Manager,
    ManagerRef, util::{OptBool, Subst}, WorkerManager,
};
use petgraph::prelude::EdgeRef;

use crate::{
    datatypes::Priority,
    explicit::{ParityGame, ParityGraph, VertexId},
    Owner,
    symbolic,
    symbolic::{
        BCDD,
        BDD,
        helpers::CachedSymbolicEncoder,
        oxidd_extensions::{GeneralBooleanFunction}, sat::TruthAssignmentsIterator,
    },
};

pub struct SymbolicParityGame<F: Function> {
    pub pg_vertex_count: usize,
    pub manager: F::ManagerRef,
    pub variables: EcoVec<F>,
    pub variables_edges: EcoVec<F>,
    pub conjugated_variables: F,
    pub conjugated_v_edges: F,

    pub vertices: F,
    pub vertices_even: F,
    pub vertices_odd: F,
    pub priorities: ahash::HashMap<Priority, F>,

    pub edges: F,

    pub base_true: F,
    pub base_false: F,
}

impl SymbolicParityGame<BDD> {
    pub fn from_explicit_bdd(explicit: &ParityGame) -> eyre::Result<SymbolicParityGame<BDD>> {
        Self::from_explicit(explicit)
    }
}

impl SymbolicParityGame<BCDD> {
    pub fn from_explicit_bcdd(explicit: &ParityGame) -> eyre::Result<SymbolicParityGame<BCDD>> {
        Self::from_explicit(explicit)
    }
}

impl<F: GeneralBooleanFunction> SymbolicParityGame<F>
where
    for<'id> F::Manager<'id>: WorkerManager,
    for<'a, 'b> TruthAssignmentsIterator<'b, 'a, F>: Iterator<Item = Vec<OptBool>>,
{
    /// For now, mostly translated from [here](https://github.com/olijzenga/bdd-parity-game-solver/blob/master/src/pg.py).
    ///
    /// Constructs the following BDDs:
    /// * A BDD for the sets `V_even` and `V_odd`
    /// * A BDD representing the edge relation `E`
    /// * A BDD for every priority in the game, containing the equation for the vertices which have that priority.
    #[tracing::instrument(name = "Build Symbolic Parity Game", skip_all)]
    pub fn from_explicit(explicit: &ParityGame) -> eyre::Result<Self> {
        Self::from_explicit_impl(explicit).map_err(|e| eyre::eyre!("Could not construct BDD due to: {e:?}"))
    }

    fn from_explicit_impl(explicit: &ParityGame) -> symbolic::Result<Self> {
        let n_variables = (explicit.vertex_count() as f64).log2().ceil() as usize;
        let manager = F::new_manager(explicit.vertex_count(), explicit.vertex_count(), 12);

        // Construction is _much_ faster single-threaded.
        manager.with_manager_exclusive(|man| {
            man.set_threading_enabled(false);
        });

        // Construct base building blocks for the BDD
        let variables: EcoVec<F> =
            manager.with_manager_exclusive(|man| (0..n_variables).flat_map(|_| F::new_var(man)).collect());
        let edge_variables: EcoVec<F> =
            manager.with_manager_exclusive(|man| (0..n_variables).flat_map(|_| F::new_var(man)).collect());

        Self::from_explicit_impl_vars(explicit, manager, variables, edge_variables)
    }

    /// Create the symbolic game with the given variables.
    /// It is assumed that the amount of variables matches the size of the binary encoding of the max vertex id.
    pub(crate) fn from_explicit_impl_vars(
        explicit: &ParityGame,
        manager: F::ManagerRef,
        variables: EcoVec<F>,
        edge_variables: EcoVec<F>,
    ) -> symbolic::Result<Self> {
        let base_true = manager.with_manager_exclusive(|man| F::t(man));
        let base_false = manager.with_manager_exclusive(|man| F::f(man));

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
            .collect::<ahash::HashMap<_, _>>();
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
            .try_fold(base_true.clone(), |acc, next| acc.and(next))?;
        let conj_e = edge_variables
            .iter()
            .try_fold(base_true.clone(), |acc, next| acc.and(next))?;

        Ok(Self {
            pg_vertex_count: explicit.vertex_count(),
            manager,
            variables,
            variables_edges: edge_variables,
            conjugated_variables: conj_v,
            conjugated_v_edges: conj_e,
            vertices: s_even.or(&s_odd)?,
            vertices_even: s_even,
            vertices_odd: s_odd,
            edges: s_edges,
            base_true,
            priorities: s_priorities,
            base_false,
        })
    }

    /// Remove unreferenced nodes.
    pub fn gc(&self) -> usize {
        self.manager.with_manager_exclusive(|man| man.gc())
    }

    /// The amount of vertices in the underlying parity game.
    pub fn vertex_count(&self) -> usize {
        self.pg_vertex_count
    }

    /// Get the amount of BDD nodes.
    pub fn bdd_node_count(&self) -> usize {
        self.manager.with_manager_shared(|man| man.num_inner_nodes())
    }

    /// Count of all variables used in the BDD.
    pub fn bdd_variable_count(&self) -> u32 {
        (self.variables.len() + self.variables_edges.len()) as u32
    }

    pub fn encode_vertex(&self, v_idx: VertexId) -> symbolic::Result<F> {
        CachedSymbolicEncoder::<_, F>::encode_impl(&self.variables, v_idx.index())
    }

    pub fn encode_edge_vertex(&self, v_idx: VertexId) -> symbolic::Result<F> {
        CachedSymbolicEncoder::<_, F>::encode_impl(&self.variables_edges, v_idx.index())
    }

    /// Calculate all vertex ids which belong to the set represented by the `bdd`.
    ///
    /// Inefficient, and should only be used for debugging.
    pub fn vertices_of_bdd(&self, bdd: &F) -> Vec<VertexId> {
        self.manager.with_manager_shared(|man| {
            let valuations = bdd.sat_assignments(man);
            crate::symbolic::sat::decode_assignments(valuations, self.variables.len())
        })
    }

    pub fn create_subgame(&self, ignored: &F) -> symbolic::Result<Self> {
        let ignored_edge = self.edge_substitute(ignored)?;

        let priorities = self
            .priorities
            .iter()
            .flat_map(|(priority, bdd)| Ok::<_, symbolic::BddError>((*priority, bdd.diff(&ignored)?)))
            .collect();

        Ok(Self {
            pg_vertex_count: self.pg_vertex_count,
            manager: self.manager.clone(),
            variables: self.variables.clone(),
            variables_edges: self.variables_edges.clone(),
            conjugated_variables: self.conjugated_variables.clone(),
            conjugated_v_edges: self.conjugated_v_edges.clone(),
            vertices: self.vertices.diff(&ignored)?,
            vertices_even: self.vertices_even.diff(&ignored)?,
            vertices_odd: self.vertices_odd.diff(&ignored)?,
            edges: self.edges.diff(&ignored)?.diff(&ignored_edge)?,
            base_true: self.base_true.clone(),
            priorities,
            base_false: self.base_false.clone(),
        })
    }

    /// Return the maximal priority found in the given game.
    #[inline(always)]
    pub fn priority_max(&self) -> Priority {
        *self
            .priorities
            .iter()
            .filter(|(p, bdd)| self.base_false != **bdd)
            .map(|(p, _)| p)
            .max()
            .expect("No priority in game")
    }

    /// Calculate the predecessors of the set of vertices `of`.
    pub fn predecessors(&self, of: &F) -> symbolic::Result<F> {
        let of_subs = self.edge_substitute(of)?;

        Ok(self.edges.and(&of_subs)?)
    }

    /// Calculate the attraction set for the given starting set.
    ///
    /// This resulting set will contain all vertices which, after a fixed-point iteration:
    /// * If a vertex is owned by `player`, then if any edge leads to the attraction set it will be added to the resulting set.
    /// * If a vertex is _not_ owned by `player`, then only if _all_ edges lead to the attraction set will it be added.
    #[tracing::instrument(level = "trace", skip_all, fields(player))]
    pub fn attractor_set(&self, player: Owner, starting_set: &F) -> symbolic::Result<F> {
        let (player_set, opponent_set) = self.get_player_sets(player);
        let mut output = starting_set.clone();
        let edge_player_set = self.edges.and(player_set)?;

        loop {
            let edge_starting_set = self.edge_substitute(&output)?;
            // Set of elements which have _any_ edge leading to our `starting_set` and are owned by `player`.
            let any_edge_set = edge_player_set
                .and(&edge_starting_set)?
                .exist(&self.conjugated_v_edges)?;

            // !edge_starting_set & self.edges
            let edges_to_outside = self.edges.diff(&edge_starting_set)?;
            // Set of elements which have _no_ edges leading outside our `starting_set`. In other words, all edges point to our attractor set.
            let all_edge_set = opponent_set.diff(&edges_to_outside.exist(&self.conjugated_v_edges)?)?;

            let new_output = output.or(&any_edge_set)?.or(&all_edge_set)?;

            if new_output == output {
                break;
            } else {
                output = new_output;
            }
        }

        Ok(output)
    }

    /// Substitute all vertex variables `x0..xn` with the edge variable `x0'..xn'`.
    #[inline]
    fn edge_substitute(&self, bdd: &F) -> symbolic::Result<F> {
        let subs = Subst::new(&self.variables, &self.variables_edges);
        Ok(bdd.substitute(subs)?)
    }

    /// Return both sets of vertices, with the first element of the returned tuple matching the `player`.
    #[inline(always)]
    pub(crate) fn get_player_sets(&self, player: Owner) -> (&F, &F) {
        match player {
            Owner::Even => (&self.vertices_even, &self.vertices_odd),
            Owner::Odd => (&self.vertices_odd, &self.vertices_even),
        }
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use oxidd::bdd::BDDFunction;
    use oxidd_core::{function::BooleanFunction, util::OptBool};

    use crate::{
        explicit::{ParityGame, ParityGraph, solvers::AttractionComputer},
        Owner,
        symbolic::{BDD, oxidd_extensions::BddExtensions, parity_game::SymbolicParityGame},
        tests::load_example,
        visualize::DotWriter,
    };

    fn small_pg() -> eyre::Result<ParityGame> {
        let mut pg = r#"parity 3;
0 1 1 0,1 "0";
1 1 0 2 "1";
2 2 0 2 "2";"#;
        let pg = pg_parser::parse_pg(&mut pg).unwrap();
        ParityGame::new(pg)
    }

    fn other_pg() -> eyre::Result<ParityGame> {
        let mut pg = r#"parity 4;
0 1 1 0,1 "0";
1 1 0 2 "1";
2 2 0 2 "2";
3 1 1 2 "3";"#;
        let pg = pg_parser::parse_pg(&mut pg).unwrap();
        ParityGame::new(pg)
    }

    macro_rules! id_vec {
        ($($x:expr),+ $(,)?) => (
            vec![$($x.into()),+]
        );
    }

    #[test]
    pub fn test_symbolic() -> eyre::Result<()> {
        let s = symbolic_pg(small_pg()?)?;

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
        let s = symbolic_pg(small_pg()?)?;

        let predecessor = s.pg.predecessors(&s.vertices[2])?;

        let count = predecessor.sat_quick_count(s.pg.bdd_variable_count() as u32);
        assert_eq!(count, 2);

        let valuations = [true, false]
            .into_iter()
            .flat_map(|v| predecessor.pick_cube(None, |_, _| v))
            .collect_vec();

        assert_eq!(
            valuations,
            vec![
                vec![OptBool::True, OptBool::False, OptBool::False, OptBool::True],
                vec![OptBool::False, OptBool::True, OptBool::False, OptBool::True]
            ]
        );

        Ok(())
    }

    #[test]
    pub fn test_attraction_set() -> eyre::Result<()> {
        let s = symbolic_pg(other_pg()?)?;

        let attr_set = s.pg.attractor_set(Owner::Even, &s.vertices[2])?;

        let vertices = s.pg.vertices_of_bdd(&attr_set);
        assert_eq!(vertices, id_vec![2, 3, 1]);

        let s_tue = symbolic_pg(load_example("tue_example.pg"))?;

        let start_set = &s_tue.pg.priorities.get(&s_tue.pg.priority_max()).unwrap();
        println!("Starting set: {:#?}", s_tue.pg.vertices_of_bdd(start_set));

        let attr_set = s_tue.pg.attractor_set(Owner::Odd, start_set)?;

        s_tue.pg.gc();

        let attr_set_vertices = s_tue.pg.vertices_of_bdd(&attr_set);

        println!("Attraction set: {:#?}", s_tue.pg.vertices_of_bdd(&attr_set));
        let mut real_attraction = AttractionComputer::new();
        let underlying_attr_set = real_attraction.attractor_set(
            &s_tue.original,
            Owner::Odd,
            s_tue
                .original
                .vertices_by_priority_idx(s_tue.pg.priority_max())
                .map(|(a, b)| a),
        );

        assert_eq!(underlying_attr_set, attr_set_vertices.into_iter().collect());

        Ok(())
    }

    #[test]
    fn test_attraction_set_large() -> eyre::Result<()> {
        let s = symbolic_pg(load_example("amba_decomposed_arbiter_2.tlsf.ehoa.pg"))?;

        let start_set = &s.pg.priorities.get(&s.pg.priority_max()).unwrap();
        println!("Starting set: {:#?}", s.pg.vertices_of_bdd(start_set));

        let attr_set = s.pg.attractor_set(Owner::Even, start_set)?;
        let attr_set_vertices = s.pg.vertices_of_bdd(&attr_set);

        println!("Attraction set: {:#?}", s.pg.vertices_of_bdd(&attr_set));
        let mut real_attraction = AttractionComputer::new();
        let underlying_attr_set = real_attraction.attractor_set(
            &s.original,
            Owner::Even,
            s.original.vertices_by_priority_idx(s.pg.priority_max()).map(|(a, b)| a),
        );

        assert_eq!(underlying_attr_set, attr_set_vertices.into_iter().collect());
        Ok(())
    }

    #[test]
    pub fn test_symbolic_substitution() -> eyre::Result<()> {
        let s = symbolic_pg(small_pg()?)?;
        let s_pg = &s.pg;

        let final_subs_bdd = s_pg.edge_substitute(&s_pg.vertices).unwrap();

        // Ensure that the variables were substituted correctly
        let edge_nodes_exist = &s.edge_vertices[0];
        for edge_vertex in s.edge_vertices {
            assert!(final_subs_bdd.and(&edge_vertex).unwrap().satisfiable());
        }

        s.pg.gc();
        let dot = DotWriter::write_dot_symbolic(&s_pg, [(&final_subs_bdd, "substitution".into())])?;
        std::fs::write("out_sym.dot", dot)?;

        Ok(())
    }

    struct SymbolicTest {
        pg: SymbolicParityGame<BDD>,
        original: ParityGame,
        vertices: Vec<BDDFunction>,
        edge_vertices: Vec<BDDFunction>,
    }

    fn symbolic_pg(pg: ParityGame) -> eyre::Result<SymbolicTest> {
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
            original: pg,
            vertices: nodes,
            edge_vertices: edge_nodes,
        })
    }
}
