use oxidd_core::{Manager, ManagerRef};
use oxidd_core::util::Subst;

use crate::{Owner, Priority, symbolic};
use crate::symbolic::oxidd_extensions::GeneralBooleanFunction;
use crate::symbolic::register_game::SymbolicRegisterGame;

impl<F: GeneralBooleanFunction> SymbolicRegisterGame<F>
{
    pub fn gc(&self) -> usize {
        self.manager.with_manager_exclusive(|m| m.gc())
    }

    pub fn bdd_node_count(&self) -> usize {
        self.manager.with_manager_exclusive(|m| m.num_inner_nodes())
    }
    
    /// Return the maximal priority found in the given game.
    pub fn priority_max(&self) -> Priority {
        *self
            .priorities
            .iter()
            .filter(|(p, bdd)| self.base_false != **bdd)
            .map(|(p, _)| p)
            .max()
            .expect("No priority in game")
    }

    pub fn create_subgame(&self, ignored: &F) -> symbolic::Result<Self> {
        // .and(ignored)?
        let ignored_edge = self.edge_substitute(ignored)?;
        // let ig = ignored.diff(&ignored_edge)?;

        let priorities = self
            .priorities
            .iter()
            .flat_map(|(priority, bdd)| Ok::<_, symbolic::BddError>((*priority, bdd.diff(ignored)?)))
            .collect();

        Ok(Self {
            k: self.k,
            controller: self.controller,
            manager: self.manager.clone(),
            variables: self.variables.clone(),
            variables_edges: self.variables_edges.clone(),
            conjugated_variables: self.conjugated_variables.clone(),
            conjugated_v_edges: self.conjugated_v_edges.clone(),
            vertices: self.vertices.diff(ignored)?,
            v_even: self.v_even.diff(ignored)?,
            v_odd: self.v_odd.diff(ignored)?,
            priorities,
            edges: self.edges.diff(ignored)?.diff(&ignored_edge)?,
            e_move: self.e_move.diff(ignored)?.diff(&ignored_edge)?,
            e_i_all: self.e_i_all.diff(ignored)?.diff(&ignored_edge)?,
            e_i: self.e_i.iter().flat_map(|e_i| Ok::<_, symbolic::BddError>(e_i.diff(ignored)?.diff(&ignored_edge)?)).collect(),
            base_true: self.base_true.clone(),
            base_false: self.base_false.clone(),
        })
    }

    pub fn attractor_priority_set(&self, player: Owner, starting_set: &F, priority: Priority) -> symbolic::Result<F> {
        // Due to the structure of the register game _all_ vertices in the game will be part of the attractor set when the
        // target priority is 0, as all E_i vertices have 0 priority.
        // println!("Custom Attractor: {priority}");
        if priority == 0 {
            Ok(self.vertices.clone())
        } else {
            // We know that any non-zero priority will _only_ include E_move vertices, thus we can attract two iterations
            // of vertices in one go when the controller is aligned.
            if player == self.controller {
                // println!("Running controller attractor");
                // self.gc();
                // let now = std::time::Instant::now();
                // let out = self.attractor_set_controller(starting_set);
                // tracing::debug!(priority, "Controller set took: {:?}", now.elapsed());
                // drop(out);
                // self.gc();
                // let now = std::time::Instant::now();
                // let out = self.attractor_set(player, starting_set);
                // tracing::debug!(priority, "Normal set took: {:?}", now.elapsed());
                // out
                self.attractor_set_controller(starting_set)
            } else {
                self.attractor_set(player, starting_set)
                // self.attractor_set_opposite_controller(starting_set)
            }
        }
    }

    /// Compute the attractor set for the special case when the target starting set's player matches the controller.
    fn attractor_set_controller(&self, starting_set: &F) -> symbolic::Result<F> {
        let (player_set, opponent_set) = self.get_player_sets(self.controller);
        let mut output = starting_set.clone();
        let e_move_player_set = self.e_move.and(player_set)?;

        loop {
            let edge_starting_set = self.edge_substitute(&output)?;
            // Set of elements which have _any_ E_i edge leading to our `starting_set` of E_move.
            // We know that all `e_i` vertices are owned by the controller
            let e_i_attracted_set = self.e_i_all
                .and(&edge_starting_set)?
                .exist(&self.conjugated_v_edges)?;

            // The E_move vertices can be owned by either player, so we'll need to do the whole procedure.
            let next_edge_starting_set = self.edge_substitute(&e_i_attracted_set)?.or(&edge_starting_set)?;

            let any_edge_set = e_move_player_set
                .and(&next_edge_starting_set)?
                .exist(&self.conjugated_v_edges)?;

            // !edge_starting_set & self.e_move
            let edges_to_outside = self.e_move.diff(&next_edge_starting_set)?;
            // Set of elements which have _no_ edges leading outside our `starting_set`. In other words, all edges point to our attractor set.
            let all_edge_set = opponent_set.diff(&edges_to_outside.exist(&self.conjugated_v_edges)?)?;

            let new_output = output.or(&e_i_attracted_set)?.or(&any_edge_set)?.or(&all_edge_set)?;

            if new_output == output {
                break;
            } else {
                output = new_output;
            }
        }

        Ok(output)
    }

    /// Attractor set for when the owning player does _not_ equal the controller
    fn attractor_set_opposite_controller(&self, starting_set: &F) -> symbolic::Result<F> {
        let (player_set, opponent_set) = self.get_player_sets(self.controller.other());
        let mut output = starting_set.clone();
        let e_move_player_set = self.e_move.and(player_set)?;

        loop {
            let edge_starting_set = self.edge_substitute(&output)?;
            // Set of elements which have _all_ E_i edge leading to our `starting_set` of E_move.
            // We know that all `e_i` vertices are owned by the controller, and the target set is owned by the opponent.
            let edges_to_outside = self.e_i_all.diff(&edge_starting_set)?;
            // Set of elements which have _no_ edges leading outside our `starting_set`. In other words, all edges point to our attractor set.
            let e_i_attracted_set = opponent_set.diff(&edges_to_outside.exist(&self.conjugated_v_edges)?)?;
            
            // The E_move vertices can be owned by either player, so we'll need to do the whole procedure.
            let next_edge_starting_set = self.edge_substitute(&e_i_attracted_set)?.or(&edge_starting_set)?;

            let any_edge_set = e_move_player_set
                .and(&next_edge_starting_set)?
                .exist(&self.conjugated_v_edges)?;

            // !edge_starting_set & self.e_move
            let edges_to_outside = self.e_move.diff(&next_edge_starting_set)?;
            // Set of elements which have _no_ edges leading outside our `starting_set`. In other words, all edges point to our attractor set.
            let all_edge_set = opponent_set.diff(&edges_to_outside.exist(&self.conjugated_v_edges)?)?;

            let new_output = output.or(&e_i_attracted_set)?.or(&any_edge_set)?.or(&all_edge_set)?;

            if new_output == output {
                break;
            } else {
                output = new_output;
            }
        }
        
        Ok(output)
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

    /// Return both sets of vertices, with the first element of the returned tuple matching the `player`.
    #[inline(always)]
    pub(crate) fn get_player_sets(&self, player: Owner) -> (&F, &F) {
        match player {
            Owner::Even => (&self.v_even, &self.v_odd),
            Owner::Odd => (&self.v_odd, &self.v_even),
        }
    }

    // pub fn encode_vertex(&self, v_idx: VertexId) -> symbolic::Result<F> {
    //     CachedSymbolicEncoder::<_, F>::encode_impl(&self.variables., v_idx.index())
    // }
    //
    // pub fn encode_edge_vertex(&self, v_idx: VertexId) -> symbolic::Result<F> {
    //     CachedSymbolicEncoder::<_, F>::encode_impl(&self.variables_edges, v_idx.index())
    // }
    //
    // /// Calculate all vertex ids which belong to the set represented by the `bdd`.
    // ///
    // /// Inefficient, and should only be used for debugging.
    // pub fn vertices_of_bdd(&self, bdd: &F) -> Vec<VertexId> {
    //     self.manager.with_manager_shared(|man| {
    //         let valuations = bdd.sat_assignments(man);
    //         crate::symbolic::sat::decode_assignments(valuations, self.variables.len())
    //     })
    // }
}

#[cfg(test)]
mod tests {
    use crate::{
        explicit::ParityGame,
        Owner,
        symbolic::BDD

        ,
    };
    use crate::explicit::register_game::Rank;
    use crate::symbolic::register_game::SymbolicRegisterGame;

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

    //
    // #[test]
    // pub fn test_attraction_set() -> eyre::Result<()> {
    //     let s = symbolic_rg(other_pg()?, 0)?;
    //
    //     let attr_set = s.pg.attractor_set(Owner::Even, &s.pg.priorities.g.vertices[2])?;
    //
    //     let vertices = s.pg.vertices_of_bdd(&attr_set);
    //     assert_eq!(vertices, id_vec![2, 3, 1]);
    //
    //     Ok(())
    // }

    struct SymbolicTest {
        pg: SymbolicRegisterGame<BDD>,
        original: ParityGame,
    }

    fn symbolic_rg(pg: ParityGame, k: Rank) -> eyre::Result<SymbolicTest> {
        let s_pg = super::SymbolicRegisterGame::from_symbolic(&pg, k, Owner::Even)?;


        Ok(SymbolicTest {
            pg: s_pg,
            original: pg,
        })
    }
}
