use oxidd_core::{Manager, ManagerRef};
use oxidd_core::function::{BooleanFunction, BooleanFunctionQuant, FunctionSubst};
use oxidd_core::util::Subst;

use crate::{Owner, Priority, symbolic};
use crate::symbolic::oxidd_extensions::{BooleanFunctionExtensions, GeneralBooleanFunction};
use crate::symbolic::register_game::SymbolicRegisterGame;

impl<F: GeneralBooleanFunction> SymbolicRegisterGame<F>
{
    pub fn gc(&self) -> usize {
        self.manager.with_manager_exclusive(|m| m.gc())
    }

    pub fn bdd_node_count(&self) -> usize {
        self.manager.with_manager_exclusive(|m| m.num_inner_nodes())
    }
    
    pub fn edges(&self) -> symbolic::Result<F> {
        Ok(self.e_move.or(&self.e_i_all)?)
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
        let ignored_edge = self.edge_substitute(ignored)?;

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
        if priority == 0 {
            Ok(self.vertices.clone())
        } else {
            self.attractor_set(player, starting_set)
        }
    }

    #[tracing::instrument(level = "trace", skip_all, fields(player))]
    pub fn attractor_set(&self, player: Owner, starting_set: &F) -> symbolic::Result<F> {
        if player == self.controller {
            self.attractor_set_controller(starting_set)
        } else {
            self.attractor_set_opposite_controller(starting_set)
        }
    }

    /// Compute the attractor set for the special case when the target starting set's player matches the controller.
    fn attractor_set_controller(&self, starting_set: &F) -> symbolic::Result<F> {
        let (player_set, opponent_set) = self.get_player_sets(self.controller);
        let mut output = starting_set.clone();
        let e_move_player_set = self.e_move.and(player_set)?;
        let mut last_e_i = None;
        
        loop {
            let edge_starting_set = self.edge_substitute(&output)?;
            // Set of elements which have _any_ E_i edge leading to our `starting_set` of E_move.
            // We know that all `e_i` vertices are owned by the controller, so no need to check for opponent vertices.
            let e_i_attracted_set = self.e_i_all
                .and(&edge_starting_set)?
                .exist(&self.conjugated_v_edges)?;
            
            if let Some(e_i) = last_e_i {
                if e_i == e_i_attracted_set {
                    break;
                } else {
                    last_e_i = Some(e_i_attracted_set.clone());
                }
            } else {
                last_e_i = Some(e_i_attracted_set.clone());
            }

            // The E_move vertices can be owned by either player, so we'll need to do the whole procedure.
            let next_edge_starting_set = self.edge_substitute(&e_i_attracted_set)?.or(&edge_starting_set)?;

            let any_edge_set = e_move_player_set
                .and(&next_edge_starting_set)?
                .exist(&self.conjugated_v_edges)?;

            // !edge_starting_set & self.e_move
            let edges_to_outside = self.e_move.diff(&next_edge_starting_set)?;
            // Set of elements which have _no_ edges leading outside our `starting_set`. In other words, all edges point to our attractor set.
            let all_edge_set = opponent_set.diff(&edges_to_outside.exist(&self.conjugated_v_edges)?)?;
            let e_move_attracted_set = all_edge_set.or(&any_edge_set)?;
            
            let new_output = output.or(&e_i_attracted_set)?.or(&e_move_attracted_set)?;
            
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
        let mut last_e_i = None;

        loop {
            let edge_starting_set = self.edge_substitute(&output)?;
            // Set of elements which have _all_ E_i edge leading to our `starting_set` of E_move.
            // We know that all `e_i` vertices are owned by the controller, and the target set is owned by the opponent.
            let edges_to_outside = self.e_i_all.diff(&edge_starting_set)?;
            // Set of elements which have _no_ edges leading outside our `starting_set`. In other words, all edges point to our attractor set.
            let e_i_attracted_set = opponent_set.diff(&edges_to_outside.exist(&self.conjugated_v_edges)?)?;
            let e_i_attracted_set = e_i_attracted_set.and(&self.variables.next_move_var().not()?)?;
            
            if let Some(e_i) = last_e_i {
                if e_i == e_i_attracted_set {
                    break;
                } else {
                    last_e_i = Some(e_i_attracted_set.clone());
                }
            } else {
                last_e_i = Some(e_i_attracted_set.clone());
            }
            
            // The E_move vertices can be owned by either player, so we'll need to do the whole procedure.
            let next_edge_starting_set = self.edge_substitute(&e_i_attracted_set)?.or(&edge_starting_set)?;

            let any_edge_set = e_move_player_set
                .and(&next_edge_starting_set)?
                .exist(&self.conjugated_v_edges)?;

            // !edge_starting_set & self.e_move
            let edges_to_outside = self.e_move.diff(&next_edge_starting_set)?;
            // Set of elements which have _no_ edges leading outside our `starting_set`. In other words, all edges point to our attractor set.
            let all_edge_set = opponent_set.diff(&edges_to_outside.exist(&self.conjugated_v_edges)?)?;
            let e_move_attracted_set = self.variables.next_move_var().and(&all_edge_set)?.or(&any_edge_set)?;
            
            let new_output = output.or(&e_i_attracted_set)?.or(&e_move_attracted_set)?;
            
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
}

pub mod further_investigation {
    use oxidd_core::function::{BooleanFunction, BooleanFunctionQuant};

    use crate::{Owner, symbolic};
    use crate::symbolic::BDD;
    use crate::symbolic::oxidd_extensions::BooleanFunctionExtensions;
    use crate::symbolic::register_game::SymbolicRegisterGame;

    impl SymbolicRegisterGame<BDD>
    {
        /// This needs further investigation.
        ///
        /// Using the early stopping criterion of `e_move` not attracting any new vertices works, but by default we
        /// (mistakenly) initially didn't return the disjunction of `output || e_i_attracted_vertices`. However,
        /// this not only seemed to produce correct results for all (two_counters/amba_decomposed) games that we tried,
        /// it also reduced the quantity of recursive Zielonka calls considerably, reducing computation time as a result!
        ///
        /// Intuitively this just means that the last set of `e_i` vertices aren't in the returned attractor set, making for more
        /// situations where the `!controller` player has a set with _just_ `E_move` vertices, making for fewer recursive calls?
        /// Why this still results in correct results in the (projected!) game is a mystery pending further investigation.
        pub fn attractor_set_early_stop(&self, player: Owner, starting_set: &BDD) -> symbolic::Result<BDD> {
            let (player_set, opponent_set) = self.get_player_sets(player);
            let mut output = starting_set.clone();
            let edge_move_set = self.e_move.and(player_set)?;
            let edge_i_set = self.e_i_all.and(player_set)?;

            loop {
                let edge_starting_set = self.edge_substitute(&output)?;
                let e_i_attracted_vertices = self.small_attractor(&self.e_i_all, &edge_i_set, opponent_set, &edge_starting_set)?;
                let e_i_attracted_vertices = e_i_attracted_vertices.and(&self.variables.next_move_var().not()?)?;

                let e_i_starting_set = self.edge_substitute(&e_i_attracted_vertices)?.or(&edge_starting_set)?;
                let e_move_attracted_vertices = self.small_attractor(&self.e_move, &edge_move_set, opponent_set, &e_i_starting_set)?;
                let e_move_attracted_vertices = e_move_attracted_vertices.and(self.variables.next_move_var())?;

                // Since we know these two sets alternate, if neither has made any changes we don't need to do another full iteration.
                if e_move_attracted_vertices.diff(&output)? == self.base_false {
                    // Uncomment the below for a normal attractor return.
                    // output = output.or(&e_i_attracted_vertices)?;
                    break;
                } else {
                    output = output.or(&e_move_attracted_vertices)?.or(&e_i_attracted_vertices)?;
                }
            }

            Ok(output)
        }

        /// Attractor set computation which works as a general replacement to the original one.
        #[tracing::instrument(level = "trace", skip_all, fields(player))]
        pub fn attractor_set_general(&self, player: Owner, starting_set: &BDD) -> symbolic::Result<BDD> {
            let (player_set, opponent_set) = self.get_player_sets(player);
            let mut output = starting_set.clone();
            let edge_move_set = self.e_move.and(player_set)?;
            let edge_i_set = self.e_i_all.and(player_set)?;

            loop {
                let edge_starting_set = self.edge_substitute(&output)?;
                let e_i_attracted_vertices = self.small_attractor(&self.e_i_all, &edge_i_set, opponent_set, &edge_starting_set)?;
                let e_i_attracted_vertices = e_i_attracted_vertices.and(&self.variables.next_move_var().not()?)?;

                let e_i_starting_set = self.edge_substitute(&e_i_attracted_vertices)?.or(&edge_starting_set)?;
                let e_move_attracted_vertices = self.small_attractor(&self.e_move, &edge_move_set, opponent_set, &e_i_starting_set)?;
                // The above can return a tautology when `next_move_var` is not set due to inversions, leading to incorrect attractor sets
                // We need the below to re-assert
                let e_move_attracted_vertices = e_move_attracted_vertices.and(self.variables.next_move_var())?;

                let new_output = output.or(&e_move_attracted_vertices)?.or(&e_i_attracted_vertices)?;

                if new_output == output {
                    break;
                } else {
                    output = new_output;
                }
            }

            Ok(output)
        }

        fn small_attractor(&self, edge_set: &BDD, edge_player_set: &BDD, opponent: &BDD, targets: &BDD) -> symbolic::Result<BDD> {
            // Set of elements which have _any_ edge leading to our `starting_set` and are owned by `player`.
            let any_edge_set = edge_player_set
                .and(targets)?
                .exist(&self.conjugated_v_edges)?;

            // !edge_starting_set & self.edges
            let edges_to_outside = edge_set.diff(targets)?;

            // Set of elements which have _no_ edges leading outside our `starting_set`. In other words, all edges point to our attractor set.
            let all_edge_set = opponent.diff(&edges_to_outside.exist(&self.conjugated_v_edges)?)?;

            Ok(any_edge_set.or(&all_edge_set)?)
        }
    }
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
