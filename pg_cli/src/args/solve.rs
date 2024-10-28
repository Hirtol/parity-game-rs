use std::path::PathBuf;

use pg_graph::explicit::reduced_register_game::ReducedRegisterGame;
use pg_graph::{
    explicit::{register_game::RegisterGame, ParityGame, ParityGraph},
    symbolic::{
        register_game::SymbolicRegisterGame,
        register_game_one_hot::OneHotRegisterGame,
        solvers::{
            symbolic_register_zielonka::SymbolicRegisterZielonkaSolver, symbolic_zielonka::SymbolicZielonkaSolver,
        },
        BDD,
    },
    visualize::DotWriter,
    Owner,
};

#[derive(clap::Args, Debug)]
pub struct SolveCommand {
    #[clap(subcommand)]
    solve_type: SolveType,
    /// The `.pg` file to load
    game_path: PathBuf,
    /// Whether to first convert the given game into an explicit expanded `k`-register game.
    ///
    /// Creates a `k`-register-index game, upper bound for valid results is `1 + log(n)`
    /// Provide `k = 0` to use the upper-bound k for correct results.
    /// Default assumes `k = 1 + log(n)`, where `n` is the amount of vertices in the parity game.
    #[clap(short, global = true)]
    register_game_k: Option<u32>,
    /// The controller of the game.
    /// 
    /// Defaults to `Even`.
    #[clap(short, global = true, value_enum, default_value_t)]
    controller: ClapOwner,
    /// Export the solution to a GraphViz graph where the vertices are coloured according to their winner
    /// Green = Even, Red = Odd
    #[clap(short = 'd', global = true)]
    solution_dot: Option<PathBuf>,
    /// Whether to print which vertices are won by which player.
    #[clap(short, global = true)]
    print_solution: bool,
}

#[derive(clap::Subcommand, Debug)]
pub enum SolveType {
    Explicit(ExplicitSolveCommand),
    Symbolic(SymbolicSolveCommand),
}

#[derive(clap::Args, Debug)]
pub struct ExplicitSolveCommand {
    /// Which solver to use, by default will use Zielonka
    #[clap(subcommand)]
    solver: Option<ExplicitSolvers>,
    /// Whether the register game should be reduced or not.
    #[clap(short = 's', value_enum, default_value_t)]
    reduced: RegisterReductionType,
}

#[derive(clap::Args, Debug)]
pub struct SymbolicSolveCommand {
    /// The type of register game instantiation to use.
    ///
    /// Using `Explicit` uses the naive expansion algorithm and subsequently converts that to a symbolic parity game.
    ///
    /// Using `Symbolic` directly constructs the symbolic register game, and converts that to a symbolic parity game.
    #[clap(short = 't', global = true, value_enum, default_value_t)]
    register_game_type: RegisterGameType,
    #[clap(short = 'o', global = true, default_value = "false")]
    one_hot: bool,
    /// Which solver to use, by default will use Zielonka
    #[clap(subcommand)]
    solver: Option<SymbolicSolvers>,
}

#[derive(clap::ValueEnum, Debug, Clone, Default)]
pub enum RegisterGameType {
    Explicit,
    #[default]
    Symbolic,
}

#[derive(clap::ValueEnum, Debug, Clone, Default, PartialEq)]
pub enum RegisterReductionType {
    #[default]
    /// No reduction is done, the full state-space is explored with all vertices and edges created in a game graph
    Normal,
    /// The `E_i` relation is eliminated, reducing the amount of vertices and increasing the amount of edges for k > 0
    PartialReduced,
    /// Nothing is explored and kept, everything is done just-in-time
    Reduced,
}

#[derive(clap::ValueEnum, Debug, Clone, Default)]
pub enum ClapOwner {
    #[default]
    Even,
    Odd
}

impl From<ClapOwner> for Owner {
    fn from(value: ClapOwner) -> Self {
        match value {
            ClapOwner::Even => Owner::Even,
            ClapOwner::Odd => Owner::Odd
        }
    }
}

#[derive(clap::Subcommand, Debug)]
pub enum ExplicitSolvers {
    /// Use the traditional small progress measure algorithm
    Spm,
    /// Use the recursive Zielonka algorithm
    Zielonka,
    /// Use the Priority Promotion algorithm
    PP,
    /// Use Tangle Learning
    TL
}

#[derive(clap::Subcommand, Debug)]
pub enum SymbolicSolvers {
    #[clap(alias = "zie")]
    /// Use the recursive Zielonka algorithm
    Zielonka,
}

macro_rules! timed_solve {
    ($to_run:expr) => {
        timed_solve!($to_run, "Solving done")
    };
    ($to_run:expr, $text:expr) => {
        {
            let now = std::time::Instant::now();
            let out = $to_run;
            tracing::info!(elapsed=?now.elapsed(), $text);
            out
        }
    };
}

impl SolveCommand {
    #[tracing::instrument(name="Solve Parity Game",skip(self), fields(path=?self.game_path))]
    pub fn run(self) -> eyre::Result<()> {
        let parity_game = crate::utils::load_parity_game(&self.game_path)?;

        let solution = match self.solve_type {
            SolveType::Explicit(explicit) => {
                let solver = explicit.solver.unwrap_or(ExplicitSolvers::Zielonka);
                tracing::info!(?solver, "Using explicit solver");

                match self.register_game_k {
                    None => match solver {
                        ExplicitSolvers::Spm => {
                            let mut solver =
                                pg_graph::explicit::solvers::small_progress::SmallProgressSolver::new(&parity_game);

                            timed_solve!(solver.run()).winners
                        }
                        ExplicitSolvers::Zielonka => {
                            let mut solver = pg_graph::explicit::solvers::zielonka::ZielonkaSolver::new(&parity_game);

                            let out = timed_solve!(solver.run());
                            tracing::info!(n = solver.recursive_calls, "Solved with recursive calls");
                            out.winners
                        }
                        ExplicitSolvers::PP => {
                            let mut solver = pg_graph::explicit::solvers::priority_promotion::PPSolver::new(&parity_game);

                            let out = timed_solve!(solver.run());
                            tracing::info!(n = solver.promotions, "Solved with promotions");
                            out.winners
                        }
                        ExplicitSolvers::TL => {
                            let mut solver = pg_graph::explicit::solvers::tangle_learning::TangleSolver::new(&parity_game);

                            let out = timed_solve!(solver.run());
                            tracing::info!(tangles = solver.tangles_found, dominions = solver.dominions_found, "Solved");
                            out.winners
                        }
                    }
                    Some(k) => {
                        let k = k as u8;

                        tracing::debug!(k, "Constructing with register index");

                        match solver {
                            ExplicitSolvers::Spm => {
                                // SPM can't handle the reduced game properly (as it needs to handle the controller vertices)
                                // So we construct the full game instead.
                                let rg = timed_solve!(
                                    RegisterGame::construct_2021(&parity_game, k, self.controller.into()),
                                    "Constructed Register Game"
                                );
                                let rg_pg = rg.to_normal_game()?;

                                tracing::debug!(
                                    from_vertex = rg.original_game.vertex_count(),
                                    to_vertex = rg_pg.vertex_count(),
                                    ratio = rg_pg.vertex_count() / rg.original_game.vertex_count(),
                                    from_edges = rg.original_game.edge_count(),
                                    to_edges = rg_pg.edge_count(),
                                    ratio = rg_pg.edge_count() as f64 / rg.original_game.edge_count() as f64,
                                    "Converted from parity game to register game"
                                );

                                let mut solver =
                                    pg_graph::explicit::solvers::small_progress::SmallProgressSolver::new(&rg_pg);

                                rg.project_winners_original(&timed_solve!(solver.run()).winners)
                            },
                            ExplicitSolvers::PP => {
                                // PP can't handle the reduced game properly (as it needs to handle the controller vertices)
                                // So we construct the full game instead.
                                let rg = timed_solve!(
                                    RegisterGame::construct_2021(&parity_game, k, self.controller.into()),
                                    "Constructed Register Game"
                                );
                                let rg_pg = rg.to_small_game()?;

                                tracing::debug!(
                                    from_vertex = rg.original_game.vertex_count(),
                                    to_vertex = rg_pg.vertex_count(),
                                    ratio = rg_pg.vertex_count() / rg.original_game.vertex_count(),
                                    from_edges = rg.original_game.edge_count(),
                                    to_edges = rg_pg.edge_count(),
                                    ratio = rg_pg.edge_count() as f64 / rg.original_game.edge_count() as f64,
                                    "Converted from parity game to register game"
                                );

                                let mut solver = pg_graph::explicit::solvers::priority_promotion::PPSolver::new(&rg_pg);
                                let out = timed_solve!(solver.run());
                                tracing::info!(n = solver.promotions, "Solved with promotions");
                                
                                rg.project_winners_original(&out.winners)
                            }
                            ExplicitSolvers::Zielonka if explicit.reduced == RegisterReductionType::PartialReduced => {
                                let rg = timed_solve!(
                                    RegisterGame::construct_2021_reduced(&parity_game, k, self.controller.into()),
                                    "Constructed Partial Reduced Register Game"
                                );
                                let rg_pg = rg.to_small_game()?;

                                tracing::debug!(
                                    from_vertex = rg.original_game.vertex_count(),
                                    to_vertex = rg_pg.vertex_count(),
                                    ratio = rg_pg.vertex_count() / rg.original_game.vertex_count(),
                                    from_edges = rg.original_game.edge_count(),
                                    to_edges = rg_pg.edge_count(),
                                    ratio = rg_pg.edge_count() as f64 / rg.original_game.edge_count() as f64,
                                    "Converted from parity game to register game"
                                );
                                let mut solver = pg_graph::explicit::solvers::register_zielonka::ZielonkaSolver::new(&rg_pg, &rg);

                                let solution = timed_solve!(solver.run());
                                tracing::info!(n = solver.recursive_calls, "Solved with recursive calls");
                                rg.project_winners_original(&solution.winners)
                            }
                            ExplicitSolvers::Zielonka if explicit.reduced == RegisterReductionType::Reduced => {
                                let rg = timed_solve!(
                                    ReducedRegisterGame::construct_2021_reduced(&parity_game, k, self.controller.into()),
                                    "Constructed Reduced Register Game"
                                );

                                tracing::debug!(
                                    from_vertex = rg.original_game.vertex_count(),
                                    to_vertex = rg.vertex_count(),
                                    ratio = rg.vertex_count() / rg.original_game.vertex_count(),
                                    from_edges = rg.original_game.edge_count(),
                                    to_edges = rg.edge_count(),
                                    ratio = rg.edge_count() as f64 / rg.original_game.edge_count() as f64,
                                    "Converted from parity game to register game"
                                );
                                let mut solver = pg_graph::explicit::solvers::fully_reduced_reg_zielonka::ZielonkaSolver::new(&rg);

                                let solution = timed_solve!(solver.run());
                                tracing::info!(n = solver.recursive_calls, "Solved with recursive calls");
                                rg.project_winners_original(&solution.winners)
                            }
                            ExplicitSolvers::Zielonka => {
                                let rg = timed_solve!(
                                    RegisterGame::construct_2021(&parity_game, k, self.controller.into()),
                                    "Constructed Register Game"
                                );
                                let rg_pg = rg.to_normal_game()?;

                                tracing::debug!(
                                    from_vertex = rg.original_game.vertex_count(),
                                    to_vertex = rg_pg.vertex_count(),
                                    ratio = rg_pg.vertex_count() / rg.original_game.vertex_count(),
                                    from_edges = rg.original_game.edge_count(),
                                    to_edges = rg_pg.edge_count(),
                                    ratio = rg_pg.edge_count() as f64 / rg.original_game.edge_count() as f64,
                                    "Converted from parity game to register game"
                                );
                                let mut solver = pg_graph::explicit::solvers::zielonka::ZielonkaSolver::new(&rg_pg);

                                let solution = timed_solve!(solver.run());
                                tracing::info!(n = solver.recursive_calls, "Solved with recursive calls");
                                rg.project_winners_original(&solution.winners)
                            }
                            ExplicitSolvers::TL => {
                                unimplemented!();
                            }
                        }
                    }
                }
            }
            SolveType::Symbolic(symbolic) => match symbolic.register_game_type {
                RegisterGameType::Explicit => {
                    let register_game = if let Some(k) = self.register_game_k {
                        let k = k as u8;

                        tracing::debug!(k, "Constructing with register index");
                        let register_game = timed_solve!(
                            RegisterGame::construct_2021(&parity_game, k, self.controller.into()),
                            "Constructed Register Game"
                        );

                        Some((register_game.to_normal_game()?, register_game))
                    } else {
                        None
                    };

                    let solver = symbolic.solver.unwrap_or(SymbolicSolvers::Zielonka);
                    tracing::info!(?solver, "Using symbolic solver");

                    let game_to_solve = if let Some((rg_pg, rg)) = &register_game {
                        tracing::debug!(
                            from_vertex = rg.original_game.vertex_count(),
                            to_vertex = rg_pg.vertex_count(),
                            ratio = rg_pg.vertex_count() / rg.original_game.vertex_count(),
                            from_edges = rg.original_game.edge_count(),
                            to_edges = rg_pg.edge_count(),
                            "Converted from parity game to register game"
                        );
                        rg_pg
                    } else {
                        &parity_game
                    };

                    let symbolic_game = timed_solve!(
                        pg_graph::symbolic::SymbolicParityGame::<BDD>::from_explicit(game_to_solve),
                        "Constructed Symbolic PG"
                    )?;
                    symbolic_game.gc();
                    tracing::debug!(nodes = symbolic_game.bdd_node_count(), "Created symbolic game");
                    

                    let solution = match solver {
                        SymbolicSolvers::Zielonka => {
                            let mut solver = SymbolicZielonkaSolver::new(&symbolic_game);

                            let solution = timed_solve!(solver.run());
                            tracing::info!(n = solver.recursive_calls, "Solved with recursive calls");
                            solution
                        }
                    };

                    if let Some((_, rg)) = register_game {
                        rg.project_winners_original(&solution.winners)
                    } else {
                        solution.winners
                    }
                }
                RegisterGameType::Symbolic => {
                    let solver = symbolic.solver.unwrap_or(SymbolicSolvers::Zielonka);
                    tracing::info!(?solver, "Using symbolic solver");

                    if let Some(k) = self.register_game_k {
                        let k = k as u8;

                        tracing::debug!(k, "Constructing with register index");

                        if symbolic.one_hot {
                            Self::run_srg_one_hot(&parity_game, &solver, k, self.controller.into())?
                        } else {
                            Self::run_srg(&parity_game, &solver, k, self.controller.into())?
                        }
                    } else {
                        let game_to_solve = timed_solve!(
                            pg_graph::symbolic::SymbolicParityGame::<BDD>::from_explicit(&parity_game),
                            "Constructed Symbolic PG"
                        )?;
                        game_to_solve.gc();
                        tracing::debug!(nodes = game_to_solve.bdd_node_count(), "Created symbolic game");

                        let solution = match solver {
                            SymbolicSolvers::Zielonka => {
                                let mut solver = SymbolicZielonkaSolver::new(&game_to_solve);

                                let solution = timed_solve!(solver.run());
                                tracing::info!(n = solver.recursive_calls, "Solved with recursive calls");
                                solution
                            }
                        };

                        solution.winners
                    }
                }
            },
        };

        let dot_output = self
            .solution_dot
            .map(|out| (out, DotWriter::write_dot(&parity_game).unwrap()));

        let (even_wins, odd_wins) = solution.iter().fold((0, 0), |acc, win| match win {
            Owner::Even => (acc.0 + 1, acc.1),
            Owner::Odd => (acc.0, acc.1 + 1),
        });
        tracing::info!(even_wins, odd_wins, "Results");

        if self.print_solution {
            tracing::info!("Solution: {:?}", solution);
        }

        if let Some((out_path, dot)) = dot_output {
            use std::fmt::Write;
            let mut new_dot = dot.strip_suffix("}").unwrap().to_string();
            for (v_id, winner) in solution.iter().enumerate() {
                let fill = match winner {
                    Owner::Even => "#013220",
                    Owner::Odd => "#b10000",
                };
                writeln!(&mut new_dot, "{v_id} [fillcolor = \"{fill}\"]")?;
            }
            write!(&mut new_dot, "}}")?;

            std::fs::write(out_path, new_dot)?;
        }

        Ok(())
    }

    fn run_srg(parity_game: &ParityGame, solver: &SymbolicSolvers, k: u8, controller: Owner) -> eyre::Result<Vec<Owner>> {
        let register_game = timed_solve!(
                                SymbolicRegisterGame::<BDD>::from_symbolic(parity_game, k, controller),
                                "Constructed Register Game"
                            )?;
        let game_to_solve = register_game.to_symbolic_parity_game()?;
        game_to_solve.gc();
        tracing::debug!(
                                from_vertex = parity_game.vertex_count(),
                                rg_vertex = game_to_solve.vertex_count(),
                                rg_bdd_nodes = register_game.bdd_node_count(),
                                ratio = game_to_solve.vertex_count() / parity_game.vertex_count(),
                                "Converted from parity game to symbolic register game"
                            );
        #[cfg(not(feature = "dhat-heap"))]
        tracing::debug!("Current memory usage: {} MB", crate::PEAK_ALLOC.current_usage_as_mb());
        let (w_even, w_odd) = match solver {
            SymbolicSolvers::Zielonka => {
                let mut solver = SymbolicRegisterZielonkaSolver::new(&register_game);
                
                let solution = timed_solve!(solver.run_symbolic());
                tracing::info!(n = solver.recursive_calls, "Solved with recursive calls");
                solution
            }
        };

        let (_, odd) = register_game.project_winning_regions(&w_even, &w_odd)?;
        let mut winners = vec![Owner::Even; parity_game.vertex_count()];
        for v_idx in odd {
            winners[v_idx as usize] = Owner::Odd;
        }

        Ok(winners)
    }

    fn run_srg_one_hot(parity_game: &ParityGame, solver: &SymbolicSolvers, k: u8, controller: Owner) -> eyre::Result<Vec<Owner>> {
        let register_game = timed_solve!(
            OneHotRegisterGame::<BDD>::from_symbolic(&parity_game, k, controller),
            "Constructed Register Game"
        )?;
        let game_to_solve = register_game.to_symbolic_parity_game()?;
        game_to_solve.gc();
        tracing::debug!(
            from_vertex = parity_game.vertex_count(),
            rg_vertex = game_to_solve.vertex_count(),
            rg_bdd_nodes = register_game.bdd_node_count(),
            ratio = game_to_solve.vertex_count() / parity_game.vertex_count(),
            "Converted from parity game to one-hot symbolic register game"
        );

        let (w_even, w_odd) = match solver {
            SymbolicSolvers::Zielonka => {
                let mut solver = SymbolicZielonkaSolver::new(&game_to_solve);

                let solution = timed_solve!(solver.run_symbolic());
                tracing::info!(n = solver.recursive_calls, "Solved with recursive calls");
                solution
            }
        };

        let (_, odd) = register_game.project_winning_regions(&w_even, &w_odd)?;
        let mut winners = vec![Owner::Even; parity_game.vertex_count()];
        for v_idx in odd {
            winners[v_idx as usize] = Owner::Odd;
        }

        Ok(winners)
    }
}
