use eyre::Context;
use pg_graph::{
    explicit::{reduced_register_game::ReducedRegisterGame, register_game::RegisterGame, ParityGame, ParityGraph},
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
use std::{
    fmt::{Formatter, Write},
    path::PathBuf,
    str::FromStr,
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
    /// If left empty the parametrised form of Lehtinen's algorithm will be used to solve the game correctly.
    #[clap(short, global = true)]
    register_game_k: Option<RegisterGameIndex>,
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

#[derive(Debug, Clone, Copy)]
enum RegisterGameIndex {
    /// Create a register game with the given `k`
    Number(u32),
    /// Run the parametrised form of Lehtinen's algorithm, and report the discovered Register index
    Parametrised,
}

impl FromStr for RegisterGameIndex {
    type Err = eyre::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse::<u32>()
            .map(Self::Number)
            .context("Could not parse register index")
            .or_else(|_| {
                if s == "param" {
                    Ok(Self::Parametrised)
                } else {
                    Err(eyre::eyre!(
                        "Either provide a register-index as a number, or 'param' to run the parametrised version"
                    ))
                }
            })
    }
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

#[derive(clap::ValueEnum, Debug, Clone, Default, PartialEq, Copy)]
pub enum RegisterReductionType {
    #[default]
    /// No reduction is done, the full state-space is explored with all vertices and edges created in a game graph
    Normal,
    /// The `E_i` relation is eliminated, reducing the amount of vertices and increasing the amount of edges for k > 0
    PartialReduced,
    /// Nothing is explored and kept, everything is done just-in-time
    Reduced,
}

#[derive(clap::ValueEnum, Debug, Clone, Default, Copy)]
pub enum ClapOwner {
    #[default]
    Even,
    Odd,
}

impl From<ClapOwner> for Owner {
    fn from(value: ClapOwner) -> Self {
        match value {
            ClapOwner::Even => Owner::Even,
            ClapOwner::Odd => Owner::Odd,
        }
    }
}

#[derive(clap::Subcommand, Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum ExplicitSolvers {
    /// Use the traditional small progress measure algorithm
    Spm,
    /// Use the recursive Zielonka algorithm
    Zielonka,
    /// Use the quasi-polynomial Zielonka algorithm
    QZielonka {
        /// Enable experimental tangle support.
        #[clap(long)]
        tangles: bool,
    },
    /// Liverpool quasi-polynomial Zielonka
    Qlz {
        /// Enable experimental tangle support.
        #[clap(long)]
        tangles: bool,
    },
    /// Use the Priority Promotion algorithm
    PP,
    /// Use Tangle Learning
    TL,
}

impl std::fmt::Display for ExplicitSolvers {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ExplicitSolvers::Spm => f.write_str("spm"),
            ExplicitSolvers::Zielonka => f.write_str("zielonka"),
            ExplicitSolvers::QZielonka { tangles } => {
                if *tangles {
                    f.write_str("qwzt")
                } else {
                    f.write_str("qwz")
                }
            }
            ExplicitSolvers::Qlz { tangles } => {
                if *tangles {
                    f.write_str("qlzt")
                } else {
                    f.write_str("qlz")
                }
            }
            ExplicitSolvers::PP => f.write_str("pp"),
            ExplicitSolvers::TL => f.write_str("tl"),
        }
    }
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
        let parity_game = pg_graph::load_parity_game(&self.game_path)?;

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
                        ExplicitSolvers::QZielonka { tangles } => {
                            if tangles {
                                let mut solver =
                                    pg_graph::explicit::solvers::qpt_tangle_zielonka::ZielonkaSolver::new(&parity_game);

                                let out = timed_solve!(solver.run());
                                tracing::info!(n = solver.iterations, "Solved with iterations");
                                out.winners
                            } else {
                                let mut solver =
                                    pg_graph::explicit::solvers::qpt_zielonka::ZielonkaSolver::new(&parity_game);

                                let out = timed_solve!(solver.run());
                                tracing::info!(n = solver.iterations, "Solved with iterations");
                                out.winners
                            }
                        }
                        ExplicitSolvers::Qlz { tangles } => {
                            if tangles {
                                let mut solver =
                                    pg_graph::explicit::solvers::qpt_tangle_liverpool::TLZSolver::new(&parity_game);

                                let out = timed_solve!(solver.run());
                                tracing::info!(n = solver.iterations, "Solved with iterations");
                                out.winners
                            } else {
                                let mut solver =
                                    pg_graph::explicit::solvers::qpt_liverpool::LiverpoolSolver::new(&parity_game);

                                let out = timed_solve!(solver.run());
                                tracing::info!(n = solver.iterations, "Solved with iterations");
                                out.winners
                            }
                        }
                        ExplicitSolvers::PP => {
                            let mut solver =
                                pg_graph::explicit::solvers::priority_promotion::PPSolver::new(&parity_game);

                            let out = timed_solve!(solver.run());
                            tracing::info!(n = solver.promotions, "Solved with promotions");
                            out.winners
                        }
                        ExplicitSolvers::TL => {
                            let mut solver =
                                pg_graph::explicit::solvers::tangle_learning::TangleSolver::new(&parity_game);

                            let out = timed_solve!(solver.run());
                            tracing::info!(
                                tangles = solver.tangles.tangles_found,
                                dominions = solver.tangles.dominions_found,
                                iterations = solver.iterations,
                                "Solved"
                            );
                            out.winners
                        }
                    },
                    Some(k) => match k {
                        RegisterGameIndex::Number(k) => Self::run_erg(
                            &parity_game,
                            &solver,
                            k as u8,
                            &explicit.reduced,
                            self.controller.into(),
                        )?,
                        RegisterGameIndex::Parametrised => {
                            tracing::debug!("Running parametrised Lehtinen");
                            let max_k = RegisterGame::max_register_index(&parity_game);
                            let mut output = None;

                            timed_solve! {
                                for i in 0..max_k {
                                    tracing::debug!(k=i, owner=?Owner::Even, "Constructing with register index");
                                    let even_sol = Self::run_erg(&parity_game, &solver, i, &explicit.reduced, Owner::Even)?;
                                    tracing::debug!(k=i, owner=?Owner::Odd, "Constructing with register index");
                                    let odd_sol = Self::run_erg(&parity_game, &solver, i, &explicit.reduced, Owner::Odd)?;

                                    if even_sol == odd_sol {
                                        tracing::info!(k=i, "Discovered register index");
                                        output = Some(even_sol);
                                        break;
                                    }
                                }, "Finished running parametrised Lehtinen"
                            }

                            output.expect("Could not find a solution within the k bounds")
                        }
                    },
                }
            }
            SolveType::Symbolic(symbolic) => {
                let solver = symbolic.solver.unwrap_or(SymbolicSolvers::Zielonka);
                tracing::info!(?solver, "Using symbolic solver");

                match self.register_game_k {
                    None => {
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
                    Some(k) => match k {
                        RegisterGameIndex::Number(k) => {
                            let k = k as u8;

                            tracing::debug!(k, "Constructing with register index");

                            if symbolic.one_hot {
                                Self::run_srg_one_hot(&parity_game, &solver, k, self.controller.into())?
                            } else {
                                Self::run_srg(&parity_game, &solver, k, self.controller.into())?
                            }
                        }
                        RegisterGameIndex::Parametrised => {
                            tracing::debug!("Running parametrised Lehtinen");
                            let max_k = RegisterGame::max_register_index(&parity_game);
                            let mut output = None;

                            timed_solve! {
                                for i in 0..max_k {
                                    tracing::debug!(k=i, owner=?Owner::Even, "Constructing with register index");
                                    let even_sol = Self::run_srg(&parity_game, &solver, i, Owner::Even)?;
                                    tracing::debug!(k=i, owner=?Owner::Odd, "Constructing with register index");
                                    let odd_sol = Self::run_srg(&parity_game, &solver, i, Owner::Odd)?;
    
                                    if even_sol == odd_sol {
                                        tracing::info!(k = i, "Discovered register index");
                                        output = Some(even_sol);
                                        break;
                                    }
                                }, "Finished running parametrised Lehtinen" 
                            }

                            output.expect("Could not find a solution within the k bounds")
                        }
                    },
                }
            }
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

    fn run_erg(
        parity_game: &ParityGame,
        solver: &ExplicitSolvers,
        k: u8,
        reduced: &RegisterReductionType,
        controller: Owner,
    ) -> eyre::Result<Vec<Owner>> {
        macro_rules! make_rg_pg {
            () => {{
                let rg = timed_solve!(
                    RegisterGame::construct_2021(&parity_game, k, controller.into()),
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
                (rg, rg_pg)
            }};
        }
        
        Ok(match solver {
            ExplicitSolvers::Spm => {
                // SPM can't handle the reduced game properly (as it needs to handle the controller vertices)
                // So we construct the full game instead.
                let (rg, rg_pg) = make_rg_pg!();

                let mut solver = pg_graph::explicit::solvers::small_progress::SmallProgressSolver::new(&rg_pg);

                rg.project_winners_original(&timed_solve!(solver.run()).winners)
            }
            ExplicitSolvers::PP => {
                // PP can't handle the reduced game properly (as it needs to handle the controller vertices)
                // So we construct the full game instead.
                let (rg, rg_pg) = make_rg_pg!();

                let mut solver = pg_graph::explicit::solvers::priority_promotion::PPSolver::new(&rg_pg);
                let out = timed_solve!(solver.run());
                tracing::info!(n = solver.promotions, "Solved with promotions");

                rg.project_winners_original(&out.winners)
            }
            ExplicitSolvers::TL => {
                let (rg, rg_pg) = make_rg_pg!();

                let mut solver = pg_graph::explicit::solvers::tangle_learning::TangleSolver::new(&rg_pg);
                let out = timed_solve!(solver.run());
                tracing::info!(
                    tangles = solver.tangles.tangles_found,
                    dominions = solver.tangles.dominions_found,
                    iterations = solver.iterations,
                    "Solved"
                );

                rg.project_winners_original(&out.winners)
            }
            ExplicitSolvers::QZielonka { tangles } => {
                let (rg, rg_pg) = make_rg_pg!();

                if *tangles {
                    let mut solver = pg_graph::explicit::solvers::qpt_tangle_zielonka::ZielonkaSolver::new(&rg_pg);

                    let out = timed_solve!(solver.run());
                    tracing::info!(n = solver.iterations, "Solved with iterations");
                    rg.project_winners_original(&out.winners)
                } else {
                    let mut solver = pg_graph::explicit::solvers::qpt_zielonka::ZielonkaSolver::new(&rg_pg);

                    let out = timed_solve!(solver.run());
                    tracing::info!(n = solver.iterations, "Solved with iterations");
                    rg.project_winners_original(&out.winners)
                }
            }
            ExplicitSolvers::Qlz { tangles } => {
                let (rg, rg_pg) = make_rg_pg!();

                if *tangles {
                    let mut solver = pg_graph::explicit::solvers::qpt_tangle_liverpool::TLZSolver::new(&rg_pg);

                    let out = timed_solve!(solver.run());
                    tracing::info!(n = solver.iterations, "Solved with iterations");
                    rg.project_winners_original(&out.winners)
                } else {
                    let mut solver = pg_graph::explicit::solvers::qpt_liverpool::LiverpoolSolver::new(&rg_pg);

                    let out = timed_solve!(solver.run());
                    tracing::info!(n = solver.iterations, "Solved with iterations");
                    rg.project_winners_original(&out.winners)
                }
            }
            ExplicitSolvers::Zielonka if *reduced == RegisterReductionType::PartialReduced => {
                let rg = timed_solve!(
                    RegisterGame::construct_2021_reduced(parity_game, k, controller),
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
            ExplicitSolvers::Zielonka if *reduced == RegisterReductionType::Reduced => {
                let rg = timed_solve!(
                    ReducedRegisterGame::construct_2021_reduced(parity_game, k, controller),
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
                let (rg, rg_pg) = make_rg_pg!();
                let mut solver = pg_graph::explicit::solvers::zielonka::ZielonkaSolver::new(&rg_pg);

                let solution = timed_solve!(solver.run());
                tracing::info!(n = solver.recursive_calls, "Solved with recursive calls");
                rg.project_winners_original(&solution.winners)
            }
        })
    }

    fn run_srg(
        parity_game: &ParityGame,
        solver: &SymbolicSolvers,
        k: u8,
        controller: Owner,
    ) -> eyre::Result<Vec<Owner>> {
        let register_game = timed_solve!(
            SymbolicRegisterGame::<BDD>::from_symbolic(parity_game, k, controller),
            "Constructed Register Game"
        )?;
        let game_to_solve = timed_solve!(
            {
                let game_to_solve = register_game.to_symbolic_parity_game()?;
                game_to_solve.gc();
                game_to_solve
            },
            "Garbage collection"
        );
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

    fn run_srg_one_hot(
        parity_game: &ParityGame,
        solver: &SymbolicSolvers,
        k: u8,
        controller: Owner,
    ) -> eyre::Result<Vec<Owner>> {
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
