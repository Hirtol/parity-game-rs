use itertools::Itertools;

use crate::{
    symbolic,
    symbolic::{
        oxidd_extensions::BooleanFunctionExtensions,
        register_game_one_hot::{RegisterLayers, RegisterVertexVars},
    },
};

pub type VariableOrder = ahash::HashMap<RegisterLayers, Vec<usize>>;

#[derive(Clone, Copy, Debug)]
pub struct VariableAllocatorInfo {
    pub n_register_vars: usize,
    pub n_vertex_vars: usize,
    pub n_priority_vars: usize,
    pub n_next_move_vars: usize,
}

impl VariableAllocatorInfo {
    /// Return the amount of vars for ONE set of [RegisterVariables].
    ///
    /// For a symbolic game this will need to be multiplied by `2`.
    pub fn total_vars(&self) -> usize {
        self.n_vertex_vars + self.n_priority_vars + self.n_register_vars + self.n_next_move_vars
    }
}

/// Create the default variable order which gives the smallest overall BDDs.
pub fn default_alloc_vars<F: BooleanFunctionExtensions>(
    man: &mut F::Manager<'_>,
    alloc: VariableAllocatorInfo,
) -> symbolic::Result<(RegisterVertexVars<F>, RegisterVertexVars<F>)> {
    // For converting from symbolic to explicit:
    // let next_move = F::new_var_layer_idx(man)?;
    // let next_move_edge = F::new_var_layer_idx(man)?;
    //
    // let mut vertex_vars = (0..alloc.n_vertex_vars)
    //     .flat_map(|_| F::new_var_layer_idx(man))
    //     .collect_vec();
    //
    // let mut registers = (0..alloc.n_register_vars)
    //     .flat_map(|_| F::new_var_layer_idx(man))
    //     .collect_vec();
    //
    // let mut prio_vars = (0..alloc.n_priority_vars)
    //     .flat_map(|_| F::new_var_layer_idx(man))
    //     .collect_vec();
    //
    // let mut vertex_vars_edge = (0..alloc.n_vertex_vars)
    //     .flat_map(|_| F::new_var_layer_idx(man))
    //     .collect_vec();
    // let mut registers_edge = (0..alloc.n_register_vars)
    //     .flat_map(|_| F::new_var_layer_idx(man))
    //     .collect_vec();
    // let mut prio_vars_edge = (0..alloc.n_priority_vars)
    //     .flat_map(|_| F::new_var_layer_idx(man))
    //     .collect_vec();
    
    // Note that this variable ordering will give _small_ BDDs, but will solve the overall game slower, as the Zielonka algorithm takes longer
    // with this ordering... for some reason. The total memory usage will remain lower though.
    // let next_move = F::new_var_layer_idx(man)?;
    // let mut prio_vars = (0..alloc.n_priority_vars)
    //     .flat_map(|_| F::new_var_layer_idx(man))
    //     .collect_vec();
    //
    // let next_move_edge = F::new_var_layer_idx(man)?;
    // let mut prio_vars_edge = (0..alloc.n_priority_vars)
    //     .flat_map(|_| F::new_var_layer_idx(man))
    //     .collect_vec();
    //
    // let mut vertex_vars = (0..alloc.n_vertex_vars)
    //     .flat_map(|_| F::new_var_layer_idx(man))
    //     .collect_vec();
    // let mut vertex_vars_edge = (0..alloc.n_vertex_vars)
    //     .flat_map(|_| F::new_var_layer_idx(man))
    //     .collect_vec();
    //
    // let mut registers = (0..alloc.n_register_vars)
    //     .flat_map(|_| F::new_var_layer_idx(man))
    //     .collect_vec();
    // let mut registers_edge = (0..alloc.n_register_vars)
    //     .flat_map(|_| F::new_var_layer_idx(man))
    //     .collect_vec();
    // prio_vars.reverse();
    // prio_vars_edge.reverse();
    // vertex_vars.reverse();
    // vertex_vars_edge.reverse();
    // registers.reverse();
    // registers_edge.reverse();
    
    // This variable ordering is optimised for speed on two_counters, but not memory usage!
    // This order: 10.3s solver (19631 nodes), memory efficient order: 28.6s solver (14773 nodes)
    // It seems slower on `amba_decomposed`
    let next_move_edge = F::new_var_layer_idx(man)?;

    let mut vertex_vars = (0..alloc.n_vertex_vars).flat_map(|_| F::new_var_layer_idx(man)).collect_vec();
    let next_move = F::new_var_layer_idx(man)?;

    let mut vertex_vars_edge = (0..alloc.n_vertex_vars).flat_map(|_| F::new_var_layer_idx(man)).collect_vec();
    let mut prio_vars_edge = (0..alloc.n_priority_vars).flat_map(|_| F::new_var_layer_idx(man)).collect_vec();

    // This substantially decreases the size and computation time of BDDs
    let mut registers_edge = Vec::new();
    let mut registers = Vec::new();

    for _ in 0..alloc.n_register_vars {
        registers_edge.push(F::new_var_layer_idx(man)?);
        registers.push(F::new_var_layer_idx(man)?);
    }

    // let mut registers_edge = (0..alloc.n_register_vars)
    //     .flat_map(|_| F::new_var_layer_idx(man))
    //     .collect_vec();
    // let mut registers = (0..alloc.n_register_vars)
    //     .flat_map(|_| F::new_var_layer_idx(man))
    //     .collect_vec();

    let mut prio_vars = (0..alloc.n_priority_vars).flat_map(|_| F::new_var_layer_idx(man)).collect_vec();

    vertex_vars.reverse();
    vertex_vars_edge.reverse();
    prio_vars_edge.reverse();
    registers_edge.reverse();
    registers.reverse();
    prio_vars.reverse();
    Ok((
        RegisterVertexVars::new(
            next_move,
            registers.into_iter(),
            prio_vars.into_iter(),
            vertex_vars.into_iter(),
        ),
        RegisterVertexVars::new(
            next_move_edge,
            registers_edge.into_iter(),
            prio_vars_edge.into_iter(),
            vertex_vars_edge.into_iter(),
        ),
    ))
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, Mutex},
        time::Instant,
    };

    use itertools::Itertools;
    use oxidd_core::{Manager, ManagerRef};
    use rayon::prelude::*;

    use crate::{
        explicit::register_game::Rank,
        Owner,
        symbolic,
        symbolic::{
            BDD,
            oxidd_extensions::BooleanFunctionExtensions,
            register_game_one_hot::{
                RegisterLayers,
                RegisterVertexVars, variable_order::{VariableAllocatorInfo, VariableOrder},
            },
            solvers::symbolic_zielonka::SymbolicZielonkaSolver,
        },
    };
    use crate::symbolic::register_game_one_hot::OneHotRegisterGame;

    // #[test]
    pub fn test_variable_orders() -> eyre::Result<()> {
        rayon::ThreadPoolBuilder::default()
            .num_threads(12)
            .build_global()
            .unwrap();

        let k = 0usize;
        let controller = Owner::Even;
        // let game = crate::tests::trivial_pg_2()?;
        let game = crate::tests::load_example("ActionConverter.tlsf.ehoa.pg");
        
        let srg: OneHotRegisterGame<BDD> = OneHotRegisterGame::from_symbolic(&game, k as Rank, controller).unwrap();
        let total_variables = srg.manager.with_manager_shared(|man| man.num_levels());

        println!("Levels: {total_variables}");
        let order_permutations = (0..total_variables as usize)
            .permutations(total_variables as usize)
            .enumerate();

        let best_node_count = Arc::new(Mutex::new(None));
        let best_order = Arc::new(Mutex::new(None));

        order_permutations.par_bridge().for_each(|(i, perm)| {
            let manager = oxidd::bdd::new_manager(0, 12, 12);
            let mut current_order = None;

            let rg: OneHotRegisterGame<BDD> = OneHotRegisterGame::from_manager(manager, &game, k as Rank, controller, |man, n_variable| {
                let orders = permutation_to_order(n_variable, &perm);
                current_order = Some(orders.clone());

                alloc_var_with_order(man, n_variable, orders.into_iter().tuples().next().expect("Impossible"))
            })
            .unwrap();
            rg.gc();
            let nodes = rg.bdd_node_count();
            tracing::debug!(nodes, "Constructed RG `{i}`");

            let mut best_node_lock = best_node_count.lock().unwrap();
            let mut best_order = best_order.lock().unwrap();

            match (*best_node_lock, best_order.as_ref()) {
                (Some(best_nodes), Some(_)) => {
                    if best_nodes > nodes {
                        // tracing::warn!(nodes, ?orders, "Found new best");
                        println!("New found: {:?} - Order: {:?}", best_node_lock, best_order);
                        *best_node_lock = Some(nodes);
                        *best_order = current_order;
                    }
                }
                (None, None) => {
                    *best_node_lock = Some(nodes);
                    *best_order = current_order;
                }
                _ => panic!("Impossible"),
            }
        });

        println!("Best found: {:?} - Order: {:#?}", best_node_count, best_order);

        Ok(())
    }

    // #[tracing_test::traced_test]
    // #[test]
    pub fn test_variable_orders_grouped() -> eyre::Result<()> {
        rayon::ThreadPoolBuilder::default()
            .num_threads(12)
            .build_global()
            .unwrap();

        let k = 2usize;
        let controller = Owner::Even;
        let game = crate::tests::load_example("two_counters_1.pg");

        let order_permutations = (0..8_usize).permutations(8).enumerate();

        let best_node_count = Arc::new(Mutex::new(None));
        let best_order = Arc::new(Mutex::new(None));

        order_permutations.par_bridge().for_each(|(i, perm)| {
            let manager = oxidd::bdd::new_manager(0, 12, 12);
            let mut current_order = None;

            let rg: OneHotRegisterGame<BDD> = OneHotRegisterGame::from_manager(manager, &game, k as Rank, controller, |man, n_variable| {
                let orders = permutation_to_order_grouped(n_variable, &perm);
                current_order = Some(orders.clone());

                alloc_var_with_order(man, n_variable, orders.into_iter().tuples().next().expect("Impossible"))
            })
            .unwrap();
            rg.gc();
            let nodes = rg.bdd_node_count();
            tracing::debug!(nodes, "Constructed RG `{i}`");

            let mut best_node_lock = best_node_count.lock().unwrap();
            let mut best_order = best_order.lock().unwrap();

            match (*best_node_lock, best_order.as_ref()) {
                (Some(best_nodes), Some(_)) => {
                    if best_nodes > nodes {
                        *best_node_lock = Some(nodes);
                        *best_order = current_order;
                        println!("New found: {:?} - Order: {:?}", best_node_lock, best_order);
                    }
                }
                (None, None) => {
                    *best_node_lock = Some(nodes);
                    *best_order = current_order;
                    println!("New found: {:?} - Order: {:?}", best_node_lock, best_order);
                }
                _ => panic!("Impossible"),
            }
        });

        println!("Best found: {:?} - Order: {:#?}", best_node_count, best_order);

        Ok(())
    }

    // #[test]
    pub fn test_variable_orders_grouped_time_taken() -> eyre::Result<()> {
        rayon::ThreadPoolBuilder::default()
            .num_threads(4)
            .build_global()
            .unwrap();

        let k = 2usize;
        let controller = Owner::Even;
        let game = crate::tests::load_example("two_counters_1.pg");

        let order_permutations = (0..8_usize).permutations(8).enumerate();

        let best_solve_time = Arc::new(Mutex::new(None));
        let best_order = Arc::new(Mutex::new(None));

        order_permutations.par_bridge().for_each(|(i, perm)| {
            let manager = oxidd::bdd::new_manager(0, 12, 12);

            let mut current_order = None;
            let rg: OneHotRegisterGame<BDD> = OneHotRegisterGame::from_manager(manager, &game, k as Rank, controller, |man, n_variable| {
                let orders = permutation_to_order_grouped(n_variable, &perm);
                current_order = Some(orders.clone());

                alloc_var_with_order(man, n_variable, orders.into_iter().tuples().next().expect("Impossible"))
            })
            .unwrap();

            rg.gc();
            let nodes = rg.bdd_node_count();
            tracing::debug!(nodes, "Constructed RG `{i}`");

            let spg = rg.to_symbolic_parity_game().unwrap();
            let mut solver = SymbolicZielonkaSolver::new(&spg);
            let now = Instant::now();
            let _ = solver.run_symbolic();
            let solve_time = now.elapsed();

            let mut best_solve_time_lock = best_solve_time.lock().unwrap();
            let mut best_order = best_order.lock().unwrap();

            match (*best_solve_time_lock, best_order.as_ref()) {
                (Some(best_solve), Some(_)) => {
                    if best_solve > solve_time {
                        println!("New found: {:?} - Order: {:?}", best_solve_time_lock, best_order);
                        *best_solve_time_lock = Some(solve_time);
                        *best_order = current_order;
                    }
                }
                (None, None) => {
                    *best_solve_time_lock = Some(solve_time);
                    *best_order = current_order;
                    println!("New found: {:?} - Order: {:?}", best_solve_time_lock, best_order);
                }
                _ => panic!("Impossible"),
            }
        });

        println!("Best found: {:?} - Order: {:#?}", best_solve_time, best_order);

        Ok(())
    }

    fn alloc_var_with_order<F: BooleanFunctionExtensions>(
        man: &mut F::Manager<'_>,
        alloc: VariableAllocatorInfo,
        order: (VariableOrder, VariableOrder),
    ) -> symbolic::Result<(RegisterVertexVars<F>, RegisterVertexVars<F>)> {
        // multiply by 2 for
        let variables = (0..alloc.total_vars() * 2)
            .flat_map(|_| F::new_var_layer_idx(man))
            .collect_vec();

        macro_rules! dis {
            ($part:tt, $var_type:expr) => {{
                let window = order.$part.get(&$var_type).unwrap();
                crate::symbolic::sat::DiscontiguousArrayIterator::new(&variables, window.iter().copied()).cloned()
            }};
        }

        Ok((
            RegisterVertexVars::new(
                dis!(0, RegisterLayers::NextMove).next().unwrap(),
                dis!(0, RegisterLayers::Registers),
                dis!(0, RegisterLayers::Priority),
                dis!(0, RegisterLayers::Vertex),
            ),
            RegisterVertexVars::new(
                dis!(1, RegisterLayers::NextMove).next().unwrap(),
                dis!(1, RegisterLayers::Registers),
                dis!(1, RegisterLayers::Priority),
                dis!(1, RegisterLayers::Vertex),
            ),
        ))
    }

    fn permutation_to_order_grouped(vars: VariableAllocatorInfo, perm: &[usize]) -> Vec<VariableOrder> {
        // first item is the order for the first variables set, second is for next variables.
        let mut orders = vec![ahash::HashMap::default(); 2];
        let var_iter = perm.iter().copied();
        let mut var_counter = 0;

        let VariableAllocatorInfo {
            n_register_vars,
            n_vertex_vars,
            n_priority_vars,
            ..
        } = vars;

        for (i, group_id) in var_iter.enumerate() {
            let order_set_id = group_id / 4;
            let hash_set = &mut orders[order_set_id];
            let register_layer = group_id % 4;
            match register_layer {
                0 => {
                    hash_set.insert(RegisterLayers::NextMove, vec![var_counter]);
                    var_counter += 1;
                }
                1 => {
                    let contents = (var_counter..).take(n_register_vars).collect();
                    hash_set.insert(RegisterLayers::Registers, contents);
                    var_counter += n_register_vars;
                }
                2 => {
                    let contents: Vec<usize> = (var_counter..).take(n_vertex_vars).collect();
                    hash_set.insert(RegisterLayers::Vertex, contents);
                    var_counter += n_vertex_vars;
                }
                3 => {
                    let contents = (var_counter..).take(n_priority_vars).collect();
                    hash_set.insert(RegisterLayers::Priority, contents);
                    var_counter += n_priority_vars;
                }
                _ => unreachable!(),
            }
        }
        orders
    }

    fn permutation_to_order(
        vars: VariableAllocatorInfo,
        perm: &[usize],
    ) -> Vec<ahash::HashMap<RegisterLayers, Vec<usize>>> {
        let VariableAllocatorInfo {
            n_register_vars,
            n_vertex_vars,
            n_priority_vars,
            ..
        } = vars;
        // first item is the order for the first variables set, second is for next variables.
        let mut orders = vec![ahash::HashMap::default(); 2];
        let mut var_iter = perm.iter().copied();

        orders[0].insert(RegisterLayers::NextMove, vec![var_iter.next().unwrap()]);
        orders[1].insert(RegisterLayers::NextMove, vec![var_iter.next().unwrap()]);

        let reg_vars = (0..n_register_vars).flat_map(|_| var_iter.next()).collect_vec();
        let next_reg_vars = (0..n_register_vars).flat_map(|_| var_iter.next()).collect_vec();

        orders[0].insert(RegisterLayers::Registers, reg_vars);
        orders[1].insert(RegisterLayers::Registers, next_reg_vars);

        let vertex_vars = (0..n_vertex_vars).flat_map(|_| var_iter.next()).collect_vec();
        let next_vertex_vars = (0..n_vertex_vars).flat_map(|_| var_iter.next()).collect_vec();

        orders[0].insert(RegisterLayers::Vertex, vertex_vars);
        orders[1].insert(RegisterLayers::Vertex, next_vertex_vars);

        let prio_vars = (0..n_priority_vars).flat_map(|_| var_iter.next()).collect_vec();
        let next_prio_vars = (0..n_priority_vars).flat_map(|_| var_iter.next()).collect_vec();

        orders[0].insert(RegisterLayers::Priority, prio_vars);
        orders[1].insert(RegisterLayers::Priority, next_prio_vars);
        orders
    }
}
