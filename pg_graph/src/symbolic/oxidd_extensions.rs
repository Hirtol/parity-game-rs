use oxidd::{bcdd::BCDDFunction, bdd::BDDFunction, zbdd::ZBDDFunction};
use oxidd_core::{
    function::{BooleanFunction, Function},
    Manager,
    util::{AllocResult, SatCountCache},
};

use crate::symbolic::sat::{TruthAssignmentsIterator, TruthIteratorHelper};

#[derive(Clone)]
pub struct FunctionVarRef<F> {
    pub func: F,
    pub idx: u32,
}

pub trait BddExtensions: Function + TruthIteratorHelper {
    /// Returns an [Iterator] which results in all possible satisfying assignments of the variables.
    fn sat_assignments<'b, 'a>(&self, manager: &'b Self::Manager<'a>) -> TruthAssignmentsIterator<'b, 'a, Self>;

    fn sat_quick_count(&self, n_vars: u32) -> u64;
}

pub trait BooleanFunctionExtensions: BooleanFunction {
    /// Efficiently compute `self & !rhs`.
    #[inline(always)]
    fn diff(&self, rhs: &Self) -> AllocResult<Self> {
        // `imp_strict` <=> !rhs & self <=> self & !rhs
        rhs.imp_strict(self)
    }

    /// Create a new variable and record the index of the new level in the manager.
    fn new_var_layer_idx(man: &mut Self::Manager<'_>) -> AllocResult<FunctionVarRef<Self>> {
        Ok(FunctionVarRef {
            func: Self::new_var(man)?,
            idx: man.num_levels() - 1,
        })
    }
}

impl<F: BooleanFunction> BooleanFunctionExtensions for F {}

impl<F: TruthIteratorHelper + BooleanFunction> BddExtensions for F {
    fn sat_assignments<'b, 'a>(&self, manager: &'b Self::Manager<'a>) -> TruthAssignmentsIterator<'b, 'a, F> {
        TruthAssignmentsIterator::new(manager, self)
    }

    fn sat_quick_count(&self, n_vars: u32) -> u64 {
        let mut cache: SatCountCache<u64, ahash::RandomState> = SatCountCache::default();
        self.sat_count(n_vars, &mut cache)
    }
}

pub trait FunctionManagerExtension: Function {
    fn new_manager(inner_node_capacity: usize, apply_cache_capacity: usize, threads: u32) -> Self::ManagerRef;
}

impl FunctionManagerExtension for BDDFunction {
    fn new_manager(inner_node_capacity: usize, apply_cache_capacity: usize, threads: u32) -> Self::ManagerRef {
        oxidd::bdd::new_manager(inner_node_capacity, apply_cache_capacity, threads)
    }
}
impl FunctionManagerExtension for BCDDFunction {
    fn new_manager(inner_node_capacity: usize, apply_cache_capacity: usize, threads: u32) -> Self::ManagerRef {
        oxidd::bcdd::new_manager(inner_node_capacity, apply_cache_capacity, threads)
    }
}
impl FunctionManagerExtension for ZBDDFunction {
    fn new_manager(inner_node_capacity: usize, apply_cache_capacity: usize, threads: u32) -> Self::ManagerRef {
        oxidd::zbdd::new_manager(inner_node_capacity, apply_cache_capacity, threads)
    }
}
