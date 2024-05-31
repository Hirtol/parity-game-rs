use oxidd::{bcdd::BCDDFunction, bdd::BDDFunction, zbdd::ZBDDFunction};
use oxidd_core::{
    function::{BooleanFunction, Function},
    util::{AllocResult, SatCountCache},
};

use crate::symbolic::helpers::truth_assignments::TruthAssignmentsIterator;

pub trait BddExtensions: Function {
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
}

impl<F: BooleanFunction> BooleanFunctionExtensions for F {}

impl BddExtensions for BDDFunction {
    fn sat_assignments<'b, 'a>(&self, manager: &'b Self::Manager<'a>) -> TruthAssignmentsIterator<'b, 'a, BDDFunction> {
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
