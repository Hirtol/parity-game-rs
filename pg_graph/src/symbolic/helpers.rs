use oxidd::bdd::BDDFunction;
use oxidd_core::util::{AllocResult, OptBool, OutOfMemory, SatCountCache};
use std::hash::Hash;
use std::collections::hash_map::Entry;
use oxidd_core::function::{BooleanFunction, BooleanFunctionQuant, Function};
use oxidd_core::Manager;

/// Bit-wise encoder of given values
pub struct CachedSymbolicEncoder<T> {
    cache: ahash::HashMap<T, BDDFunction>,
    variables: Vec<BDDFunction>,
}

impl<T> CachedSymbolicEncoder<T>
where
    T: std::ops::BitAnd + std::ops::Shl<Output = T> + Copy + From<u8>,
    T: Eq + Hash,
    <T as std::ops::BitAnd>::Output: PartialEq<T>,
{
    pub fn new(variables: Vec<BDDFunction>) -> Self {
        Self {
            cache: ahash::HashMap::default(),
            variables,
        }
    }

    /// Encode the given value as a [BDDFunction], caching it for future use.
    ///
    /// If `value` was already provided once the previously created [BDDFunction] will be returned.
    pub fn encode(&mut self, value: T) -> super::Result<&BDDFunction> {
        let out = match self.cache.entry(value) {
            Entry::Occupied(val) => val.into_mut(),
            Entry::Vacant(val) => val.insert(Self::encode_impl(&self.variables, value)?),
        };

        Ok(out)
    }

    pub(crate) fn encode_impl(variables: &[BDDFunction], value: T) -> super::Result<BDDFunction> {
        let mut expr = if value & 1.into() != 0.into() {
            variables[0].clone()
        } else {
            variables[0].not()?
        };

        for i in 1..variables.len() {
            // Check if bit is set
            if value & (T::from(1) << T::from(i as u8)) != 0.into() {
                expr = expr.and(&variables[i])?;
            } else {
                expr = expr.and(&variables[i].not()?)?;
            }
        }

        Ok(expr)
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum BddError {
    #[error("Failed to allocate memory")]
    AllocError(OutOfMemory),
    #[error("No input was provided")]
    NoInput
}

impl From<OutOfMemory> for BddError {
    fn from(value: OutOfMemory) -> Self {
        Self::AllocError(value)
    }
}

pub trait BddExtensions {
    /// Conceptually substitute the given `vars` in `self` with `replace_with`.
    ///
    /// `vars` and `replace_with` should, ideally, be individual variables. `replace_with` can't rely on the provided `vars`.
    ///
    /// In practice this (inefficiently) creates a new BDD with: `exists vars. (replace_with <=> vars) && self`.
    fn substitute(&self, var: &BDDFunction, replace_with: &BDDFunction) -> AllocResult<BDDFunction>;
    
    /// TODO: Only partially complete... skips out every choice after the first
    fn sat_valuations(&self) -> ValuationsIterator;
    
    fn sat_quick_count(&self, n_vars: u32) -> u64;
    
    /// Does a sequential substitution (not simultaneous!)
    /// 
    /// See [Self::substitute]
    fn bulk_substitute<'a>(&self, vars: impl IntoIterator<Item=&'a BDDFunction>, replace_vars: impl IntoIterator<Item=&'a BDDFunction>) -> crate::symbolic::Result<BDDFunction> {
        // This turns out to be more efficient than doing a bulk `exist` call on all conjugated `vars`, surprisingly.
        let mut iter = vars.into_iter().zip(replace_vars);
        if let Some((var, replace_with)) = iter.next() {
            let mut accumulator = self.substitute(var, replace_with)?;
            
            for (var, replace_with) in iter {
                accumulator = accumulator.substitute(var, replace_with)?;
            }
            
            Ok(accumulator)
        } else {
            Err(BddError::NoInput)
        }
    }
}

impl BddExtensions for BDDFunction {
    fn substitute(&self, var: &BDDFunction, replace_with: &BDDFunction) -> AllocResult<BDDFunction> {
        var.with_manager_shared(|manager, vars| {
            let iff = Self::equiv_edge(manager, vars, replace_with.as_edge(manager))?;
            let and = Self::and_edge(manager, self.as_edge(manager), &iff)?;
            let exists = Self::exist_edge(manager, &and, vars)?;
        
            manager.drop_edge(iff);
            manager.drop_edge(and);
        
            Ok(Self::from_edge(manager, exists))
        })
    }

    fn sat_valuations(&self) -> ValuationsIterator {
        ValuationsIterator::new(self)
    }

    fn sat_quick_count(&self, n_vars: u32) -> u64 {
        let mut cache: SatCountCache<u64, ahash::RandomState> = SatCountCache::default();
        self.sat_count(n_vars, &mut cache)
    }
}

pub struct ValuationsIterator {
    bit_vec: Option<Vec<OptBool>>,
    choices: Vec<usize>,
    counter: usize,
    total_size: usize,
}

impl ValuationsIterator {
    pub fn new(bdd: &BDDFunction) -> Self {
        let mut choices = vec![];

        let cube = bdd.pick_cube(None, |man, edge| {
            choices.push(man.get_node(edge).level() as usize);
            false
        });

        Self {
            bit_vec: cube,
            total_size: 2_usize.pow(choices.len() as u32),
            choices,
            counter: 0,
        }
    }
}

impl Iterator for ValuationsIterator {
    type Item = Vec<OptBool>;

    fn next(&mut self) -> Option<Self::Item> {
        let Some(bit_vec) = &self.bit_vec else {
            return None;
        };

        if self.counter < self.total_size {
            let mut permutation = bit_vec.clone();
            for (i, &idx) in self.choices.iter().enumerate() {
                permutation[idx] = OptBool::from((self.counter & (1 << i)) != 0);
            }
            self.counter += 1;
            Some(permutation)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use oxidd::bdd::BDDFunction;
    use oxidd_core::function::BooleanFunction;
    use oxidd_core::ManagerRef;
    use oxidd_core::util::OptBool;
    use crate::symbolic::helpers::BddExtensions;

    #[test]
    pub fn test_substitute() -> crate::symbolic::Result<()> {
        let manager = oxidd::bdd::new_manager(0, 0, 12);

        // Construct base building blocks for the BDD
        let base_true = manager.with_manager_exclusive(|man| BDDFunction::t(man));
        let base_false = manager.with_manager_exclusive(|man| BDDFunction::f(man));
        let variables = manager
            .with_manager_exclusive(|man| (0..3).flat_map(|_| BDDFunction::new_var(man)).collect_vec());
        
        let v0_and_v1 = variables[0].and(&variables[1])?;
        let v0_and_v2 = variables[0].and(&variables[2])?;
        
        let res = v0_and_v1.substitute(&variables[1], &variables[2])?;
        
        assert!(res == v0_and_v2);
        
        Ok(())
    }

    // #[test]
    // pub fn test_valuations() -> crate::symbolic::Result<()> {
    //     let manager = oxidd::bdd::new_manager(0, 0, 12);
    //
    //     // Construct base building blocks for the BDD
    //     let base_true = manager.with_manager_exclusive(|man| BDDFunction::t(man));
    //     let base_false = manager.with_manager_exclusive(|man| BDDFunction::f(man));
    //     let variables = manager
    //         .with_manager_exclusive(|man| (0..3).flat_map(|_| BDDFunction::new_var(man)).collect_vec());
    //
    //     let v0_and_v1 = variables[0].and(&variables[1])?;
    //     let v0_and_v2 = variables[0].and(&variables[2])?;
    //     let both = v0_and_v1.or(&v0_and_v2)?;
    //
    //     assert_eq!(v0_and_v1.sat_valuations().collect::<Vec<_>>(), vec![
    //         vec![OptBool::True, OptBool::True, OptBool::None]
    //     ]);
    //     assert_eq!(both.sat_valuations().collect::<Vec<_>>(), vec![
    //         vec![OptBool::True, OptBool::False]
    //     ]);
    //
    //     let res = v0_and_v1.substitute(&variables[1], &variables[2])?;
    //
    //     assert!(res == v0_and_v2);
    //
    //     Ok(())
    // }
}