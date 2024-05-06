use oxidd::bdd::BDDFunction;
use oxidd_core::util::AllocResult;
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
    pub fn encode(&mut self, value: T) -> AllocResult<&BDDFunction> {
        let out = match self.cache.entry(value) {
            Entry::Occupied(val) => val.into_mut(),
            Entry::Vacant(val) => val.insert(Self::encode_impl(&self.variables, value)?),
        };

        Ok(out)
    }

    pub(crate) fn encode_impl(variables: &[BDDFunction], value: T) -> AllocResult<BDDFunction> {
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

pub trait BddExtensions {
    /// Conceptually substitute the given `vars` in `self` for `replace_with`.
    ///
    /// `vars` and `replace_with` can be conjunctions of variables (e.g., `x1 ^ x2 ^ x3 ^...`) to substitute multiple in one go.
    ///
    /// In practice this (inefficiently) creates a new BDD with: `exists vars. (replace_with <=> vars) && self`.
    fn substitute(&self, vars: &BDDFunction, replace_with: &BDDFunction) -> AllocResult<BDDFunction>;
}

impl BddExtensions for BDDFunction {
    fn substitute(&self, vars: &BDDFunction, replace_with: &BDDFunction) -> AllocResult<BDDFunction> {
        vars.with_manager_shared(|manager, vars| {
            let iff = Self::equiv_edge(manager, vars, replace_with.as_edge(manager))?;
            let and = Self::and_edge(manager, self.as_edge(manager), &iff)?;
            let exists = Self::exist_edge(manager, &and, vars)?;

            manager.drop_edge(iff);
            manager.drop_edge(and);

            Ok(Self::from_edge(manager, exists))
        })
    }
}
