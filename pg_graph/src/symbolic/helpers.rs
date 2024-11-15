use std::marker::PhantomData;
use std::ops::Range;
use std::{borrow::Borrow, collections::hash_map::Entry, fmt::Debug, hash::Hash};

use ecow::EcoVec;
use oxidd_core::util::AllocResult;
use oxidd_core::{function::Function, ManagerRef};

use crate::symbolic::{oxidd_extensions::BooleanFunctionExtensions, BddError};

pub trait SymbolicEncoder<T, F>
    where F: Function {

    /// Encode the given value as a [BDDFunction].
    fn encode(&mut self, value: T) -> super::Result<&F>;
}

/// Bit-wise encoder of given values
pub struct CachedBinaryEncoder<T, F> {
    cache: ahash::HashMap<T, F>,
    variables: EcoVec<F>,
    leading_zeros: EcoVec<F>,
}

impl<T, F> SymbolicEncoder<T, F> for CachedBinaryEncoder<T, F>
    where
        T: std::ops::BitAnd + std::ops::Shl<Output = T> + Copy + From<u8> + BitHelper,
        T: Eq + Hash + Debug,
        <T as std::ops::BitAnd>::Output: PartialEq<T>,
        F: Function + BooleanFunctionExtensions {
    
    /// Encode the given value as a [BDDFunction], caching it for future use.
    ///
    /// If `value` was already provided once the previously created [BDDFunction] will be returned.
    fn encode(&mut self, value: T) -> crate::symbolic::Result<&F> {
        let out = match self.cache.entry(value) {
            Entry::Occupied(val) => val.into_mut(),
            Entry::Vacant(val) => val.insert(Self::efficient_encode_impl(
                &self.leading_zeros,
                &self.variables,
                value,
            )?),
        };

        Ok(out)
    }
}

impl<T, F> CachedBinaryEncoder<T, F>
where
    T: std::ops::BitAnd + std::ops::Shl<Output = T> + Copy + From<u8> + BitHelper,
    T: Eq + Hash + Debug,
    <T as std::ops::BitAnd>::Output: PartialEq<T>,
    F: Function + BooleanFunctionExtensions,
{
    pub fn new(manager: &F::ManagerRef, variables: EcoVec<F>) -> Self {
        // First cache a few BDDs for leading zeros, which allows much faster vertex encoding in 50% of cases.
        let base_true = manager.with_manager_shared(|f| F::t(f));
        let mut leading_zeros_bdds = EcoVec::new();

        for leading_zeros in 0..(variables.len() + 1) {
            let conjugated = variables
                .iter()
                .rev()
                .take(leading_zeros)
                .try_fold(base_true.clone(), |acc, b| acc.diff(b))
                .expect("Not enough memory");

            leading_zeros_bdds.push(conjugated);
        }

        Self {
            cache: ahash::HashMap::default(),
            variables,
            leading_zeros: leading_zeros_bdds,
        }
    }

    /// Perform a binary encoding of the given value.
    ///
    /// Uses a cached `trailing_zeros_fns` to skip a lot of conjugations in 50% of cases.
    pub(crate) fn efficient_encode_impl(leading_zero_fns: &[F], variables: &[F], value: T) -> super::Result<F> {
        let leading_zeros = value.leading_zeros_help();
        let base_subtraction = value.num_bits() - variables.len() as u32;
        let actual_leading_zeros = (leading_zeros - base_subtraction) as usize;

        let mut expr = leading_zero_fns[actual_leading_zeros].clone();
        for i in 0..(variables.len() - actual_leading_zeros) {
            // Check if bit is set
            if value & (T::from(1) << T::from(i as u8)) != 0.into() {
                expr = expr.and(&variables[i])?;
            } else {
                expr = expr.diff(&variables[i])?;
            }
        }

        Ok(expr)
    }

    /// Perform a binary encoding of the given value.
    pub(crate) fn encode_impl(variables: &[F], value: T) -> super::Result<F> {
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
                expr = expr.diff(&variables[i])?;
            }
        }

        Ok(expr)
    }

    pub(crate) fn encode_eval<'a>(
        variables: &'a [F],
        value: T,
    ) -> super::Result<impl Iterator<Item = (&'a F, bool)> + 'a>
    where
        T: 'a,
    {
        Ok(variables
            .iter()
            .enumerate()
            .map(move |(bit_index, variable)| (variable, value & (T::from(1) << T::from(bit_index as u8)) != 0.into())))
    }
}

pub struct MultiEncoder<T, F: Function, E: SymbolicEncoder<T, F>> {
    encoders: Vec<E>,
    _ph: PhantomData<T>,
    __ph: PhantomData<F>
}

impl<T, F> MultiEncoder<T, F, CachedBinaryEncoder<T, F>>
    where
        T: std::ops::BitAnd + std::ops::Shl<Output = T> + Copy + From<u8> + BitHelper,
        T: Eq + Hash + Debug,
        <T as std::ops::BitAnd>::Output: PartialEq<T>,
        F: Function + BooleanFunctionExtensions {
    
    pub fn new<I: IntoIterator<Item = F>>(
        manager: &F::ManagerRef,
        variable_slices: impl IntoIterator<Item = I>,
    ) -> Self {
        Self {
            encoders: variable_slices
                .into_iter()
                .map(|slice| CachedBinaryEncoder::new(manager, slice.into_iter().collect()))
                .collect(),
            _ph: Default::default(),
            __ph: Default::default(),
        }
    }
}

impl<T, F, E> MultiEncoder<T, F, E>
where
    T: std::ops::BitAnd + std::ops::Shl<Output = T> + Copy + From<u8> + BitHelper,
    T: Eq + Hash + Debug,
    <T as std::ops::BitAnd>::Output: PartialEq<T>,
    F: Function + BooleanFunctionExtensions,
    E: SymbolicEncoder<T, F>
{
    pub fn new_collection(
        encoders: impl IntoIterator<Item = E>,
    ) -> Self {
        Self {
            encoders: encoders.into_iter().collect(),
            _ph: Default::default(),
            __ph: Default::default(),
        }
    }
    
    /// Encode the given value as a [BDDFunction], where we use the specific encoder at `index` for the given `value`.
    /// 
    /// The result is fully cached.
    pub fn encode_single(&mut self, index: usize, value: T) -> super::Result<&F> {
        let encoder = self.encoders.get_mut(index).ok_or(BddError::NoInput)?;
        encoder.encode(value)
    }

    /// Encode the given values as a [BDDFunction], where each individual value is potentially cached and `AND`ed together to form the final BDD.
    ///
    /// Note that the entire `value` is not cached.
    pub fn encode_many_range(&mut self, encoder_range: Range<usize>, values: &[T]) -> super::Result<F> {
        let mut it = encoder_range.zip(values);
        let (i, first_item) = it.next().ok_or(BddError::NoInput)?;
        let mut first_encoding = self.encoders[i].encode(*first_item.borrow())?.clone();

        for (i, item) in it {
            let encoder = &mut self.encoders[i];
            first_encoding = first_encoding.and(encoder.encode(*item.borrow())?)?;
        }

        Ok(first_encoding)
    }

    /// Encode the given values as a [BDDFunction], where each individual value is potentially cached and `AND`ed together to form the final BDD.
    ///
    /// Note that the entire `value` is not cached.
    pub fn encode_many(&mut self, value: &[T]) -> super::Result<F> {
        self.encode_many_range(0..value.len(), value)
    }

    /// Encode the given values as a [BDDFunction], where each individual value is potentially cached and `AND`ed together to form the final BDD.
    ///
    /// Note that the entire `value` is not cached.
    ///
    /// Each individual BDD in the result is the partial AND result. The second item in the result is the BDD representing `value[0] ^ value[1]`.
    /// The last item in the array is the same as the result of [Self::encode_many].
    pub fn encode_many_partial_rev(&mut self, value: &[T]) -> super::Result<Vec<F>> {
        let mut result = Vec::with_capacity(value.len());
        let mut it = value.iter().enumerate().rev();
        let (i, last_item) = it.next().ok_or(BddError::NoInput)?;
        let mut first_encoding = self.encoders[i].encode(*last_item.borrow())?.clone();
        result.push(first_encoding.clone());

        for (i, item) in it {
            let encoder = &mut self.encoders[i];
            first_encoding = first_encoding.and(encoder.encode(*item.borrow())?)?;
            result.push(first_encoding.clone());
        }

        Ok(result)
    }
}

#[derive(Debug, Clone, Copy, PartialOrd, PartialEq, Hash, Ord, Eq)]
pub enum Inequality {
    /// Less than or equal to
    Leq,
    /// Greater than
    Gt,
    /// Greater than or equal to
    Geq,
}

pub struct CachedInequalityEncoder<T, F: Function> {
    base_true: F,
    variables: EcoVec<F>,
    value_domain: EcoVec<T>,
    value_encoder: CachedBinaryEncoder<T, F>,
    cache: ahash::HashMap<(Inequality, T), F>,
}

impl<T, F> CachedInequalityEncoder<T, F>
    where
        T: std::ops::BitAnd + std::ops::Shl<Output = T> +  std::ops::Sub<Output = T> + Copy + From<u8> + BitHelper + Sized + Ord,
        T: Eq + Hash + Debug + PartialOrd<T>,
        <T as std::ops::BitAnd>::Output: PartialEq<T>,
        F: Function + BooleanFunctionExtensions {
    /// Create a new [CachedInequalityEncoder].
    ///
    /// Expects `variables` to be sorted least-significant bit to most-significant bit.
    pub fn new(manager: &F::ManagerRef, variables: EcoVec<F>, value_domain: EcoVec<T>) -> Self {
        Self {
            value_encoder: CachedBinaryEncoder::new(manager, variables.clone()),
            variables,
            value_domain,
            base_true: manager.with_manager_exclusive(|man| F::t(man)),
            cache: ahash::HashMap::default(),
        }
    }

    pub fn encode(&mut self, ineq: Inequality, value: T) -> super::Result<&F> {
        match ineq {
            Inequality::Leq => Ok(
                    match self.cache.entry((ineq, value)) {
                        Entry::Occupied(val) => val.into_mut(),
                        Entry::Vacant(val) => {
                            // First calculate how many bits are significant
                            let significant_bits = (value.num_bits() - value.leading_zeros_help()) as usize;
                            // We can exclude all states which are binary wise higher than our current `value`
                            // We still need to exclude special cases such as 3, when value = 2.
                            let exclude_bits = self.variables[significant_bits..]
                                .iter()
                                .try_fold(self.base_true.clone(), |acc, var| acc.diff(var))?;
                            
                            // Assume we have a continuous domain for the estimate.
                            let estimated_items_to_exclude = value.add_one().next_power_of_two_help() - value;
                            // After this point it can become slower to manually exclude values
                            // up to the next power of two. However, with discontinuous domains it frequently results in faster to _solve_
                            // BDDs to construct with explicit exclusion up to the next power of two.
                            let remaining_exclude = if estimated_items_to_exclude.to_usize() > 20 {
                                // We'll need to exclude values larger than priority up to the next power of two to ensure they don't get included
                                if significant_bits > 0 {
                                    Self::recursive_bit_encode_leq(value, significant_bits - 1, &self.variables, &self.base_true)?.and(&exclude_bits)?
                                } else {
                                    exclude_bits
                                }
                            } else {
                                self.value_domain
                                    .iter()
                                    .filter(|&&p| p > value && p < value.add_one().next_power_of_two_help())
                                    .try_fold(exclude_bits, |acc, p| {
                                        acc.diff(self.value_encoder.encode(*p).unwrap())
                                    })?
                            };

                            val.insert(remaining_exclude)
                        }
                    }
                ),
            Inequality::Gt => {
                // Have to avoid multiple mutable borrows....
                if !self.cache.contains_key(&(ineq, value)) {
                    let leq_value = self.encode(Inequality::Leq, value)?.clone();
                    let out = self.cache.entry((ineq, value)).or_insert(leq_value.not_owned()?);
                    Ok(out)
                } else {
                    self.cache.get(&(ineq,value)).ok_or(BddError::NoMatchingInput)
                }
            }
            Inequality::Geq => {
                // Have to avoid multiple mutable borrows....
                if !self.cache.contains_key(&(ineq, value)) {
                    let gt_value = self.encode(Inequality::Gt, value)?.clone();
                    let geq = gt_value.or(self.value_encoder.encode(value)?)?;
                    let out = self.cache.entry((ineq, value)).or_insert(geq);
                    Ok(out)
                } else {
                    self.cache.get(&(ineq,value)).ok_or(BddError::NoMatchingInput)
                }
            }
            // Currently the BDDs we get here are quite large, so just defer to the above for now.
            // We'd have to mirror the explicit exclusion case of LEQ... or just piggyback of it by inverting
            // Inequality::Geq => Ok(
            //     match self.cache.entry((ineq, value)) {
            //         Entry::Occupied(val) => val.into_mut(),
            //         Entry::Vacant(val) => {
            //             // First calculate how many bits are significant
            //             let significant_bits = (value.num_bits() - value.leading_zeros_help()) as usize;
            //             // We must include all states which are binary wise higher than our current `value`
            //             // We just `or` all higher bits together
            //             let include_bits = self.variables[significant_bits..]
            //                 .iter()
            //                 .try_fold(self.base_true.not()?, |acc, var| acc.or(var))?;
            // 
            //             // We'll need to exclude values lower than priority
            //             let geq_inequality = if significant_bits > 0 {
            //                 Self::recursive_bit_encode_geq(value, significant_bits - 1, &self.variables, &self.base_true)?.or(&include_bits)?
            //             } else {
            //                 include_bits.clone()
            //             };
            // 
            // 
            //             val.insert(geq_inequality)
            //         }
            //     }
            // ),
        }
    }

    #[inline]
    fn recursive_bit_encode_leq(value: T, bit: usize, variables: &[F], base_true: &F) -> AllocResult<F> {
        let bit_set = (value & (T::from(1) << T::from(bit as u8))) != T::from(0);
        let bit_var = &variables[bit];
        // End recursion
        if bit == 0 {
            // If `bit_set` then it doesn't matter what `bit_var` is as it will always be <=
            // Only if it's not set _then_ we need to exclude the bit.
            return if bit_set {
                Ok(base_true.clone())
            } else {
                bit_var.not()
            }
        }
        
        let recursive_value = Self::recursive_bit_encode_leq(value, bit - 1, variables, base_true)?;
        if bit_set {
            // We have to encode the following two cases:
            // 1. bit_var == 1 -> formula needs to be `AND`ed with self.recursive_bit_encode_lt(value, bit - 1)
            // 2. bit_var == 0 -> formula should return `true` as it's guaranteed to be smaller than `value`
            bit_var.imp(&recursive_value)
        } else {
            // We have to encode the following two cases:
            // 1. bit_var == 1 -> formula should return `false` as it's guaranteed to be bigger than `value`
            // 2. bit_var == 0 -> formula needs to be `AND`ed with self.recursive_bit_encode_lt(value, bit - 1)
            recursive_value.diff(bit_var)
        }
    }

    #[inline]
    fn recursive_bit_encode_geq(value: T, bit: usize, variables: &[F], base_true: &F) -> AllocResult<F> {
        let bit_set = (value & (T::from(1) << T::from(bit as u8))) != T::from(0);
        let bit_var = &variables[bit];
        // End recursion
        if bit == 0 {
            // If the bit is set _then_ we need to exclude the bit.
            // Otherwise, it doesn't matter.
            return if bit_set {
                Ok(bit_var.clone())
            } else {
                Ok(base_true.clone())
            }
        }

        let recursive_value = Self::recursive_bit_encode_geq(value, bit - 1, variables, base_true)?;
        if bit_set {
            // We have to encode the following two cases:
            // 1. bit_var == 1 -> formula needs to be `AND`ed with self.recursive_bit_encode_lt(value, bit - 1)
            // 2. bit_var == 0 -> formula should return `false` as it's guaranteed to be smaller than `value`
            bit_var.and(&recursive_value)
        } else {
            // We have to encode the following two cases:
            // 1. bit_var == 1 -> formula should return `true` as it's guaranteed to be bigger than `value`
            // 2. bit_var == 0 -> formula needs to be `AND`ed with self.recursive_bit_encode_lt(value, bit - 1)
            bit_var.or(&recursive_value)
        }
    }
}

pub trait BitHelper {
    const NUM_BITS: u32;
    fn leading_zeros_help(&self) -> u32;
    fn num_bits(&self) -> u32;
    
    fn add_one(&self) -> Self;
    
    fn to_usize(&self) -> usize;
    
    fn next_power_of_two_help(&self) -> Self;
}

macro_rules! impl_bithelpers {
    ($($value:ty),*) => {
        $(
            impl BitHelper for $value {
                const NUM_BITS: u32 = Self::BITS;

                #[inline(always)]
                fn leading_zeros_help(&self) -> u32 {
                    self.leading_zeros()
                }

                #[inline(always)]
                fn num_bits(&self) -> u32 {
                    Self::BITS
                }
                
                #[inline(always)]
                fn add_one(&self) -> Self {
                    self + 1
                }
                
                #[inline(always)]
                fn to_usize(&self) -> usize {
                    *self as usize
                }
                
                #[inline(always)]
                fn next_power_of_two_help(&self) -> Self {
                    self.next_power_of_two()
                }
            }
        )*
    };
}

impl_bithelpers!(u8, u16, u32, u64, usize);

#[cfg(test)]
mod tests {
    use crate::explicit::register_game::Rank;
    use crate::explicit::{ParityGame, ParityGraph};
    use crate::symbolic::helpers::{CachedBinaryEncoder, CachedInequalityEncoder, Inequality, SymbolicEncoder};
    use crate::symbolic::oxidd_extensions::FunctionManagerExtension;
    use crate::symbolic::register_game::SymbolicRegisterGame;
    use crate::symbolic::BDD;
    use crate::{load_example, Owner};
    use ecow::EcoVec;
    use oxidd_core::function::BooleanFunction;

    struct SymbolicTest {
        pg: SymbolicRegisterGame<BDD>,
        original: ParityGame,
    }

    fn symbolic_rg(pg: ParityGame, k: Rank) -> eyre::Result<SymbolicTest> {
        let s_pg = SymbolicRegisterGame::from_symbolic(&pg, k, Owner::Even)?;


        Ok(SymbolicTest {
            pg: s_pg,
            original: pg,
        })
    }

    #[test]
    pub fn test_inequality() {
        let spg = symbolic_rg(load_example("two_counters_14.pg"), 2).unwrap();
        let parity_domain = spg.original.priorities_unique().collect::<EcoVec<_>>();
        println!("Parity Domain: {parity_domain:?}");
        let manager = BDD::new_manager(
            100,
            100,
            12,
        );

        let mut cached = CachedInequalityEncoder::new(&spg.pg.manager, spg.pg.variables.all_variables.clone(), parity_domain);
        let mut bin = CachedBinaryEncoder::new(&spg.pg.manager, spg.pg.variables.all_variables.clone());
        let value = bin.encode(4u32).unwrap();
        let ineq = cached.encode(Inequality::Geq, 5).unwrap();

        assert!(!value.and(ineq).unwrap().satisfiable());

        for i in 6..59 {
            let value2 = bin.encode(i).unwrap();
            assert!(value2.and(ineq).unwrap().satisfiable(), "Value: {i} not satisfiable");
        }
        let value3 = bin.encode(5u32).unwrap();
        assert!(value3.and(ineq).unwrap().satisfiable());
    }
}
