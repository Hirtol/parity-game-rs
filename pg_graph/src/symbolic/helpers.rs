use std::{borrow::Borrow, collections::hash_map::Entry, fmt::Debug, hash::Hash};

use ecow::EcoVec;
use oxidd_core::{function::Function, ManagerRef};

use crate::symbolic::{BddError, oxidd_extensions::BooleanFunctionExtensions};

/// Bit-wise encoder of given values
pub struct CachedSymbolicEncoder<T, F> {
    cache: ahash::HashMap<T, F>,
    variables: EcoVec<F>,
    leading_zeros: EcoVec<F>,
}

impl<T, F> CachedSymbolicEncoder<T, F>
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

    /// Encode the given value as a [BDDFunction], caching it for future use.
    ///
    /// If `value` was already provided once the previously created [BDDFunction] will be returned.
    pub fn encode(&mut self, value: T) -> super::Result<&F> {
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

pub struct MultiEncoder<T, F> {
    encoders: Vec<CachedSymbolicEncoder<T, F>>,
}

impl<T, F> MultiEncoder<T, F>
where
    T: std::ops::BitAnd + std::ops::Shl<Output = T> + Copy + From<u8> + BitHelper,
    T: Eq + Hash + Debug,
    <T as std::ops::BitAnd>::Output: PartialEq<T>,
    F: Function + BooleanFunctionExtensions,
{
    pub fn new<I: IntoIterator<Item = F>>(
        manager: &F::ManagerRef,
        variable_slices: impl IntoIterator<Item = I>,
    ) -> Self {
        Self {
            encoders: variable_slices
                .into_iter()
                .map(|slice| CachedSymbolicEncoder::new(manager, slice.into_iter().collect()))
                .collect(),
        }
    }

    /// Encode the given values as a [BDDFunction], where each individual value is potentially cached and `AND`ed together to form the final BDD.
    ///
    /// Note that the entire `value` is not cached.
    pub fn encode_many(&mut self, value: &[T]) -> super::Result<F> {
        let mut it = value.into_iter().enumerate();
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
    ///
    /// Each individual BDD in the result is the partial AND result. The second item in the result is the BDD representing `value[0] ^ value[1]`.
    /// The last item in the array is the same as the result of [Self::encode_many].
    pub fn encode_many_partial_rev(&mut self, value: &[T]) -> super::Result<Vec<F>> {
        let mut result = Vec::with_capacity(value.len());
        let mut it = value.into_iter().enumerate().rev();
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

pub trait BitHelper {
    fn leading_zeros_help(&self) -> u32;
    fn num_bits(&self) -> u32;
}

macro_rules! impl_bithelpers {
    ($($value:ty),*) => {
        $(
            impl BitHelper for $value {
                #[inline(always)]
                fn leading_zeros_help(&self) -> u32 {
                    self.leading_zeros()
                }

                #[inline(always)]
                fn num_bits(&self) -> u32 {
                    Self::BITS
                }
            }
        )*
    };
}

impl_bithelpers!(u8, u16, u32, u64, usize);
