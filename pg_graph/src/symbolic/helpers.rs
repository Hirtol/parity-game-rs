use std::{borrow::Borrow, collections::hash_map::Entry, fmt::Debug, hash::Hash};

use ecow::EcoVec;
use oxidd_core::{function::Function, ManagerRef, util::OptBool};

use crate::{
    explicit::VertexId,
    symbolic::{BddError, oxidd_extensions::BooleanFunctionExtensions},
};

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
                .fold(base_true.clone(), |acc, b| acc.diff(b).unwrap());

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
    pub fn encode_many(&mut self, value: impl IntoIterator<Item = impl Borrow<T>>) -> super::Result<F> {
        let mut it = value.into_iter().enumerate();
        let (i, first_item) = it.next().ok_or(BddError::NoInput)?;
        let mut first_encoding = self.encoders[i].encode(*first_item.borrow())?.clone();

        for (i, item) in it {
            let encoder = &mut self.encoders[i];
            first_encoding = first_encoding.and(encoder.encode(*item.borrow())?)?;
        }

        Ok(first_encoding)
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

pub mod truth_assignments {
    use std::collections::VecDeque;

    use oxidd::bdd::BDDFunction;
    use oxidd_core::{
        Edge,
        function::{EdgeOfFunc, Function},
        HasLevel, InnerNode, Manager, Node, util::{Borrowed, OptBool},
    };
    use oxidd_rules_bdd::simple::BDDTerminal;

    #[inline]
    #[must_use]
    fn collect_children<E: Edge, N: InnerNode<E>>(node: &N) -> (Borrowed<E>, Borrowed<E>) {
        let mut it = node.children();
        let f_then = it.next().unwrap();
        let f_else = it.next().unwrap();
        (f_then, f_else)
    }

    pub struct MidAssignment<'a, F: Function> {
        next_edge: EdgeOfFunc<'a, F>,
        valuation_progress: Vec<OptBool>,
    }

    pub struct TruthAssignmentsIterator<'b, 'a, F: Function> {
        manager: &'b F::Manager<'a>,
        queue: VecDeque<MidAssignment<'a, F>>,
        counter: usize,
    }

    impl<'b, 'a> TruthAssignmentsIterator<'b, 'a, BDDFunction> {
        pub fn new(manager: &'b <BDDFunction as Function>::Manager<'a>, bdd: &BDDFunction) -> Self {
            let base_edge = bdd.as_edge(manager);
            let begin = match manager.get_node(base_edge) {
                Node::Inner(_) => Some(vec![OptBool::None; manager.num_levels() as usize]),
                Node::Terminal(t) => match t {
                    BDDTerminal::False => None,
                    BDDTerminal::True => Some(vec![OptBool::None; manager.num_levels() as usize]),
                },
            }
            .map(|start_valuation| MidAssignment {
                next_edge: manager.clone_edge(base_edge),
                valuation_progress: start_valuation,
            });

            Self {
                manager,
                queue: begin.into_iter().collect(),
                counter: 0,
            }
        }
    }

    impl<'b, 'a> Iterator for TruthAssignmentsIterator<'b, 'a, BDDFunction> {
        type Item = Vec<OptBool>;

        fn next(&mut self) -> Option<Self::Item> {
            let mut next_evaluate = self.queue.pop_back()?;

            let mut next_cube = Some(next_evaluate.next_edge.borrowed());

            while let Some(next) = &next_cube {
                let Node::Inner(node) = self.manager.get_node(next) else {
                    break;
                };

                let (true_edge, false_edge) = collect_children(node);
                let next_edge = if self.manager.get_node(&true_edge).is_terminal(&BDDTerminal::False) {
                    false
                } else if self.manager.get_node(&false_edge).is_terminal(&BDDTerminal::False) {
                    true
                } else {
                    // First queue up the next item (`true`)
                    let mut queued_val = MidAssignment {
                        next_edge: self.manager.clone_edge(&true_edge),
                        valuation_progress: next_evaluate.valuation_progress.clone(),
                    };
                    queued_val.valuation_progress[node.level() as usize] = OptBool::True;
                    self.queue.push_back(queued_val);

                    // Always take the `false` edge first
                    false
                };

                next_evaluate.valuation_progress[node.level() as usize] = OptBool::from(next_edge);
                next_cube = Some(if next_edge { true_edge } else { false_edge })
            }

            self.manager.drop_edge(next_evaluate.next_edge);

            Some(next_evaluate.valuation_progress)
        }
    }
}

/// Turn a sequence of booleans into a sequence of `VertexId`s based on a binary encoding.
///
/// # Arguments
/// * `values` - The binary encoding
/// * `first` - The amount of `bools` to take from each individual `value in values`, this assumes that all vertex
/// variables are first.
pub fn decode_assignments<'a>(values: impl IntoIterator<Item = impl AsRef<[OptBool]>>, first: usize) -> Vec<VertexId> {
    let mut output = Vec::new();

    fn inner(current_value: u32, idx: usize, assignments: &[OptBool], out: &mut Vec<VertexId>) {
        let Some(next) = assignments.get(idx) else {
            out.push(current_value.into());
            return;
        };

        match next {
            OptBool::None => {
                // True
                inner(current_value | (1 << idx), idx + 1, assignments, out);
                // False
                inner(current_value, idx + 1, assignments, out);
            }
            OptBool::False => {
                inner(current_value, idx + 1, assignments, out);
            }
            OptBool::True => {
                inner(current_value | (1 << idx), idx + 1, assignments, out);
            }
        }
    }

    for vertex_assigment in values {
        inner(0, 0, &vertex_assigment.as_ref()[0..first], &mut output)
    }

    output
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use oxidd::bdd::BDDFunction;
    use oxidd_core::{function::BooleanFunction, ManagerRef, util::OptBool};

    use crate::symbolic::{helpers::truth_assignments::TruthAssignmentsIterator, oxidd_extensions::BddExtensions};

    #[test]
    pub fn test_valuations() -> crate::symbolic::Result<()> {
        let manager = oxidd::bdd::new_manager(0, 0, 12);

        // Construct base building blocks for the BDD
        let base_true = manager.with_manager_exclusive(|man| BDDFunction::t(man));
        let base_false = manager.with_manager_exclusive(|man| BDDFunction::f(man));
        let variables =
            manager.with_manager_exclusive(|man| (0..3).flat_map(|_| BDDFunction::new_var(man)).collect_vec());

        let v0_and_v1 = variables[0].and(&variables[1])?;
        let v0_and_v2 = variables[0].and(&variables[2])?;
        let both = v0_and_v1.or(&v0_and_v2)?;
        let multi = variables[0].and(&variables[1].or(&variables[2])?)?;

        manager.with_manager_shared(|man| {
            let iter = TruthAssignmentsIterator::new(man, &v0_and_v1);
            assert_eq!(
                iter.collect::<Vec<_>>(),
                vec![vec![OptBool::True, OptBool::True, OptBool::None]]
            );

            let multi_assignments = multi.sat_assignments(man).collect_vec();
            assert_eq!(
                multi_assignments,
                vec![
                    vec![OptBool::True, OptBool::False, OptBool::True],
                    vec![OptBool::True, OptBool::True, OptBool::None]
                ]
            );
        });

        Ok(())
    }
}
