use crate::explicit::VertexId;
use soa_rs::{Soa, Soars};
use std::fmt::Debug;
use std::hash::Hash;

pub type Priority = u32;

#[derive(Debug, Copy, Clone, serde::Serialize, serde::Deserialize, Ord, PartialOrd, Eq, PartialEq, Hash, Default)]
pub struct NodeIndex<Ix = u32>(Ix);

impl<Ix: IndexType> NodeIndex<Ix> {
    #[inline]
    pub fn new(root: usize) -> Self {
        Self(Ix::from_index(root))
    }

    #[inline]
    pub fn index(self) -> usize {
        self.0.index()
    }
}

impl<Ix: IndexType> From<Ix> for NodeIndex<Ix> {
    fn from(value: Ix) -> Self {
        Self(value)
    }
}

/// Represents a particular player in the parity game.
#[repr(u8)]
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Owner {
    Even = 0x0,
    Odd = 0x1,
}

impl Owner {
    /// Get the player associated with the parity of the given `priority`
    pub fn from_priority(priority: Priority) -> Owner {
        if priority % 2 == 0 {
            Owner::Even
        } else {
            Owner::Odd
        }
    }

    /// Return the opposite player
    pub fn other(&self) -> Owner {
        match self {
            Owner::Even => Owner::Odd,
            Owner::Odd => Owner::Even,
        }
    }

    #[inline]
    pub fn is_even(&self) -> bool {
        matches!(self, Owner::Even)
    }

    #[inline]
    pub fn is_odd(&self) -> bool {
        matches!(self, Owner::Odd)
    }

    /// Check whether the given priority is aligned with this owner
    ///
    /// (E.g., odd priorities match with [Owner::Odd], and even with [Owner::Even]).
    pub fn priority_aligned(&self, priority: u32) -> bool {
        match self {
            Owner::Even => priority % 2 == 0,
            Owner::Odd => priority % 2 == 1,
        }
    }
}

impl TryFrom<u8> for Owner {
    type Error = eyre::Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        Ok(match value {
            0x0 => Owner::Even,
            0x1 => Owner::Odd,
            _ => return Err(eyre::eyre!("`{}` is out of bounds for an Owner variable", value)),
        })
    }
}

pub trait ParityVertexSoa<Ix> {

    #[inline(always)]
    fn priority_of(&self, idx: VertexId<Ix>) -> Priority {
        self.get_priority(idx).unwrap()
    }

    #[inline(always)]
    fn owner_of(&self, idx: VertexId<Ix>) -> Owner {
        self.get_owner(idx).unwrap()
    }

    fn get_priority(&self, idx: VertexId<Ix>) -> Option<Priority>;

    fn get_owner(&self, idx: VertexId<Ix>) -> Option<Owner>;
}

#[derive(Debug, Clone, Copy, Hash, Ord, PartialOrd, Eq, PartialEq, Soars, serde::Deserialize, serde::Serialize)]
#[soa_derive(include(Ref), serde::Serialize)]
pub struct Vertex {
    pub priority: Priority,
    pub owner: Owner,
}

impl<Ix: IndexType> ParityVertexSoa<Ix> for Soa<Vertex> {
    #[inline(always)]
    fn priority_of(&self, idx: VertexId<Ix>) -> Priority {
        unsafe {
            *self.priority().get_unchecked(idx.index())
        }
    }

    #[inline(always)]
    fn owner_of(&self, idx: VertexId<Ix>) -> Owner {
        unsafe {
            *self.owner().get_unchecked(idx.index())
        }
    }
    
    #[inline(always)]
    fn get_priority(&self, idx: VertexId<Ix>) -> Option<Priority> {
        self.priority().get(idx.index()).copied()
    }

    #[inline(always)]
    fn get_owner(&self, idx: VertexId<Ix>) -> Option<Owner> {
        self.owner().get(idx.index()).copied()
    }
}

pub trait IndexType: Copy + PartialEq + PartialOrd + Hash + Default + 'static + Debug {
    fn index(&self) -> usize;
    fn from_index(index: usize) -> Self;
}

macro_rules! index {
    ($($idx_type:ty),*) => {
        $(
        impl IndexType for $idx_type {
            fn index(&self) -> usize {
                *self as usize
            }

            fn from_index(index: usize) -> Self {
                index as Self
            }
        })*
    };
}

index!(u8, u16, u32, usize);
