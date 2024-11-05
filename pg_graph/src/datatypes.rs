use crate::explicit::VertexId;
use petgraph::graph::IndexType;
use soa_rs::{Soa, Soars};

pub type Priority = u32;

/// Represents a particular player in the parity game.
#[repr(u8)]
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
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

    fn priority_slice(&self) -> &[Priority];

    #[inline(always)]
    fn owner_of(&self, idx: VertexId<Ix>) -> Owner {
        self.get_owner(idx).unwrap()
    }

    fn get_priority(&self, idx: VertexId<Ix>) -> Option<Priority>;

    fn get_owner(&self, idx: VertexId<Ix>) -> Option<Owner>;
}

#[derive(Debug, Clone, Copy, Hash, Ord, PartialOrd, Eq, PartialEq, Soars)]
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

    #[inline]
    fn priority_slice(&self) -> &[Priority] {
        self.priority()
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
