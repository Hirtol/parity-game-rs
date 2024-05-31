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

#[derive(Debug, Clone, Copy, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub struct Vertex {
    pub priority: Priority,
    pub owner: Owner,
}

impl Vertex {
    #[inline]
    pub fn priority_even(&self) -> bool {
        self.priority % 2 == 0
    }

    #[inline]
    pub fn is_even(&self) -> bool {
        matches!(self.owner, Owner::Even)
    }
}
