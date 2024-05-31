use itertools::Itertools;
use oxidd::{bdd::BDDFunction, BooleanFunction, Manager, ManagerRef};
use oxidd_core::function::{BooleanFunctionQuant, Function, FunctionSubst};
use petgraph::prelude::EdgeRef;

pub use parity_game::SymbolicParityGame;

use crate::{ParityGraph, symbolic::helpers::BddExtensions};

pub mod helpers;
pub mod parity_game;
pub mod solvers;

pub type BDD = BDDFunction;
pub type Result<T> = std::result::Result<T, helpers::BddError>;
