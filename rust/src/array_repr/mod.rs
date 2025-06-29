//! Array-based representation of logical formulas for ML integration
//! 
//! This module provides an array-native representation of first-order logic,
//! designed for efficient ML processing and zero-copy transfer to Python.

pub mod types;
pub mod symbol_table;
pub mod builder;
pub mod unification;
pub mod rules;
pub mod saturation;

pub use types::{ArrayProblem, NodeType, EdgeType};
pub use symbol_table::SymbolTable;
pub use builder::ArrayBuilder;