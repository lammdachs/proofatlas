//! Core data structures for first-order logic
//! 
//! This module provides the fundamental types for representing FOL formulas
//! using standard Rust data structures.

pub mod term;
pub mod literal;
pub mod clause;
pub mod substitution;
pub mod proof;
pub mod ordering;
pub mod proof_event;


// Re-export commonly used types
pub use term::{Term, Variable, Constant, FunctionSymbol};
pub use literal::{Literal, Atom, PredicateSymbol};
pub use clause::{Clause, CNFFormula};
pub use substitution::Substitution;
pub use proof::{Proof, ProofStep};
pub use ordering::{KBO, KBOConfig, Ordering as TermOrdering};
pub use proof_event::{ProofEvent, SelectionType, DiscardReason};