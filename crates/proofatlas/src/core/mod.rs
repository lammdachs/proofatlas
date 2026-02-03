//! Core data structures for first-order logic
//!
//! This module provides the fundamental types for representing FOL formulas
//! using standard Rust data structures.

pub mod clause;
pub mod json;
pub mod literal;
pub mod ordering;
pub mod proof;
pub mod substitution;
pub mod term;
pub mod trace;

// Re-export commonly used types
pub use clause::{CNFFormula, Clause, ClauseRole};
pub use literal::{Atom, Literal, PredicateSymbol};
pub use ordering::{KBOConfig, Ordering as TermOrdering, KBO};
pub use proof::{Proof, ProofStep};
pub use substitution::Substitution;
pub use term::{Constant, FunctionSymbol, Term, Variable};
pub use trace::{
    BackwardSimplification, ClauseSimplification, Derivation, ForwardSimplification,
    GeneratingInference, SaturationStep, SaturationTrace, SimplificationOutcome,
};
