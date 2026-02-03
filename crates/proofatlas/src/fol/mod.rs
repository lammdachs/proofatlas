//! First-order logic data structures
//!
//! This module provides the fundamental types for representing FOL formulas:
//! terms, literals, clauses, substitutions, and term orderings.

pub mod clause;
pub mod literal;
pub mod ordering;
pub mod substitution;
pub mod term;

// Re-export commonly used types
pub use clause::{CNFFormula, Clause, ClauseRole};
pub use literal::{Atom, Literal, PredicateSymbol};
pub use ordering::{KBOConfig, Ordering as TermOrdering, KBO};
pub use substitution::Substitution;
pub use term::{Constant, FunctionSymbol, Term, Variable};
