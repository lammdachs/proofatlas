//! First-order logic data structures
//!
//! This module provides the fundamental types for representing FOL formulas:
//! terms, literals, clauses, substitutions, and term orderings.

pub mod clause;
pub mod interner;
pub mod literal;
pub mod ordering;
pub mod substitution;
pub mod term;

// Re-export commonly used types
pub use clause::{CNFFormula, Clause, ClauseDisplay, ClauseRole};
pub use interner::{ConstantId, FunctionId, Interner, PredicateId, VariableId};
pub use literal::{Atom, AtomDisplay, Literal, LiteralDisplay, PredicateSymbol};
pub use ordering::{KBOConfig, Ordering as TermOrdering, KBO};
pub use substitution::Substitution;
pub use term::{Constant, FunctionSymbol, Term, TermDisplay, Variable};
