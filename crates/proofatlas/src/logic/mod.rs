//! First-order logic representation and manipulation
//!
//! This module provides the fundamental types for representing FOL formulas:
//! terms, literals, clauses, substitutions, and term orderings.

pub mod clause_manager;
pub mod core;
pub mod interner;
pub mod literal_selection;
pub mod ordering;
pub mod unification;

// Re-export commonly used types
pub use core::clause::{CNFFormula, Clause, ClauseDisplay, ClauseKey, ClauseRole};
pub use core::literal::{Atom, AtomDisplay, Literal, LiteralDisplay, PredicateSymbol};
pub use core::position::Position;
pub use core::term::{Constant, FunctionSymbol, Term, TermDisplay, Variable};
pub use interner::{ConstantId, FunctionId, Interner, PredicateId, VariableId};
pub use ordering::{KBOConfig, Ordering as TermOrdering, KBO};
pub use unification::Substitution;
pub use unification::{
    match_term, unify, variable_ids_in_term, variables_in_term, UnificationError,
    UnificationResult,
};
