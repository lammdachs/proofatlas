//! ProofAtlas: A high-performance theorem prover for first-order logic
//!
//! This library provides a complete implementation of a theorem prover
//! using the superposition calculus with equality.

pub mod core;
pub mod inference;
pub mod ml;
pub mod parser;
pub mod saturation;
pub mod selection;
pub mod time_compat;
pub mod unification;

#[cfg(feature = "python")]
pub mod python_bindings;

// Re-export commonly used types and functions
pub use core::{
    Atom, CNFFormula, Clause, Constant, FunctionSymbol, KBOConfig, Literal, PredicateSymbol, Proof,
    ProofStep, Substitution, Term, TermOrdering, Variable, KBO,
};

pub use inference::{
    equality_factoring, equality_resolution, factoring, resolution, superposition, InferenceResult,
    InferenceRule,
};

pub use selection::{
    AgeBasedSelector, AgeWeightRatioSelector, ClauseSelector, LiteralSelector, SelectAll,
    SelectMaxWeight, SizeBasedSelector,
};

pub use saturation::{
    saturate, saturate_with_steps, LiteralSelectionStrategy, SaturationConfig, SaturationResult,
    SaturationState,
};

pub use unification::{unify, UnificationError, UnificationResult};

pub use parser::{fof_to_cnf, parse_tptp, parse_tptp_file, FOFFormula, Quantifier};
