//! ProofAtlas: A high-performance theorem prover for first-order logic
//!
//! This library provides a complete implementation of a theorem prover
//! using the superposition calculus with equality.

pub mod core;
pub mod inference;
pub mod ml;
pub mod parser;
pub mod saturation;
pub mod selectors;
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
    InferenceRule, LiteralSelector, SelectAll, SelectMaximal, SelectNegMaxWeightOrMaximal,
    SelectUniqueMaximalOrNegOrMaximal,
};

pub use selectors::{
    create_ndarray_gcn_selector, create_ndarray_mlp_selector, load_ndarray_gcn_selector,
    load_ndarray_mlp_selector, AgeWeightSelector, BurnGcnSelector, BurnMlpSelector, ClauseSelector,
    GcnModel, MlpModel, NdarrayGcnSelector, NdarrayMlpSelector,
};

pub use saturation::{
    saturate, LiteralSelectionStrategy, SaturationConfig, SaturationResult, SaturationState,
};

pub use unification::{unify, UnificationError, UnificationResult};

pub use parser::{fof_to_cnf, parse_tptp, parse_tptp_file, FOFFormula, Quantifier};
