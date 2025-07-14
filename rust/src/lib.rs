//! ProofAtlas: A high-performance theorem prover for first-order logic
//! 
//! This library provides a complete implementation of a theorem prover
//! using the superposition calculus with equality.

pub mod core;
pub mod inference; 
pub mod selection;
pub mod saturation;
pub mod unification;
pub mod parser;

#[cfg(feature = "python")]
pub mod python_bindings;

// Re-export commonly used types and functions
pub use core::{
    Term, Variable, Constant, FunctionSymbol,
    Literal, Atom, PredicateSymbol,
    Clause, CNFFormula,
    Substitution,
    Proof, ProofStep,
    KBO, KBOConfig, TermOrdering
};

pub use inference::{
    InferenceResult, InferenceRule,
    resolution, factoring, superposition,
    equality_resolution, equality_factoring
};

pub use selection::{
    LiteralSelector, ClauseSelector,
    NoSelection, SelectNegative, SelectMaxWeight,
    FIFOSelector, SizeBasedSelector, AgeBasedSelector, AgeWeightRatioSelector
};

pub use saturation::{
    saturate, SaturationConfig, SaturationResult, SaturationState, LiteralSelectionStrategy
};

pub use unification::{
    unify, UnificationResult, UnificationError
};

pub use parser::{
    parse_tptp, parse_tptp_file,
    FOFFormula, Quantifier,
    fof_to_cnf
};