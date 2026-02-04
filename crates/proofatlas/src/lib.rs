//! ProofAtlas: A high-performance theorem prover for first-order logic
//!
//! This library provides a complete implementation of a theorem prover
//! using the superposition calculus with equality.

pub mod fol;
pub mod inference;
pub mod json;
pub mod parser;
pub mod saturation;
pub mod selection;
pub mod time_compat;
pub mod unification;

#[cfg(feature = "python")]
pub mod python_bindings;

// Re-export commonly used types from fol
pub use fol::{
    Atom, CNFFormula, Clause, Constant, FunctionSymbol, Interner, KBOConfig, Literal,
    PredicateSymbol, Substitution, Term, TermOrdering, Variable, KBO,
};

// Re-export inference types
pub use inference::{
    equality_factoring, equality_resolution, factoring, resolution, superposition, Derivation,
    InferenceResult, Proof, ProofStep,
};

// Re-export selection types
pub use selection::{
    AgeWeightSelector, ClauseSelector, LiteralSelector, SelectAll, SelectMaximal,
    SelectNegMaxWeightOrMaximal, SelectUniqueMaximalOrNegOrMaximal,
};

#[cfg(feature = "ml")]
pub use selection::{load_gcn_selector, GcnEmbedder, GcnScorer, GcnSelector};

#[cfg(feature = "ml")]
pub use selection::{load_sentence_selector, PassThroughScorer, SentenceEmbedder, SentenceSelector};

// Re-export saturation types
pub use saturation::{
    saturate, EventLogReplayer, LiteralSelectionStrategy, ProofStateChange, SaturationConfig,
    SaturationEventLog, SaturationProfile, SaturationResult, SaturationState,
};

pub use unification::{unify, UnificationError, UnificationResult};

pub use parser::{fof_to_cnf, parse_tptp, parse_tptp_file, FOFFormula, Quantifier};
