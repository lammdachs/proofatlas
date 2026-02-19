//! ProofAtlas: A high-performance theorem prover for first-order logic
//!
//! This library provides a complete implementation of a theorem prover
//! using the superposition calculus with equality.

pub mod atlas;
pub mod config;
pub mod generating;
pub mod index;
pub mod logic;
pub mod state;
pub mod simplifying;
pub mod parser;
pub mod prover;
pub mod selection;

#[cfg(feature = "python")]
pub mod python_bindings;

// Re-export orchestrator and prover
pub use atlas::ProofAtlas;
pub use logic::clause_manager::ClauseManager;
pub use prover::{Prover, saturate};

// Re-export commonly used types from logic
pub use logic::{
    CNFFormula, Clause, Constant, FunctionSymbol, Interner, KBOConfig, Literal, Position,
    PredicateSymbol, Substitution, Term, TermOrdering, Variable, KBO,
};

// Re-export state types
pub use state::{
    clause_indices, EventLog, ProofResult, ProofStep,
    SaturationState, StateChange, VerificationError,
};

// Re-export config types
pub use config::{LiteralSelectionStrategy, ProverConfig};

// Re-export profile types
pub use prover::profile::SaturationProfile;

// Re-export generating inference functions
pub use generating::{
    equality_factoring, equality_resolution, factoring, resolution, superposition,
};

// Re-export selection types
pub use selection::{
    AgeWeightSink, LiteralSelector,
    ProverSink, SelectAll, SelectMaximal, SelectNegMaxWeightOrMaximal,
    SelectUniqueMaximalOrNegOrMaximal,
};

#[cfg(feature = "ml")]
pub use selection::{GcnEmbedder, GcnScorer};

#[cfg(feature = "ml")]
pub use selection::{PassThroughScorer, SentenceEmbedder};

pub use logic::{unify, UnificationError, UnificationResult};

pub use parser::{fof_to_cnf, parse_tptp, parse_tptp_file, FOFFormula, Quantifier};
