//! ProofAtlas: A high-performance theorem prover for first-order logic
//!
//! This library provides a complete implementation of a theorem prover
//! using the superposition calculus with equality.

pub mod config;
pub mod generating;
pub mod index;
pub mod logic;
pub mod state;
pub mod simplifying;
pub mod json;
pub mod parser;
pub mod prover;
pub mod selection;
pub mod time_compat;
pub mod trace;
pub mod profile;

#[cfg(feature = "python")]
pub mod python_bindings;

// Re-export ClauseManager and ProofAtlas
pub use logic::clause_manager::ClauseManager;
pub use prover::{ProofAtlas, saturate};

// Re-export commonly used types from logic
pub use logic::{
    Atom, CNFFormula, Clause, Constant, FunctionSymbol, Interner, KBOConfig, Literal, Position,
    PredicateSymbol, Substitution, Term, TermOrdering, Variable, KBO,
};
// Note: Atom is re-exported for FOF formula usage in parsers

// Re-export state types
pub use state::{
    Derivation, EventLog, InferenceResult, Proof, ProofResult, ProofStep,
    SaturationState, StateChange,
};

// Re-export config types
pub use config::{LiteralSelectionStrategy, ProverConfig};

// Re-export profile types
pub use profile::SaturationProfile;

// Re-export trace types
pub use trace::{extract_proof_from_events, EventLogReplayer};

// Re-export generating inference functions
pub use generating::{
    equality_factoring, equality_resolution, factoring, resolution, superposition,
};

// Re-export selection types
pub use selection::{
    AgeWeightSelector, ClauseSelector, FIFOSelector, LiteralSelector, SelectAll, SelectMaximal,
    SelectNegMaxWeightOrMaximal, SelectUniqueMaximalOrNegOrMaximal, WeightSelector,
};

#[cfg(feature = "ml")]
pub use selection::{load_gcn_selector, GcnEmbedder, GcnScorer, GcnSelector};

#[cfg(feature = "ml")]
pub use selection::{load_sentence_selector, PassThroughScorer, SentenceEmbedder, SentenceSelector};

pub use logic::{unify, UnificationError, UnificationResult};

pub use parser::{fof_to_cnf, parse_tptp, parse_tptp_file, FOFFormula, Quantifier};
