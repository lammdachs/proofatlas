//! Inference rules for first-order logic theorem proving

pub mod common;
pub mod demodulation;
pub mod derivation;
pub mod equality_factoring;
pub mod equality_resolution;
pub mod factoring;
pub mod proof;
pub mod resolution;
pub mod superposition;

// Re-export the main inference function and types
pub use common::InferenceResult;
pub use derivation::Derivation;
pub use proof::{Proof, ProofStep};

pub use demodulation::demodulate;
pub use equality_factoring::equality_factoring;
pub use equality_resolution::equality_resolution;
pub use factoring::factoring;
pub use resolution::resolution;
pub use superposition::superposition;
