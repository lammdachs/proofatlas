//! Re-exports for backward compatibility.
//! Canonical locations: crate::state, crate::generating, crate::simplifying

pub mod common;
pub mod demodulation;
pub mod derivation;
pub mod equality_factoring;
pub mod equality_resolution;
pub mod factoring;
pub mod proof;
pub mod resolution;
pub mod superposition;

// Re-export types from canonical locations
pub use crate::state::{InferenceResult, Derivation, Proof, ProofStep};

pub use crate::simplifying::demodulation::demodulate;
pub use crate::generating::equality_factoring::equality_factoring;
pub use crate::generating::equality_resolution::equality_resolution;
pub use crate::generating::factoring::factoring;
pub use crate::generating::resolution::resolution;
pub use crate::generating::superposition::superposition;
