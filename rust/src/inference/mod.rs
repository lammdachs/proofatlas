//! Inference rules for first-order logic theorem proving

pub mod resolution;
pub mod factoring;
pub mod superposition;
pub mod equality_resolution;
pub mod equality_factoring;
pub mod demodulation;
pub mod common;

// Re-export the main inference function and types
pub use common::{InferenceResult, InferenceRule};

// Note: The inference rules now require a LiteralSelector parameter
// Use them like: resolution(clause1, clause2, idx1, idx2, &selector)
pub use resolution::resolution;
pub use factoring::factoring;
pub use superposition::superposition;
pub use equality_resolution::equality_resolution;
pub use equality_factoring::equality_factoring;
pub use demodulation::demodulate;
