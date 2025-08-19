//! Inference rules for first-order logic theorem proving

pub mod common;
pub mod demodulation;
pub mod equality_factoring;
pub mod equality_resolution;
pub mod factoring;
pub mod resolution;
pub mod superposition;

// Re-export the main inference function and types
pub use common::{InferenceResult, InferenceRule};

// Note: The inference rules now require a LiteralSelector parameter
// Use them like: resolution(clause1, clause2, idx1, idx2, &selector)
pub use demodulation::demodulate;
pub use equality_factoring::equality_factoring;
pub use equality_resolution::equality_resolution;
pub use factoring::factoring;
pub use resolution::resolution;
pub use superposition::superposition;
