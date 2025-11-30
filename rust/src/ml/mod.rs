//! Machine learning support: graph representations of clauses and ML inference

mod graph;
pub mod inference;
pub mod proof_trace;

pub use graph::{ClauseGraph, GraphBuilder, FEATURE_DIM, NODE_TYPES};
pub use inference::{ClauseScorer, InferenceError};
