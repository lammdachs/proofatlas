//! Machine learning support: graph representations of clauses for ML-based selection

pub mod graph;
pub mod proof_trace;

pub use graph::{BatchClauseGraph, ClauseGraph, GraphBuilder, FEATURE_DIM, NODE_TYPES};
