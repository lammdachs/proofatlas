//! Machine learning support: graph representations of clauses for ML-based selection

pub mod graph;
pub mod proof_trace;

pub use graph::{BatchClauseGraph, ClauseGraph, GraphBuilder, FEATURE_DIM, NODE_TYPES};
pub use proof_trace::{
    compute_proof_statistics, extract_clause_labels_from_events, extract_training_data,
    extract_training_from_events, ProofStatistics, SelectionTrainingExample, TrainingExample,
};
