//! Selection strategies for theorem proving
//!
//! This module unifies both kinds of selection used by the saturation loop:
//!
//! - **Literal selection** ([`LiteralSelector`]) — which literal(s) in a clause
//!   are eligible for inference rules (resolution, superposition, etc.)
//! - **Clause selection** ([`ClauseSelector`]) — which clause to process next
//!   from the unprocessed set (the "given clause" choice)
//!
//! # Literal selection strategies
//!
//! Based on Hoder et al. "Selecting the selection" (2016):
//! - [`SelectAll`] (Sel0): all literals eligible
//! - [`SelectMaximal`] (Sel20): all maximal literals
//! - [`SelectUniqueMaximalOrNegOrMaximal`] (Sel21): unique maximal, else neg max-weight, else all maximal
//! - [`SelectNegMaxWeightOrMaximal`] (Sel22): neg max-weight literal, else all maximal
//!
//! # Clause selection strategies
//!
//! - [`AgeWeightSelector`]: classic age-weight ratio heuristic
//! - [`CachingSelector`]: ML-based selection with embedding cache
//! - [`GcnSelector`] (feature-gated): GCN graph neural network
//! - [`SentenceSelector`] (feature-gated): sentence transformer

pub mod age_weight;
pub mod cached;
pub mod clause;
#[cfg(feature = "ml")]
pub mod gcn;
pub mod graph;
pub mod proof_trace;
#[cfg(feature = "ml")]
pub mod sentence;

// Literal selection re-exports (from logic module)
pub use crate::logic::literal_selection::{
    LiteralSelector, SelectAll, SelectMaximal,
    SelectNegMaxWeightOrMaximal, SelectUniqueMaximalOrNegOrMaximal,
};

// Clause selection re-exports
pub use age_weight::{AgeWeightSelector, FIFOSelector, WeightSelector};
pub use cached::{CachingSelector, ClauseEmbedder, EmbeddingScorer};
pub use clause::{ClauseSelector, SelectorStats};
#[cfg(feature = "ml")]
pub use gcn::{load_gcn_selector, GcnEmbedder, GcnScorer, GcnSelector};
#[cfg(feature = "ml")]
pub use gcn::{
    load_gcn_cross_attention_selector, GcnCrossAttentionSelector, GcnEncoder, TorchScriptScorer,
};
#[cfg(feature = "ml")]
pub use sentence::{load_sentence_selector, PassThroughScorer, SentenceEmbedder, SentenceSelector};

// Graph/ML re-exports
pub use graph::{BatchClauseGraph, ClauseGraph, GraphBuilder, FEATURE_DIM, NODE_TYPES};
pub use proof_trace::{
    compute_proof_statistics, extract_training_data,
    ProofStatistics, TrainingExample,
};
