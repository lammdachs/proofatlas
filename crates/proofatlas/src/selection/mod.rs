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

// Core (top-level)
pub mod age_weight;
pub mod cached;
pub mod clause;

// Subgroups
pub mod ml;
pub mod network;
pub mod pipeline;
pub mod training;

// Literal selection re-exports (from logic module)
pub use crate::logic::literal_selection::{
    LiteralSelector, SelectAll, SelectMaximal,
    SelectNegMaxWeightOrMaximal, SelectUniqueMaximalOrNegOrMaximal,
};

// Clause selection re-exports
pub use age_weight::{AgeWeightSelector, AgeWeightSink, FIFOSelector, WeightSelector};
pub use cached::{CachingSelector, ClauseEmbedder, EmbeddingScorer};
pub use clause::{ClauseSelector, ProverSink, SelectorStats};
#[cfg(feature = "ml")]
pub use ml::gcn::{load_gcn_selector, GcnEmbedder, GcnScorer, GcnSelector};
#[cfg(feature = "ml")]
pub use ml::gcn::{
    load_gcn_cross_attention_selector, load_gcn_embedder, load_gcn_encoder_scorer,
    GcnCrossAttentionSelector, GcnEncoder, TorchScriptScorer,
};
#[cfg(feature = "ml")]
pub use ml::sentence::{
    load_sentence_embedder, load_sentence_selector, MiniLMEncoderModel, PassThroughScorer,
    SentenceEmbedder, SentenceSelector, tokenize_batch,
};
#[cfg(unix)]
pub use network::remote::{RemoteSelector, RemoteSelectorSink};
#[cfg(feature = "ml")]
pub use network::server::ScoringServer;

// Pipeline re-exports
pub use pipeline::backend::{Backend, BackendHandle, BackendRequest, BackendResponse, Model};
pub use pipeline::{
    ChannelSink, ContextScoreModel, DataProcessor, EmbedModel, EmbedScoreModel, ProverSignal,
    create_ml_pipeline, create_pipeline,
};
pub use pipeline::processors::{
    GcnScoreProcessor, GcnEmbeddingProcessor,
    SentenceScoreProcessor, SentenceEmbeddingProcessor,
    FeaturesScoreProcessor, FeaturesEmbeddingProcessor,
};

// Graph/ML re-exports
pub use ml::features::{extract_clause_features, NUM_CLAUSE_FEATURES};
pub use ml::graph::{BatchClauseGraph, ClauseGraph, GraphBuilder, FEATURE_DIM, NODE_TYPES};
pub use training::proof_trace::{
    compute_proof_statistics, extract_training_data,
    ProofStatistics, TrainingExample,
};
