//! Selection strategies for theorem proving
//!
//! This module unifies both kinds of selection used by the saturation loop:
//!
//! - **Literal selection** ([`LiteralSelector`]) — which literal(s) in a clause
//!   are eligible for inference rules (resolution, superposition, etc.)
//! - **Clause selection** ([`ProverSink`]) — which clause to process next
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
//! # Clause selection (ProverSink)
//!
//! - [`AgeWeightSink`]: classic age-weight ratio heuristic
//! - [`ChannelSink`]: pipelined ML inference via Backend

// Core (top-level)
pub mod age_weight;
pub mod cached;
pub mod clause;

// Subgroups
pub mod ml;
pub mod pipeline;
pub mod training;

// Literal selection re-exports (from logic module)
pub use crate::logic::literal_selection::{
    LiteralSelector, SelectAll, SelectMaximal,
    SelectNegMaxWeightOrMaximal, SelectUniqueMaximalOrNegOrMaximal,
};

// Clause selection re-exports
pub use age_weight::AgeWeightSink;
pub use cached::{ClauseEmbedder, EmbeddingScorer};
pub use clause::{ProverSink, SelectorStats};
#[cfg(feature = "ml")]
pub use ml::gcn::{GcnEmbedder, GcnScorer};
#[cfg(feature = "ml")]
pub use ml::gcn::{
    load_gcn_embedder, load_gcn_encoder_scorer,
    GcnEncoder, TorchScriptScorer,
};
#[cfg(feature = "ml")]
pub use ml::sentence::{
    load_sentence_embedder, MiniLMEncoderModel, PassThroughScorer,
    SentenceEmbedder, SentenceEncoder, tokenize_batch,
};

// Pipeline re-exports
pub use pipeline::backend::{Backend, BackendHandle, BackendRequest, BackendResponse, Model, ModelSpec};
pub use pipeline::{
    ChannelSink, ContextScoreModel, DataProcessor, EmbedModel, EmbedScoreModel, ProverSignal,
    create_ml_pipeline, create_pipeline,
};
pub use pipeline::processors::{
    GcnScoreProcessor, GcnEmbeddingProcessor,
    SentenceScoreProcessor, SentenceEmbeddingProcessor,
    FeaturesScoreProcessor, FeaturesEmbeddingProcessor,
};

// Features model re-exports
#[cfg(feature = "ml")]
pub use ml::features::{FeaturesEmbedder, FeaturesEncoder, load_features_embedder};

// Graph/ML re-exports
pub use ml::features::{extract_clause_features, NUM_CLAUSE_FEATURES};
pub use ml::graph::{BatchClauseGraph, ClauseGraph, GraphBuilder, FEATURE_DIM, NODE_TYPES};
