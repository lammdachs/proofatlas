//! Clause selection strategies for the given-clause algorithm
//!
//! This module provides clause selectors that determine the order of clause
//! processing during saturation. The choice of clause selector significantly
//! impacts proof search efficiency.
//!
//! # Architecture
//!
//! ML-based selectors use a caching architecture that separates embedding from scoring:
//!
//! ```text
//! Clause -> [ClauseEmbedder] -> Embedding (cached) -> [EmbeddingScorer] -> Score
//! ```
//!
//! This allows embeddings to be computed once and reused across selections.
//!
//! # Available Selectors
//!
//! ## Heuristic selectors:
//! - [`AgeWeightSelector`] - Classic age-weight ratio heuristic
//!
//! ## ML-based selectors (Burn framework):
//! - [`BurnGcnSelector`] / [`NdarrayGcnSelector`] - GCN-based clause selection
//! - [`BurnMlpSelector`] / [`NdarrayMlpSelector`] - MLP-based clause selection
//! - [`BurnSentenceSelector`] / [`NdarraySentenceSelector`] - Sentence transformer selection (requires `sentence` feature)

pub mod age_weight;
pub mod burn_gcn;
pub mod burn_mlp;
pub mod burn_sentence;
pub mod cached;
pub mod clause;
pub mod tch_gcn;

pub use age_weight::AgeWeightSelector;
pub use burn_gcn::{
    create_ndarray_gcn_selector, load_ndarray_gcn_selector, BurnGcnSelector, GcnModel,
    NdarrayGcnSelector,
};
pub use burn_mlp::{
    create_ndarray_mlp_selector, load_ndarray_mlp_selector, BurnMlpSelector, MlpModel,
    NdarrayMlpSelector,
};
pub use burn_sentence::SentenceModel;
#[cfg(feature = "sentence")]
pub use burn_sentence::{load_ndarray_sentence_selector, BurnSentenceSelector, NdarraySentenceSelector};
#[cfg(all(feature = "sentence", feature = "onnx"))]
pub use burn_sentence::{load_onnx_sentence_selector, OnnxSentenceEmbedder, OnnxSentenceSelector};
#[cfg(all(feature = "sentence", feature = "torch"))]
pub use burn_sentence::{load_tch_sentence_selector, PassThroughScorer, TchSentenceEmbedder, TchSentenceSelector};
pub use cached::{CachingSelector, ClauseEmbedder, EmbeddingScorer};
pub use clause::ClauseSelector;
#[cfg(feature = "torch")]
pub use tch_gcn::{load_tch_gcn_selector, TchGcnSelector};
