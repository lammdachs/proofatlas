//! Clause selection strategies for the given-clause algorithm
//!
//! This module provides clause selectors that determine the order of clause
//! processing during saturation. The choice of clause selector significantly
//! impacts proof search efficiency.
//!
//! # Available Selectors
//!
//! ## Heuristic selectors:
//! - [`AgeWeightSelector`] - Classic age-weight ratio heuristic
//!
//! ## ML-based selectors (requires `torch` feature):
//! - [`GcnSelector`] - GCN-based clause selection
//! - [`SentenceSelector`] - Sentence transformer selection (requires `sentence` feature)

pub mod age_weight;
pub mod cached;
pub mod clause;
#[cfg(feature = "torch")]
pub mod gcn;
#[cfg(all(feature = "sentence", feature = "torch"))]
pub mod sentence;

pub use age_weight::AgeWeightSelector;
pub use cached::{CachingSelector, ClauseEmbedder, EmbeddingScorer};
pub use clause::ClauseSelector;
#[cfg(feature = "torch")]
pub use gcn::{load_gcn_selector, GcnSelector};
#[cfg(all(feature = "sentence", feature = "torch"))]
pub use sentence::{load_sentence_selector, PassThroughScorer, SentenceEmbedder, SentenceSelector};
