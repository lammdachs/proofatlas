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
//! ## ML-based selectors (requires `ml` feature):
//! - [`GcnSelector`] - GCN-based clause selection
//! - [`SentenceSelector`] - Sentence transformer selection

pub mod age_weight;
pub mod cached;
pub mod clause;
#[cfg(feature = "ml")]
pub mod gcn;
#[cfg(feature = "ml")]
pub mod sentence;

pub use age_weight::AgeWeightSelector;
pub use cached::{CachingSelector, ClauseEmbedder, EmbeddingScorer};
pub use clause::{ClauseSelector, SelectorStats};
#[cfg(feature = "ml")]
pub use gcn::{load_gcn_selector, GcnEmbedder, GcnScorer, GcnSelector};
#[cfg(feature = "ml")]
pub use sentence::{load_sentence_selector, PassThroughScorer, SentenceEmbedder, SentenceSelector};
