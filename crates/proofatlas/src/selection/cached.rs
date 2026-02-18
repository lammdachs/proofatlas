//! Clause embedding and scoring traits
//!
//! `ClauseEmbedder` converts clauses to fixed-size embedding vectors.
//! `EmbeddingScorer` scores pre-computed embeddings.
//!
//! Used by the pipelined ML inference architecture in `selection::pipeline`.

use crate::logic::{Clause, Interner};
use std::sync::Arc;

/// Trait for models that compute clause embeddings
///
/// Implementations convert clauses to fixed-size embedding vectors.
/// The embedding computation may be expensive (e.g., BERT forward pass),
/// so results should be cached by the [`CachingSelector`].
pub trait ClauseEmbedder: Send {
    /// Compute embeddings for a batch of clauses
    ///
    /// Returns one embedding vector per input clause.
    fn embed_batch(&self, clauses: &[&Clause]) -> Vec<Vec<f32>>;

    /// Get the embedding dimension
    fn embedding_dim(&self) -> usize;

    /// Get the name of this embedder
    fn name(&self) -> &str;

    /// Compute embeddings from pre-serialized text strings.
    ///
    /// Used by the pipeline architecture where clause→string conversion
    /// happens in the data processing thread (not the model worker).
    /// Default: unimplemented (only sentence embedders need this).
    fn embed_texts(&self, _texts: &[&str]) -> Vec<Vec<f32>> {
        unimplemented!("embed_texts not supported by this embedder")
    }

    /// Compute embeddings from pre-extracted feature vectors.
    ///
    /// Used by the pipeline architecture for features-only models where
    /// 9 clause features are extracted in the data processing thread.
    /// Default: unimplemented (only features embedders need this).
    fn embed_features(&self, _features: &[&[f32]]) -> Vec<Vec<f32>> {
        unimplemented!("embed_features not supported by this embedder")
    }

    /// Provide the symbol interner for clause serialization.
    /// Embedders that need symbol names (e.g., sentence transformers) should
    /// store this and use it during `embed_batch`.
    fn set_interner(&mut self, _interner: Arc<Interner>) {}
}

/// Trait for models that score embeddings
///
/// Implementations take pre-computed embeddings and produce scores.
/// Scoring should be fast since embeddings are already computed.
pub trait EmbeddingScorer: Send {
    /// Score a batch of embeddings
    ///
    /// Returns one score per input embedding.
    fn score_batch(&self, embeddings: &[&[f32]]) -> Vec<f32>;

    /// Score unprocessed embeddings with processed set context (cross-attention).
    ///
    /// When processed embeddings are available, the scorer can use them as
    /// context (K/V in cross-attention). Falls back to `score_batch` by default.
    fn score_with_context(
        &self,
        u_embeddings: &[&[f32]],
        p_embeddings: &[&[f32]],
    ) -> Vec<f32> {
        let _ = p_embeddings; // Default: ignore context
        self.score_batch(u_embeddings)
    }

    /// Whether this scorer uses cross-attention (needs P context)
    fn uses_context(&self) -> bool {
        false
    }

    /// Get the name of this scorer
    fn name(&self) -> &str;
}
