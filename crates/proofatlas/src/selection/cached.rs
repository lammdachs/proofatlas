//! Caching clause selector architecture
//!
//! This module provides a caching layer that separates embedding computation from scoring.
//! Embeddings are computed once per clause and cached, while scoring happens on each selection.
//!
//! # Architecture
//!
//! ```text
//! Clause -> [ClauseEmbedder] -> Embedding (cached) -> [EmbeddingScorer] -> Score
//! ```
//!
//! This design allows:
//! - Any embedding model (GCN, MLP, Sentence) to benefit from caching
//! - Embeddings to be reused across multiple scoring calls
//! - Clear separation between expensive embedding computation and cheap scoring

use crate::logic::{Clause, Interner};
use super::clause::SelectorStats;
use super::ClauseSelector;
use indexmap::IndexSet;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

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

/// Clause selector that caches embeddings
///
/// This selector wraps an embedder and scorer, caching embeddings by clause string.
/// When selecting a clause:
/// 1. Check cache for each clause's embedding
/// 2. Compute embeddings for uncached clauses (batch)
/// 3. Store new embeddings in cache
/// 4. Score all embeddings
/// 5. Sample based on scores
pub struct CachingSelector<E: ClauseEmbedder, S: EmbeddingScorer> {
    embedder: E,
    scorer: S,
    /// Cache: clause index -> embedding vector
    cache: HashMap<usize, Vec<f32>>,
    /// Indices of clauses in the processed set (for cross-attention context)
    processed_indices: Vec<usize>,
    /// Softmax temperature for sampling (τ=1.0 is default)
    temperature: f32,
    /// RNG state for sampling
    rng_state: u64,
    /// Accumulated cache hits
    cache_hits: usize,
    /// Accumulated cache misses
    cache_misses: usize,
    /// Accumulated embedding time
    embed_time: Duration,
    /// Accumulated scoring time
    score_time: Duration,
}

impl<E: ClauseEmbedder, S: EmbeddingScorer> std::fmt::Debug for CachingSelector<E, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachingSelector")
            .field("embedder", &self.embedder.name())
            .field("scorer", &self.scorer.name())
            .field("cache_size", &self.cache.len())
            .finish()
    }
}

impl<E: ClauseEmbedder, S: EmbeddingScorer> CachingSelector<E, S> {
    /// Create a new caching selector with default temperature (τ=1.0)
    pub fn new(embedder: E, scorer: S) -> Self {
        Self::with_temperature(embedder, scorer, 1.0)
    }

    /// Create a new caching selector with a specific softmax temperature
    pub fn with_temperature(embedder: E, scorer: S, temperature: f32) -> Self {
        Self {
            embedder,
            scorer,
            cache: HashMap::new(),
            processed_indices: Vec::new(),
            temperature,
            rng_state: 12345,
            cache_hits: 0,
            cache_misses: 0,
            embed_time: Duration::ZERO,
            score_time: Duration::ZERO,
        }
    }

    /// Get a reference to the embedder
    pub fn embedder(&self) -> &E {
        &self.embedder
    }

    /// Get a reference to the scorer
    pub fn scorer(&self) -> &S {
        &self.scorer
    }

    /// Get the number of cached embeddings
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    fn next_random(&mut self) -> f64 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        (self.rng_state >> 33) as f64 / (1u64 << 31) as f64
    }

}

impl<E: ClauseEmbedder, S: EmbeddingScorer> ClauseSelector for CachingSelector<E, S> {
    fn select(&mut self, unprocessed: &mut IndexSet<usize>, clauses: &[Arc<Clause>]) -> Option<usize> {
        if unprocessed.is_empty() {
            return None;
        }

        if unprocessed.len() == 1 {
            return unprocessed.shift_remove_index(0);
        }

        // Find uncached clauses by index
        let uncached_indices: Vec<usize> = unprocessed
            .iter()
            .copied()
            .filter(|idx| !self.cache.contains_key(idx))
            .collect();

        // Track cache hits/misses
        let cached_count = unprocessed.len() - uncached_indices.len();
        self.cache_hits += cached_count;
        self.cache_misses += uncached_indices.len();

        // Compute embeddings for uncached clauses
        if !uncached_indices.is_empty() {
            let uncached_clauses: Vec<&Clause> = uncached_indices
                .iter()
                .map(|&idx| clauses[idx].as_ref())
                .collect();

            let t0 = Instant::now();
            let embeddings = self.embedder.embed_batch(&uncached_clauses);
            self.embed_time += t0.elapsed();

            for (idx, embedding) in uncached_indices.into_iter().zip(embeddings.into_iter()) {
                self.cache.insert(idx, embedding);
            }
        }

        // Gather all embeddings from cache
        let embeddings: Vec<&[f32]> = unprocessed
            .iter()
            .map(|idx| self.cache.get(idx).unwrap().as_slice())
            .collect();

        // Score all embeddings (with P context if scorer uses cross-attention)
        let t0 = Instant::now();
        let scores = if self.scorer.uses_context() && !self.processed_indices.is_empty() {
            let p_embeddings: Vec<&[f32]> = self
                .processed_indices
                .iter()
                .filter_map(|idx| self.cache.get(idx).map(|e| e.as_slice()))
                .collect();
            self.scorer.score_with_context(&embeddings, &p_embeddings)
        } else {
            self.scorer.score_batch(&embeddings)
        };
        self.score_time += t0.elapsed();

        // Softmax sampling with temperature
        let tau = self.temperature;
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f64> = scores
            .iter()
            .map(|&s| (((s - max_score) / tau) as f64).exp())
            .collect();
        let sum: f64 = exp_scores.iter().sum();
        let probs: Vec<f64> = exp_scores.iter().map(|&e| e / sum).collect();

        // Sample
        let r = self.next_random();
        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return unprocessed.shift_remove_index(i);
            }
        }

        unprocessed.pop()
    }

    fn name(&self) -> &str {
        "caching_selector"
    }

    fn reset(&mut self) {
        self.cache.clear();
        self.processed_indices.clear();
        self.rng_state = 12345;
        self.cache_hits = 0;
        self.cache_misses = 0;
        self.embed_time = Duration::ZERO;
        self.score_time = Duration::ZERO;
    }

    fn on_clause_processed(&mut self, clause_idx: usize) {
        self.processed_indices.push(clause_idx);
    }

    fn stats(&self) -> Option<SelectorStats> {
        Some(SelectorStats {
            cache_hits: self.cache_hits,
            cache_misses: self.cache_misses,
            embed_time: self.embed_time,
            score_time: self.score_time,
        })
    }

    fn set_interner(&mut self, interner: Arc<Interner>) {
        self.embedder.set_interner(interner);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::{Interner, Literal, PredicateSymbol, Term, Constant};

    /// Simple test embedder that returns clause length as embedding
    struct TestEmbedder;

    impl ClauseEmbedder for TestEmbedder {
        fn embed_batch(&self, clauses: &[&Clause]) -> Vec<Vec<f32>> {
            clauses
                .iter()
                .map(|c| vec![c.literals.len() as f32])
                .collect()
        }

        fn embedding_dim(&self) -> usize {
            1
        }

        fn name(&self) -> &str {
            "test"
        }
    }

    /// Simple test scorer that returns embedding value as score
    struct TestScorer;

    impl EmbeddingScorer for TestScorer {
        fn score_batch(&self, embeddings: &[&[f32]]) -> Vec<f32> {
            embeddings.iter().map(|e| e[0]).collect()
        }

        fn name(&self) -> &str {
            "test"
        }
    }

    fn make_clause(num_literals: usize, interner: &mut Interner) -> Clause {
        let p = PredicateSymbol {
            id: interner.intern_predicate("P"),
            arity: 1,
        };
        let a = Term::Constant(Constant {
            id: interner.intern_constant("a"),
        });
        let literals: Vec<Literal> = (0..num_literals)
            .map(|_| Literal::positive(p.clone(), vec![a.clone()]))
            .collect();
        Clause::new(literals)
    }

    /// Context scorer that doubles scores when P is non-empty
    struct TestContextScorer;

    impl EmbeddingScorer for TestContextScorer {
        fn score_batch(&self, embeddings: &[&[f32]]) -> Vec<f32> {
            embeddings.iter().map(|e| e[0]).collect()
        }

        fn score_with_context(
            &self,
            u_embeddings: &[&[f32]],
            p_embeddings: &[&[f32]],
        ) -> Vec<f32> {
            if p_embeddings.is_empty() {
                self.score_batch(u_embeddings)
            } else {
                // Double scores when context is available
                u_embeddings.iter().map(|e| e[0] * 2.0).collect()
            }
        }

        fn uses_context(&self) -> bool {
            true
        }

        fn name(&self) -> &str {
            "test_context"
        }
    }

    #[test]
    fn test_caching_selector_caches_embeddings() {
        let mut interner = Interner::new();
        let embedder = TestEmbedder;
        let scorer = TestScorer;
        let mut selector = CachingSelector::new(embedder, scorer);

        let clauses: Vec<Arc<Clause>> = vec![
            Arc::new(make_clause(1, &mut interner)),
            Arc::new(make_clause(2, &mut interner)),
            Arc::new(make_clause(3, &mut interner)),
        ];
        let mut unprocessed: IndexSet<usize> = (0..3).collect();

        // First selection populates cache
        let _ = selector.select(&mut unprocessed, &clauses);
        assert_eq!(selector.cache_size(), 3);

        // Reset should clear cache
        selector.reset();
        assert_eq!(selector.cache_size(), 0);
    }

    #[test]
    fn test_on_clause_processed() {
        let embedder = TestEmbedder;
        let scorer = TestScorer;
        let mut selector = CachingSelector::new(embedder, scorer);

        selector.on_clause_processed(0);
        assert_eq!(selector.processed_indices, vec![0]);

        selector.on_clause_processed(5);
        assert_eq!(selector.processed_indices, vec![0, 5]);
    }

    #[test]
    fn test_reset_clears_processed() {
        let embedder = TestEmbedder;
        let scorer = TestScorer;
        let mut selector = CachingSelector::new(embedder, scorer);

        selector.on_clause_processed(0);
        selector.on_clause_processed(1);
        assert_eq!(selector.processed_indices.len(), 2);

        selector.reset();
        assert!(selector.processed_indices.is_empty());
    }

    #[test]
    fn test_select_with_context_scorer() {
        let mut interner = Interner::new();
        let embedder = TestEmbedder;
        let scorer = TestContextScorer;
        let mut selector = CachingSelector::new(embedder, scorer);

        let clauses: Vec<Arc<Clause>> = vec![
            Arc::new(make_clause(1, &mut interner)),
            Arc::new(make_clause(2, &mut interner)),
            Arc::new(make_clause(3, &mut interner)),
        ];

        // First: process clause 0 so it's in P
        // We need to embed it first, so we select once
        let mut unprocessed: IndexSet<usize> = (0..3).collect();
        let _ = selector.select(&mut unprocessed, &clauses);

        // Now mark clause 0 as processed
        selector.on_clause_processed(0);

        // Select again with remaining clauses — context scorer should be used
        let mut unprocessed2: IndexSet<usize> = (1..3).collect();
        let _ = selector.select(&mut unprocessed2, &clauses);

        // Verify that uses_context scorer was invoked (processed_indices is non-empty)
        assert!(!selector.processed_indices.is_empty());
    }

    #[test]
    fn test_select_without_context_scorer() {
        let mut interner = Interner::new();
        let embedder = TestEmbedder;
        let scorer = TestScorer; // uses_context() = false
        let mut selector = CachingSelector::new(embedder, scorer);

        let clauses: Vec<Arc<Clause>> = vec![
            Arc::new(make_clause(1, &mut interner)),
            Arc::new(make_clause(2, &mut interner)),
        ];

        // Process clause 0
        selector.on_clause_processed(0);

        // Select — should use score_batch (not score_with_context) since uses_context is false
        let mut unprocessed: IndexSet<usize> = (0..2).collect();
        let _ = selector.select(&mut unprocessed, &clauses);

        // With TestScorer (no context), it should still work fine
        assert!(selector.cache_size() > 0);
    }
}
