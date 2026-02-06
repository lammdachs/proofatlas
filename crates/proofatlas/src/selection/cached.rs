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

use crate::logic::Clause;
use super::clause::SelectorStats;
use super::ClauseSelector;
use indexmap::IndexSet;
use std::collections::HashMap;
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
    /// Create a new caching selector
    pub fn new(embedder: E, scorer: S) -> Self {
        Self {
            embedder,
            scorer,
            cache: HashMap::new(),
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
    fn select(&mut self, unprocessed: &mut IndexSet<usize>, clauses: &[Clause]) -> Option<usize> {
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
                .map(|&idx| &clauses[idx])
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

        // Score all embeddings
        let t0 = Instant::now();
        let scores = self.scorer.score_batch(&embeddings);
        self.score_time += t0.elapsed();

        // Softmax sampling
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f64> = scores
            .iter()
            .map(|&s| ((s - max_score) as f64).exp())
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
        self.rng_state = 12345;
        self.cache_hits = 0;
        self.cache_misses = 0;
        self.embed_time = Duration::ZERO;
        self.score_time = Duration::ZERO;
    }

    fn stats(&self) -> Option<SelectorStats> {
        Some(SelectorStats {
            cache_hits: self.cache_hits,
            cache_misses: self.cache_misses,
            embed_time: self.embed_time,
            score_time: self.score_time,
        })
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

    #[test]
    fn test_caching_selector_caches_embeddings() {
        let mut interner = Interner::new();
        let embedder = TestEmbedder;
        let scorer = TestScorer;
        let mut selector = CachingSelector::new(embedder, scorer);

        let clauses = vec![
            make_clause(1, &mut interner),
            make_clause(2, &mut interner),
            make_clause(3, &mut interner),
        ];
        let mut unprocessed: IndexSet<usize> = (0..3).collect();

        // First selection populates cache
        let _ = selector.select(&mut unprocessed, &clauses);
        assert_eq!(selector.cache_size(), 3);

        // Reset should clear cache
        selector.reset();
        assert_eq!(selector.cache_size(), 0);
    }
}
