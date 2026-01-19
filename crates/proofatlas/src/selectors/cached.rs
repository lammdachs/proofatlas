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

use crate::core::Clause;
use super::ClauseSelector;
use std::collections::{HashMap, VecDeque};

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
    /// Cache: clause string -> embedding vector
    cache: HashMap<String, Vec<f32>>,
    /// RNG state for sampling
    rng_state: u64,
}

impl<E: ClauseEmbedder, S: EmbeddingScorer> CachingSelector<E, S> {
    /// Create a new caching selector
    pub fn new(embedder: E, scorer: S) -> Self {
        Self {
            embedder,
            scorer,
            cache: HashMap::new(),
            rng_state: 12345,
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

    fn clause_to_key(clause: &Clause) -> String {
        clause.to_string()
    }
}

impl<E: ClauseEmbedder, S: EmbeddingScorer> ClauseSelector for CachingSelector<E, S> {
    fn select(&mut self, unprocessed: &mut VecDeque<usize>, clauses: &[Clause]) -> Option<usize> {
        if unprocessed.is_empty() {
            return None;
        }

        if unprocessed.len() == 1 {
            return unprocessed.pop_front();
        }

        // Get clause keys and find uncached clauses
        let clause_keys: Vec<String> = unprocessed
            .iter()
            .map(|&idx| Self::clause_to_key(&clauses[idx]))
            .collect();

        let uncached: Vec<(usize, &Clause)> = clause_keys
            .iter()
            .enumerate()
            .filter(|(_, key)| !self.cache.contains_key(*key))
            .map(|(i, _)| {
                let clause_idx = unprocessed[i];
                (i, &clauses[clause_idx])
            })
            .collect();

        // Compute embeddings for uncached clauses
        if !uncached.is_empty() {
            let uncached_clauses: Vec<&Clause> = uncached.iter().map(|(_, c)| *c).collect();
            let embeddings = self.embedder.embed_batch(&uncached_clauses);

            // Store in cache
            for ((i, _), embedding) in uncached.iter().zip(embeddings.into_iter()) {
                self.cache.insert(clause_keys[*i].clone(), embedding);
            }
        }

        // Gather all embeddings from cache
        let embeddings: Vec<&[f32]> = clause_keys
            .iter()
            .map(|key| self.cache.get(key).unwrap().as_slice())
            .collect();

        // Score all embeddings
        let scores = self.scorer.score_batch(&embeddings);

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
                return unprocessed.remove(i);
            }
        }

        unprocessed.pop_back()
    }

    fn name(&self) -> &str {
        "caching_selector"
    }

    fn reset(&mut self) {
        self.cache.clear();
        self.rng_state = 12345;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Atom, Literal, PredicateSymbol, Term, Constant};

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

    fn make_clause(num_literals: usize) -> Clause {
        let p = PredicateSymbol {
            name: "P".to_string(),
            arity: 1,
        };
        let a = Term::Constant(Constant {
            name: "a".to_string(),
        });
        let literals: Vec<Literal> = (0..num_literals)
            .map(|_| Literal::positive(Atom {
                predicate: p.clone(),
                args: vec![a.clone()],
            }))
            .collect();
        Clause::new(literals)
    }

    #[test]
    fn test_caching_selector_caches_embeddings() {
        let embedder = TestEmbedder;
        let scorer = TestScorer;
        let mut selector = CachingSelector::new(embedder, scorer);

        let clauses = vec![make_clause(1), make_clause(2), make_clause(3)];
        let mut unprocessed: VecDeque<usize> = (0..3).collect();

        // First selection populates cache
        let _ = selector.select(&mut unprocessed, &clauses);
        assert_eq!(selector.cache_size(), 3);

        // Reset should clear cache
        selector.reset();
        assert_eq!(selector.cache_size(), 0);
    }
}
