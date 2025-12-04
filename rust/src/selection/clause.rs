//! Clause selection strategies for the given clause algorithm
//!
//! ONNX-based clause selection is the only available strategy.

use crate::core::Clause;
use crate::ml::ClauseScorer;
use std::collections::VecDeque;
use std::path::Path;

/// Trait for clause selection strategies
pub trait ClauseSelector: Send + Sync {
    /// Select the next clause from the unprocessed set
    /// Returns the index of the selected clause, or None if empty
    fn select(&mut self, unprocessed: &mut VecDeque<usize>, clauses: &[Clause]) -> Option<usize>;

    /// Get the name of this selection strategy
    fn name(&self) -> &str;
}

/// ONNX-based clause selector using a trained ML model.
///
/// This selector uses a neural network model exported to ONNX format
/// to score clauses. The model considers clause features like structure,
/// age, and context to predict which clauses are most likely to be useful.
///
/// Scores are treated as logits and converted to probabilities via softmax.
/// Clauses are then sampled from this distribution rather than always
/// picking the highest score, allowing for exploration.
pub struct OnnxClauseSelector {
    /// The underlying clause scorer
    scorer: ClauseScorer,
    /// Maximum age for normalization (clauses with higher age get clipped)
    max_age: usize,
    /// Random number generator state (simple LCG)
    rng_state: u64,
}

impl OnnxClauseSelector {
    /// Create a new ONNX clause selector from a model file.
    ///
    /// # Arguments
    /// * `model_path` - Path to the ONNX model file (.onnx)
    ///
    /// # Returns
    /// * `Ok(Self)` if the model loads successfully
    /// * `Err(String)` with error message if loading fails
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, String> {
        let mut scorer = ClauseScorer::new();
        scorer
            .load_model(&model_path)
            .map_err(|e| format!("Failed to load ONNX model: {}", e))?;

        Ok(OnnxClauseSelector {
            scorer,
            max_age: 1000,
            rng_state: 12345,
        })
    }

    /// Create a new ONNX clause selector from model bytes.
    ///
    /// This is useful for WASM where the file system is not available
    /// and the model must be loaded from memory.
    ///
    /// # Arguments
    /// * `model_bytes` - The ONNX model as a byte slice
    ///
    /// # Returns
    /// * `Ok(Self)` if the model loads successfully
    /// * `Err(String)` with error message if loading fails
    pub fn from_bytes(model_bytes: &[u8]) -> Result<Self, String> {
        let mut scorer = ClauseScorer::new();
        scorer
            .load_model_from_bytes(model_bytes)
            .map_err(|e| format!("Failed to load ONNX model from bytes: {}", e))?;

        Ok(OnnxClauseSelector {
            scorer,
            max_age: 1000,
            rng_state: 12345,
        })
    }

    /// Set the maximum age for normalization.
    ///
    /// Clause ages are normalized to [0, 1] by dividing by max_age.
    /// Ages above max_age are clipped to 1.0.
    pub fn with_max_age(mut self, max_age: usize) -> Self {
        self.max_age = max_age;
        self
    }

    /// Score all clauses using the ONNX model.
    fn score_clauses(&self, clauses: &[&Clause]) -> Result<Vec<f32>, String> {
        self.scorer
            .score_clauses_with_context(clauses, self.max_age)
            .map_err(|e| format!("Scoring failed: {}", e))
    }

    /// Generate a random float in [0, 1)
    fn next_random(&mut self) -> f64 {
        // Simple LCG: x_{n+1} = (a * x_n + c) mod m
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.rng_state >> 33) as f64 / (1u64 << 31) as f64
    }

    /// Sample an index from scores treated as logits (softmax sampling).
    fn sample_from_logits(&mut self, logits: &[f32]) -> usize {
        if logits.is_empty() {
            return 0;
        }
        if logits.len() == 1 {
            return 0;
        }

        // Compute softmax probabilities
        // First find max for numerical stability
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let exp_scores: Vec<f64> = logits
            .iter()
            .map(|&x| ((x - max_logit) as f64).exp())
            .collect();

        let sum: f64 = exp_scores.iter().sum();
        let probs: Vec<f64> = exp_scores.iter().map(|&e| e / sum).collect();

        // Sample from the distribution
        let r = self.next_random();
        let mut cumsum = 0.0;

        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return i;
            }
        }

        // Fallback to last index (shouldn't happen with proper probabilities)
        logits.len() - 1
    }
}

impl ClauseSelector for OnnxClauseSelector {
    fn select(&mut self, unprocessed: &mut VecDeque<usize>, clauses: &[Clause]) -> Option<usize> {
        if unprocessed.is_empty() {
            return None;
        }

        // Collect clause references for scoring
        let clause_refs: Vec<&Clause> = unprocessed.iter().map(|&idx| &clauses[idx]).collect();

        // Score clauses using the model
        match self.score_clauses(&clause_refs) {
            Ok(scores) => {
                // Sample from scores treated as logits
                let selected_idx = self.sample_from_logits(&scores);

                // Remove and return the selected clause
                unprocessed.remove(selected_idx)
            }
            Err(e) => {
                // Log error and fall back to FIFO
                eprintln!("ONNX scoring failed: {}, using FIFO fallback", e);
                unprocessed.pop_front()
            }
        }
    }

    fn name(&self) -> &str {
        "OnnxModel"
    }
}
