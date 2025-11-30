//! Clause selection strategies for the given clause algorithm
//!
//! These strategies determine which clause to select next from the
//! unprocessed set during saturation.

use crate::core::Clause;
#[cfg(feature = "onnx")]
use crate::ml::ClauseScorer;
use std::collections::VecDeque;
#[cfg(feature = "onnx")]
use std::path::Path;

/// Trait for clause selection strategies
pub trait ClauseSelector: Send + Sync {
    /// Select the next clause from the unprocessed set
    /// Returns the index of the selected clause, or None if empty
    fn select(&mut self, unprocessed: &mut VecDeque<usize>, clauses: &[Clause]) -> Option<usize>;

    /// Get the name of this selection strategy
    fn name(&self) -> &str;
}

/// Select smallest clauses first
pub struct SizeBasedSelector;

impl ClauseSelector for SizeBasedSelector {
    fn select(&mut self, unprocessed: &mut VecDeque<usize>, clauses: &[Clause]) -> Option<usize> {
        if unprocessed.is_empty() {
            return None;
        }

        // Find the index with the smallest clause
        let mut best_idx = 0;
        let mut best_size = clauses[unprocessed[0]].literals.len();

        for (i, &clause_idx) in unprocessed.iter().enumerate() {
            let size = clauses[clause_idx].literals.len();
            if size < best_size {
                best_size = size;
                best_idx = i;
            }
        }

        // Remove and return the selected clause
        unprocessed.remove(best_idx)
    }

    fn name(&self) -> &str {
        "SizeBased"
    }
}

/// Select oldest clauses first (FIFO - First In First Out)
pub struct AgeBasedSelector;

impl ClauseSelector for AgeBasedSelector {
    fn select(&mut self, unprocessed: &mut VecDeque<usize>, _clauses: &[Clause]) -> Option<usize> {
        unprocessed.pop_front()
    }

    fn name(&self) -> &str {
        "AgeBased"
    }
}

/// Weighted combination of size and age
pub struct WeightedSelector {
    size_weight: f64,
    age_weight: f64,
}

impl WeightedSelector {
    pub fn new(size_weight: f64, age_weight: f64) -> Self {
        WeightedSelector {
            size_weight,
            age_weight,
        }
    }
}

impl ClauseSelector for WeightedSelector {
    fn select(&mut self, unprocessed: &mut VecDeque<usize>, clauses: &[Clause]) -> Option<usize> {
        if unprocessed.is_empty() {
            return None;
        }

        let mut best_idx = 0;
        let mut best_score = f64::MAX;

        for (i, &clause_idx) in unprocessed.iter().enumerate() {
            let clause = &clauses[clause_idx];
            let size = clause.literals.len() as f64;
            let age = clause.id.unwrap_or(usize::MAX) as f64;

            let score = self.size_weight * size + self.age_weight * age;

            if score < best_score {
                best_score = score;
                best_idx = i;
            }
        }

        unprocessed.remove(best_idx)
    }

    fn name(&self) -> &str {
        "Weighted"
    }
}

/// Age-Weight Ratio selector
/// Alternates between selecting by age and by weight (clause size)
/// Default ratio is 1:5 (age:weight)
pub struct AgeWeightRatioSelector {
    age_picks: usize,
    weight_picks: usize,
    counter: usize,
}

impl AgeWeightRatioSelector {
    /// Create with specified ratio
    pub fn new(age_picks: usize, weight_picks: usize) -> Self {
        AgeWeightRatioSelector {
            age_picks,
            weight_picks,
            counter: 0,
        }
    }

    /// Create with default 1:5 ratio
    pub fn default() -> Self {
        Self::new(1, 5)
    }

    /// Calculate the weight (symbol count) of a clause
    fn clause_weight(clause: &Clause) -> usize {
        clause
            .literals
            .iter()
            .map(|lit| {
                // Count predicate symbol + all argument symbols
                1 + lit
                    .atom
                    .args
                    .iter()
                    .map(|term| Self::term_symbol_count(term))
                    .sum::<usize>()
            })
            .sum()
    }

    /// Count symbols in a term
    fn term_symbol_count(term: &crate::core::Term) -> usize {
        use crate::core::Term;
        match term {
            Term::Variable(_) => 1,
            Term::Constant(_) => 1,
            Term::Function(_, args) => {
                1 + args
                    .iter()
                    .map(|t| Self::term_symbol_count(t))
                    .sum::<usize>()
            }
        }
    }

    fn select_by_age(&self, _unprocessed: &VecDeque<usize>, _clauses: &[Clause]) -> usize {
        // FIFO: always select the first clause
        0
    }

    fn select_by_weight(&self, unprocessed: &VecDeque<usize>, clauses: &[Clause]) -> usize {
        let mut best_idx = 0;
        let mut best_weight = Self::clause_weight(&clauses[unprocessed[0]]);

        for (i, &clause_idx) in unprocessed.iter().enumerate() {
            let weight = Self::clause_weight(&clauses[clause_idx]);
            if weight < best_weight {
                best_weight = weight;
                best_idx = i;
            }
        }

        best_idx
    }
}

impl ClauseSelector for AgeWeightRatioSelector {
    fn select(&mut self, unprocessed: &mut VecDeque<usize>, clauses: &[Clause]) -> Option<usize> {
        if unprocessed.is_empty() {
            return None;
        }

        let total_ratio = self.age_picks + self.weight_picks;
        let select_by_age = self.counter < self.age_picks;

        let best_idx = if select_by_age {
            self.select_by_age(unprocessed, clauses)
        } else {
            self.select_by_weight(unprocessed, clauses)
        };

        // Update counter
        self.counter = (self.counter + 1) % total_ratio;

        // Remove and return the selected clause
        unprocessed.remove(best_idx)
    }

    fn name(&self) -> &str {
        "AgeWeightRatio"
    }
}

impl Default for AgeWeightRatioSelector {
    fn default() -> Self {
        Self::default()
    }
}

/// Probabilistic Age/Weight selector that acts as a baseline "model".
///
/// With probability `p`, selects by age (FIFO - oldest first).
/// With probability `1-p`, selects by weight (smallest symbol count first).
///
/// This can be used as a baseline to compare against learned models.
/// The probability parameter allows tuning the exploration/exploitation tradeoff.
pub struct ProbabilisticAgeWeightSelector {
    /// Probability of selecting by age (0.0 to 1.0)
    age_probability: f64,
    /// Random number generator state (simple LCG for reproducibility)
    rng_state: u64,
}

impl ProbabilisticAgeWeightSelector {
    /// Create with specified age probability
    ///
    /// # Arguments
    /// * `age_probability` - Probability of selecting by age (0.0 = always weight, 1.0 = always age)
    pub fn new(age_probability: f64) -> Self {
        ProbabilisticAgeWeightSelector {
            age_probability: age_probability.clamp(0.0, 1.0),
            rng_state: 12345, // Default seed
        }
    }

    /// Create with specified age probability and seed
    pub fn with_seed(age_probability: f64, seed: u64) -> Self {
        ProbabilisticAgeWeightSelector {
            age_probability: age_probability.clamp(0.0, 1.0),
            rng_state: seed,
        }
    }

    /// Create with default 0.5 probability (equal chance age/weight)
    pub fn balanced() -> Self {
        Self::new(0.5)
    }

    /// Generate a random float in [0, 1)
    fn next_random(&mut self) -> f64 {
        // Simple LCG: x_{n+1} = (a * x_n + c) mod m
        // Using parameters from Numerical Recipes
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.rng_state >> 33) as f64 / (1u64 << 31) as f64
    }

    /// Calculate the weight (symbol count) of a clause
    fn clause_weight(clause: &Clause) -> usize {
        clause.symbol_count()
    }

    fn select_by_age(&self, _unprocessed: &VecDeque<usize>, _clauses: &[Clause]) -> usize {
        // FIFO: always select the first clause (oldest)
        0
    }

    fn select_by_weight(&self, unprocessed: &VecDeque<usize>, clauses: &[Clause]) -> usize {
        let mut best_idx = 0;
        let mut best_weight = Self::clause_weight(&clauses[unprocessed[0]]);

        for (i, &clause_idx) in unprocessed.iter().enumerate() {
            let weight = Self::clause_weight(&clauses[clause_idx]);
            if weight < best_weight {
                best_weight = weight;
                best_idx = i;
            }
        }

        best_idx
    }

    /// Score all clauses and return scores.
    ///
    /// This provides an interface similar to the ML-based ClauseScorer.
    /// Scores are based on a combination of age and weight, with lower scores
    /// being better (matching the selection preference).
    ///
    /// # Returns
    /// Vector of scores where lower = more likely to be selected
    pub fn score_clauses(&self, clauses: &[&Clause]) -> Vec<f64> {
        if clauses.is_empty() {
            return Vec::new();
        }

        // Find max values for normalization
        let max_weight = clauses
            .iter()
            .map(|c| Self::clause_weight(c))
            .max()
            .unwrap_or(1) as f64;
        let max_age = clauses
            .iter()
            .map(|c| c.age)
            .max()
            .unwrap_or(1)
            .max(1) as f64; // Ensure non-zero for division

        // Score = p * normalized_age + (1-p) * normalized_weight
        // Lower is better (younger age and smaller weight preferred)
        clauses
            .iter()
            .map(|clause| {
                let norm_age = clause.age as f64 / max_age;
                let norm_weight = Self::clause_weight(clause) as f64 / max_weight.max(1.0);

                self.age_probability * norm_age + (1.0 - self.age_probability) * norm_weight
            })
            .collect()
    }
}

impl ClauseSelector for ProbabilisticAgeWeightSelector {
    fn select(&mut self, unprocessed: &mut VecDeque<usize>, clauses: &[Clause]) -> Option<usize> {
        if unprocessed.is_empty() {
            return None;
        }

        // Decide whether to select by age or weight
        let use_age = self.next_random() < self.age_probability;

        let best_idx = if use_age {
            self.select_by_age(unprocessed, clauses)
        } else {
            self.select_by_weight(unprocessed, clauses)
        };

        // Remove and return the selected clause
        unprocessed.remove(best_idx)
    }

    fn name(&self) -> &str {
        "ProbabilisticAgeWeight"
    }
}

impl Default for ProbabilisticAgeWeightSelector {
    fn default() -> Self {
        Self::balanced()
    }
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
#[cfg(feature = "onnx")]
pub struct OnnxClauseSelector {
    /// The underlying clause scorer
    scorer: ClauseScorer,
    /// Maximum age for normalization (clauses with higher age get clipped)
    max_age: usize,
    /// Fallback selector to use if scoring fails
    fallback: AgeWeightRatioSelector,
    /// Random number generator state (simple LCG)
    rng_state: u64,
}

#[cfg(feature = "onnx")]
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
            fallback: AgeWeightRatioSelector::default(),
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
            fallback: AgeWeightRatioSelector::default(),
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

#[cfg(feature = "onnx")]
impl ClauseSelector for OnnxClauseSelector {
    fn select(&mut self, unprocessed: &mut VecDeque<usize>, clauses: &[Clause]) -> Option<usize> {
        if unprocessed.is_empty() {
            return None;
        }

        // Collect clause references for scoring
        let clause_refs: Vec<&Clause> = unprocessed.iter().map(|&idx| &clauses[idx]).collect();

        // Try to score clauses using the model
        match self.score_clauses(&clause_refs) {
            Ok(scores) => {
                // Sample from scores treated as logits
                let selected_idx = self.sample_from_logits(&scores);

                // Remove and return the selected clause
                unprocessed.remove(selected_idx)
            }
            Err(_) => {
                // Fall back to age-weight ratio if scoring fails
                self.fallback.select(unprocessed, clauses)
            }
        }
    }

    fn name(&self) -> &str {
        "OnnxModel"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Atom, Clause, Literal, PredicateSymbol, Term, Variable};

    fn make_clause(name: &str, num_args: usize) -> Clause {
        let args: Vec<Term> = (0..num_args)
            .map(|i| {
                Term::Variable(Variable {
                    name: format!("X{}", i),
                })
            })
            .collect();
        Clause::new(vec![Literal::positive(Atom {
            predicate: PredicateSymbol {
                name: name.to_string(),
                arity: num_args,
            },
            args,
        })])
    }

    #[test]
    fn test_probabilistic_always_age() {
        // With p=1.0, should always select by age (first in queue)
        let mut selector = ProbabilisticAgeWeightSelector::new(1.0);
        let clauses = vec![
            make_clause("P", 5), // Large clause, but should be selected first (age)
            make_clause("Q", 1), // Small clause
            make_clause("R", 3),
        ];

        let mut unprocessed: VecDeque<usize> = (0..3).collect();

        // Should always pick the first (oldest) clause
        assert_eq!(selector.select(&mut unprocessed, &clauses), Some(0));
        assert_eq!(selector.select(&mut unprocessed, &clauses), Some(1));
        assert_eq!(selector.select(&mut unprocessed, &clauses), Some(2));
    }

    #[test]
    fn test_probabilistic_always_weight() {
        // With p=0.0, should always select by weight (smallest first)
        let mut selector = ProbabilisticAgeWeightSelector::new(0.0);
        let clauses = vec![
            make_clause("P", 5), // Large clause
            make_clause("Q", 1), // Smallest clause
            make_clause("R", 3),
        ];

        let mut unprocessed: VecDeque<usize> = (0..3).collect();

        // Should pick smallest clause first (Q with 1 arg)
        let selected = selector.select(&mut unprocessed, &clauses);
        assert_eq!(selected, Some(1)); // Q is smallest
    }

    #[test]
    fn test_probabilistic_score_clauses() {
        let selector = ProbabilisticAgeWeightSelector::new(0.5);

        let mut clause1 = make_clause("P", 3);
        clause1.age = 10;
        let mut clause2 = make_clause("Q", 1);
        clause2.age = 5;
        let mut clause3 = make_clause("R", 5);
        clause3.age = 0;

        let clauses: Vec<&Clause> = vec![&clause1, &clause2, &clause3];
        let scores = selector.score_clauses(&clauses);

        assert_eq!(scores.len(), 3);

        // Clause3 has age=0 and largest weight, so it should have lowest age score
        // but highest weight score
        // Clause2 has smallest weight, so lowest weight score
        // The actual ranking depends on the balance

        // All scores should be in [0, 1] range since we normalize
        for score in &scores {
            assert!(*score >= 0.0 && *score <= 1.0, "Score out of range: {}", score);
        }
    }

    #[test]
    fn test_probabilistic_reproducibility() {
        // Same seed should give same sequence
        let mut selector1 = ProbabilisticAgeWeightSelector::with_seed(0.5, 42);
        let mut selector2 = ProbabilisticAgeWeightSelector::with_seed(0.5, 42);

        let clauses = vec![
            make_clause("P", 5),
            make_clause("Q", 1),
            make_clause("R", 3),
        ];

        for _ in 0..10 {
            let mut unprocessed1: VecDeque<usize> = (0..3).collect();
            let mut unprocessed2: VecDeque<usize> = (0..3).collect();

            let result1 = selector1.select(&mut unprocessed1, &clauses);
            let result2 = selector2.select(&mut unprocessed2, &clauses);

            assert_eq!(result1, result2, "Same seed should produce same results");
        }
    }
}
