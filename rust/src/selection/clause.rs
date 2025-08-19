//! Clause selection strategies for the given clause algorithm
//!
//! These strategies determine which clause to select next from the
//! unprocessed set during saturation.

use crate::core::Clause;
use std::collections::VecDeque;

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
