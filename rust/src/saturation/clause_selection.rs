//! Clause selection strategies for the saturation loop

use crate::core::Problem;

/// Trait for clause selection strategies
pub trait ClauseSelector: Send + Sync {
    /// Select the next clause to process from the unprocessed set
    fn select_clause(&self, unprocessed: &[usize], problem: &Problem) -> Option<usize>;
}

/// First-In-First-Out clause selection
pub struct FifoSelection;

impl ClauseSelector for FifoSelection {
    fn select_clause(&self, unprocessed: &[usize], _problem: &Problem) -> Option<usize> {
        unprocessed.first().copied()
    }
}

/// Select smallest clauses first
pub struct SmallestFirst;

impl ClauseSelector for SmallestFirst {
    fn select_clause(&self, unprocessed: &[usize], problem: &Problem) -> Option<usize> {
        unprocessed.iter()
            .min_by_key(|&&clause_idx| problem.clause_literals(clause_idx).len())
            .copied()
    }
}