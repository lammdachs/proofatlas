//! Proof state representation

use serde::{Serialize, Deserialize};
use crate::core::logic::Clause;

/// Represents a proof state with processed and unprocessed clauses
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProofState {
    pub processed: Vec<Clause>,
    pub unprocessed: Vec<Clause>,
}

impl ProofState {
    /// Create a new proof state
    pub fn new(processed: Vec<Clause>, unprocessed: Vec<Clause>) -> Self {
        ProofState {
            processed,
            unprocessed,
        }
    }
    
    /// Create an empty proof state
    pub fn empty() -> Self {
        ProofState {
            processed: Vec::new(),
            unprocessed: Vec::new(),
        }
    }
    
    /// Get all clauses (processed + unprocessed)
    pub fn all_clauses(&self) -> Vec<&Clause> {
        self.processed.iter()
            .chain(self.unprocessed.iter())
            .collect()
    }
    
    /// Check if the state contains the empty clause (contradiction)
    pub fn contains_empty_clause(&self) -> bool {
        self.all_clauses().into_iter().any(|clause| clause.is_empty())
    }
    
    /// Add a clause to the processed set
    pub fn add_processed(&mut self, clause: Clause) {
        self.processed.push(clause);
    }
    
    /// Add a clause to the unprocessed set
    pub fn add_unprocessed(&mut self, clause: Clause) {
        self.unprocessed.push(clause);
    }
    
    /// Add multiple clauses to the unprocessed set
    pub fn add_unprocessed_many(&mut self, clauses: Vec<Clause>) {
        self.unprocessed.extend(clauses);
    }
    
    /// Move a clause from unprocessed to processed by index
    /// Returns the moved clause if successful
    pub fn move_to_processed(&mut self, index: usize) -> Option<Clause> {
        if index < self.unprocessed.len() {
            let clause = self.unprocessed.remove(index);
            self.processed.push(clause.clone());
            Some(clause)
        } else {
            None
        }
    }
    
    /// Get the number of processed clauses
    pub fn num_processed(&self) -> usize {
        self.processed.len()
    }
    
    /// Get the number of unprocessed clauses
    pub fn num_unprocessed(&self) -> usize {
        self.unprocessed.len()
    }
    
    /// Get the total number of clauses
    pub fn num_clauses(&self) -> usize {
        self.processed.len() + self.unprocessed.len()
    }
}

// Include tests
#[cfg(test)]
#[path = "state_tests.rs"]
mod tests;