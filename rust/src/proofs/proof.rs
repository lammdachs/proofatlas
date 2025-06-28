//! Proof representation with history of states and selected clauses

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use serde_json::Value;

use crate::core::logic::Clause;
use super::state::ProofState;

/// Result of applying an inference rule
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuleApplication {
    pub rule_name: String,
    pub parents: Vec<usize>,
    pub generated_clauses: Vec<Clause>,
    pub deleted_clause_indices: Vec<usize>,
    pub metadata: HashMap<String, Value>,
}

impl RuleApplication {
    /// Create a new rule application
    pub fn new(rule_name: String, parents: Vec<usize>) -> Self {
        RuleApplication {
            rule_name,
            parents,
            generated_clauses: Vec::new(),
            deleted_clause_indices: Vec::new(),
            metadata: HashMap::new(),
        }
    }
    
    /// Add generated clauses
    pub fn with_generated(mut self, clauses: Vec<Clause>) -> Self {
        self.generated_clauses = clauses;
        self
    }
    
    /// Add deleted clause indices
    pub fn with_deleted(mut self, indices: Vec<usize>) -> Self {
        self.deleted_clause_indices = indices;
        self
    }
    
    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// A single step in a proof
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProofStep {
    pub state: ProofState,
    pub selected_clause: Option<usize>,
    pub applied_rules: Vec<RuleApplication>,
    pub metadata: HashMap<String, Value>,
}

impl ProofStep {
    /// Create a new proof step
    pub fn new(state: ProofState) -> Self {
        ProofStep {
            state,
            selected_clause: None,
            applied_rules: Vec::new(),
            metadata: HashMap::new(),
        }
    }
    
    /// Set the selected clause
    pub fn with_selected_clause(mut self, index: usize) -> Self {
        self.selected_clause = Some(index);
        self
    }
    
    /// Add applied rules
    pub fn with_applied_rules(mut self, rules: Vec<RuleApplication>) -> Self {
        self.applied_rules = rules;
        self
    }
    
    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// A proof consisting of a sequence of proof steps
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Proof {
    pub steps: Vec<ProofStep>,
}

impl Proof {
    /// Create a new proof with an initial state
    pub fn new(initial_state: ProofState) -> Self {
        Proof {
            steps: vec![ProofStep::new(initial_state)],
        }
    }
    
    /// Create an empty proof
    pub fn empty() -> Self {
        Proof::new(ProofState::empty())
    }
    
    /// Get the initial proof state
    pub fn initial_state(&self) -> Option<&ProofState> {
        self.steps.first().map(|step| &step.state)
    }
    
    /// Get the final proof state
    pub fn final_state(&self) -> Option<&ProofState> {
        self.steps.last().map(|step| &step.state)
    }
    
    /// Get the length of the proof (number of inference steps)
    pub fn length(&self) -> usize {
        if self.steps.is_empty() {
            return 0;
        }
        
        // If last step has no selection, don't count it as an inference step
        if self.steps.last().unwrap().selected_clause.is_none() {
            self.steps.len().saturating_sub(1)
        } else {
            self.steps.len()
        }
    }
    
    /// Add a new step to the proof
    pub fn add_step(&mut self, step: ProofStep) {
        // If the last step has no selected clause, replace it
        if let Some(last) = self.steps.last() {
            if last.selected_clause.is_none() {
                self.steps.pop();
            }
        }
        
        self.steps.push(step);
    }
    
    /// Finalize the proof by ensuring the last step has no selection
    pub fn finalize(&mut self, final_state: ProofState) {
        // Remove any existing final step with no selection
        if let Some(last) = self.steps.last() {
            if last.selected_clause.is_none() {
                self.steps.pop();
            }
        }
        
        // Add final step
        self.steps.push(ProofStep::new(final_state));
    }
    
    /// Get a specific step by index
    pub fn get_step(&self, index: usize) -> Option<&ProofStep> {
        self.steps.get(index)
    }
    
    /// Get all selected clauses in order
    pub fn get_selected_clauses(&self) -> Vec<usize> {
        self.steps.iter()
            .filter_map(|step| step.selected_clause)
            .collect()
    }
    
    /// Check if the proof found a contradiction
    pub fn found_contradiction(&self) -> bool {
        self.final_state()
            .map(|state| state.contains_empty_clause())
            .unwrap_or(false)
    }
    
    /// Get metadata history for a specific key
    pub fn get_metadata_history(&self, key: &str) -> Vec<&Value> {
        self.steps.iter()
            .filter_map(|step| step.metadata.get(key))
            .collect()
    }
}

// Include tests
#[cfg(test)]
#[path = "proof_tests.rs"]
mod tests;