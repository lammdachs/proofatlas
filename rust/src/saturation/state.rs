//! Main saturation state and algorithm

use crate::core::{Clause, Proof, ProofStep};
use crate::inference::{resolution, factoring, superposition, equality_resolution, equality_factoring, InferenceResult};
use crate::selection::{ClauseSelector, LiteralSelector, AgeWeightRatioSelector, SelectAll, SelectMaxWeight};
use crate::parser::orient_equalities::orient_clause_equalities;
use super::subsumption::{is_subsumed, has_duplicate};
use std::collections::{HashSet, VecDeque};
use std::time::{Duration, Instant};

/// Configuration for the saturation loop
#[derive(Debug, Clone)]
pub struct SaturationConfig {
    pub max_clauses: usize,
    pub max_iterations: usize,
    pub max_clause_size: usize,
    pub timeout: Duration,
    pub use_superposition: bool,
    pub literal_selection: LiteralSelectionStrategy,
    pub step_limit: Option<usize>,
}

/// Literal selection strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiteralSelectionStrategy {
    SelectAll,
    SelectMaxWeight,
}

impl Default for SaturationConfig {
    fn default() -> Self {
        SaturationConfig {
            max_clauses: 10000,
            max_iterations: 10000,
            max_clause_size: 100,
            timeout: Duration::from_secs(60),
            use_superposition: true,
            literal_selection: LiteralSelectionStrategy::SelectAll,
            step_limit: None,
        }
    }
}

/// Result of saturation
#[derive(Debug, Clone)]
pub enum SaturationResult {
    /// Empty clause derived - proof found
    Proof(Proof),
    /// Saturated without finding empty clause
    Saturated,
    /// Resource limit reached (includes proof steps so far)
    ResourceLimit(Vec<ProofStep>),
    /// Timeout reached
    Timeout,
}

/// Main saturation state
pub struct SaturationState {
    /// All clauses (processed and unprocessed)
    clauses: Vec<Clause>,
    /// Indices of processed clauses
    processed: HashSet<usize>,
    /// Queue of unprocessed clause indices
    unprocessed: VecDeque<usize>,
    /// Proof steps
    proof_steps: Vec<ProofStep>,
    /// Configuration
    config: SaturationConfig,
    /// Clause selector
    clause_selector: Box<dyn ClauseSelector>,
    /// Literal selector
    literal_selector: Box<dyn LiteralSelector>,
}

impl SaturationState {
    /// Create new saturation state from initial clauses
    pub fn new(initial_clauses: Vec<Clause>, config: SaturationConfig) -> Self {
        let mut clauses = Vec::new();
        let mut unprocessed = VecDeque::new();
        
        // Add initial clauses with IDs
        for (i, mut clause) in initial_clauses.into_iter().enumerate() {
            clause.id = Some(i);
            clauses.push(clause);
            unprocessed.push_back(i);
        }
        
        // Create literal selector based on configuration
        let literal_selector: Box<dyn LiteralSelector> = match config.literal_selection {
            LiteralSelectionStrategy::SelectAll => Box::new(SelectAll),
            LiteralSelectionStrategy::SelectMaxWeight => Box::new(SelectMaxWeight::new()),
        };
        
        SaturationState {
            clauses,
            processed: HashSet::new(),
            unprocessed,
            proof_steps: Vec::new(),
            config,
            clause_selector: Box::new(AgeWeightRatioSelector::default()),
            literal_selector,
        }
    }
    
    /// Set the literal selector
    pub fn set_literal_selector(&mut self, selector: Box<dyn LiteralSelector>) {
        self.literal_selector = selector;
    }
    
    /// Set the clause selector
    pub fn set_clause_selector(&mut self, selector: Box<dyn ClauseSelector>) {
        self.clause_selector = selector;
    }
    
    /// Run the saturation algorithm
    pub fn saturate(mut self) -> SaturationResult {
        let start_time = Instant::now();
        let mut iterations = 0;
        let mut steps = 0;
        
        while let Some(given_idx) = self.select_given_clause() {
            // Check step limit if specified
            if let Some(limit) = self.config.step_limit {
                if steps >= limit {
                    return SaturationResult::ResourceLimit(self.proof_steps.clone());
                }
            }
            
            // Check other limits
            if iterations >= self.config.max_iterations {
                return SaturationResult::ResourceLimit(self.proof_steps.clone());
            }
            if self.clauses.len() >= self.config.max_clauses {
                return SaturationResult::ResourceLimit(self.proof_steps.clone());
            }
            if start_time.elapsed() > self.config.timeout {
                return SaturationResult::Timeout;
            }
            
            iterations += 1;
            steps += 1;
            
            // Process the given clause
            let given_clause = &self.clauses[given_idx];
            
            // Check if it's the empty clause
            if given_clause.is_empty() {
                return SaturationResult::Proof(Proof {
                    steps: self.proof_steps.clone(),
                    empty_clause_idx: given_idx,
                });
            }
            
            // Generate new clauses by inference with processed clauses
            let new_inferences = self.generate_inferences(given_idx);
            
            // Add given clause to processed
            self.processed.insert(given_idx);
            
            // Process new inferences - deduplicate within the batch first
            let mut seen_in_batch = HashSet::new();
            let mut unique_inferences = Vec::new();
            
            for inference in new_inferences {
                // Create a normalized string representation for comparison
                let clause_str = format!("{}", inference.conclusion);
                if seen_in_batch.insert(clause_str) {
                    unique_inferences.push(inference);
                }
            }
            
            for inference in unique_inferences {
                if let Some(new_idx) = self.add_clause(inference) {
                    // Check if we derived empty clause
                    if self.clauses[new_idx].is_empty() {
                        return SaturationResult::Proof(Proof {
                            steps: self.proof_steps.clone(),
                            empty_clause_idx: new_idx,
                        });
                    }
                }
            }
        }
        
        // No more clauses to process
        SaturationResult::Saturated
    }
    
    /// Select the next given clause using the configured selector
    fn select_given_clause(&mut self) -> Option<usize> {
        self.clause_selector.select(&mut self.unprocessed, &self.clauses)
    }
    
    /// Generate all inferences between given clause and processed clauses
    fn generate_inferences(&self, given_idx: usize) -> Vec<InferenceResult> {
        let mut results = Vec::new();
        let given_clause = &self.clauses[given_idx];
        let selector = self.literal_selector.as_ref();
        
        // Factoring on given clause
        results.extend(factoring(given_clause, given_idx, selector));
        
        // Equality resolution on given clause
        results.extend(equality_resolution(given_clause, given_idx, selector));
        
        // Equality factoring on given clause (if superposition is enabled)
        if self.config.use_superposition {
            results.extend(equality_factoring(given_clause, given_idx, selector));
        }
        
        // Inferences with each processed clause
        for &processed_idx in &self.processed {
            let processed_clause = &self.clauses[processed_idx];
            
            // Resolution
            results.extend(resolution(given_clause, processed_clause, given_idx, processed_idx, selector));
            results.extend(resolution(processed_clause, given_clause, processed_idx, given_idx, selector));
            
            // Superposition (if enabled)
            if self.config.use_superposition {
                results.extend(superposition(given_clause, processed_clause, given_idx, processed_idx, selector));
                results.extend(superposition(processed_clause, given_clause, processed_idx, given_idx, selector));
            }
        }
        
        results
    }
    
    /// Add a new clause if it's not redundant
    fn add_clause(&mut self, inference: InferenceResult) -> Option<usize> {
        // Check clause size limit
        if inference.conclusion.literals.len() > self.config.max_clause_size {
            return None;
        }
        
        // Check if tautology
        if inference.conclusion.is_tautology() {
            return None;
        }
        
        // Check for forward subsumption
        if is_subsumed(&inference.conclusion, &self.clauses) {
            return None;
        }
        
        // Check for duplicates
        if has_duplicate(&inference.conclusion, &self.clauses) {
            return None;
        }
        
        // Add the clause
        let new_idx = self.clauses.len();
        let mut clause_with_id = inference.conclusion.clone();
        clause_with_id.id = Some(new_idx);
        
        // Orient equalities before adding
        orient_clause_equalities(&mut clause_with_id);
        
        self.clauses.push(clause_with_id.clone());
        self.unprocessed.push_back(new_idx);
        
        // Record proof step with oriented clause
        let mut oriented_inference = inference;
        oriented_inference.conclusion = clause_with_id;
        self.proof_steps.push(ProofStep {
            inference: oriented_inference,
            clause_idx: new_idx,
        });
        
        Some(new_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Term, Constant, Variable, PredicateSymbol, Atom, Literal, CNFFormula};
    
    #[test]
    fn test_simple_proof() {
        // P(a)
        // ~P(X) ∨ Q(X)
        // ~Q(a)
        // Should derive empty clause
        
        let p = PredicateSymbol { name: "P".to_string(), arity: 1 };
        let q = PredicateSymbol { name: "Q".to_string(), arity: 1 };
        let a = Term::Constant(Constant { name: "a".to_string() });
        let x = Term::Variable(Variable { name: "X".to_string() });
        
        let clauses = vec![
            Clause::new(vec![
                Literal::positive(Atom { predicate: p.clone(), args: vec![a.clone()] })
            ]),
            Clause::new(vec![
                Literal::negative(Atom { predicate: p.clone(), args: vec![x.clone()] }),
                Literal::positive(Atom { predicate: q.clone(), args: vec![x.clone()] })
            ]),
            Clause::new(vec![
                Literal::negative(Atom { predicate: q.clone(), args: vec![a.clone()] })
            ]),
        ];
        
        let formula = CNFFormula { clauses };
        let result = crate::saturation::saturate(formula, SaturationConfig::default());
        
        match result {
            SaturationResult::Proof(_) => {}, // Expected
            _ => panic!("Expected to find proof"),
        }
    }
}