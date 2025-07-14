//! Custom saturation with detailed tracing

use crate::core::{Clause, Proof, ProofStep, CNFFormula};
use crate::inference::{resolution, factoring, superposition, equality_resolution, equality_factoring, InferenceResult};
use crate::selection::{ClauseSelector, LiteralSelector, AgeWeightRatioSelector, SelectAll, SelectMaxWeight};
use crate::parser::orient_equalities::orient_clause_equalities;
use super::subsumption::{is_subsumed, has_duplicate};
use super::{SaturationConfig, SaturationResult, LiteralSelectionStrategy};
use std::collections::{HashSet, VecDeque};
use std::time::Instant;

/// Custom saturation that prints given clause selection
pub fn custom_saturate_with_trace(formula: CNFFormula, config: SaturationConfig) -> SaturationResult {
    let mut clauses = Vec::new();
    let mut unprocessed = VecDeque::new();
    
    // Add initial clauses with IDs
    for (i, mut clause) in formula.clauses.into_iter().enumerate() {
        clause.id = Some(i);
        clauses.push(clause);
        unprocessed.push_back(i);
    }
    
    // Create selectors
    let literal_selector: Box<dyn LiteralSelector> = match config.literal_selection {
        LiteralSelectionStrategy::SelectAll => Box::new(SelectAll),
        LiteralSelectionStrategy::SelectMaxWeight => Box::new(SelectMaxWeight::new()),
    };
    
    let mut clause_selector: Box<dyn ClauseSelector> = Box::new(AgeWeightRatioSelector::new(1, 5));
    
    let mut processed = HashSet::new();
    let mut proof_steps = Vec::new();
    
    let start_time = Instant::now();
    let mut iterations = 0;
    let mut steps = 0;
    
    while let Some(given_idx) = clause_selector.select(&mut unprocessed, &clauses) {
        // Check limits
        if let Some(limit) = config.step_limit {
            if steps >= limit {
                return SaturationResult::ResourceLimit(proof_steps);
            }
        }
        
        if iterations >= config.max_iterations {
            return SaturationResult::ResourceLimit(proof_steps);
        }
        if clauses.len() >= config.max_clauses {
            return SaturationResult::ResourceLimit(proof_steps);
        }
        if start_time.elapsed() > config.timeout {
            return SaturationResult::Timeout;
        }
        
        iterations += 1;
        steps += 1;
        
        // Print which clause is selected as given
        let given_clause = &clauses[given_idx];
        println!("Step {}: Given clause [{}]: {}", steps, given_idx, given_clause);
        
        // Check if it's the empty clause
        if given_clause.is_empty() {
            return SaturationResult::Proof(Proof {
                steps: proof_steps,
                empty_clause_idx: given_idx,
            });
        }
        
        // Generate new clauses by inference with processed clauses
        let mut new_inferences = Vec::new();
        
        // Factoring on given clause
        new_inferences.extend(factoring(given_clause, given_idx, literal_selector.as_ref()));
        
        // Equality resolution on given clause
        new_inferences.extend(equality_resolution(given_clause, given_idx, literal_selector.as_ref()));
        
        // Equality factoring on given clause (if superposition is enabled)
        if config.use_superposition {
            new_inferences.extend(equality_factoring(given_clause, given_idx, literal_selector.as_ref()));
        }
        
        // Inferences with each processed clause
        for &processed_idx in &processed {
            let processed_clause = &clauses[processed_idx];
            
            // Resolution
            new_inferences.extend(resolution(given_clause, processed_clause, given_idx, processed_idx, literal_selector.as_ref()));
            new_inferences.extend(resolution(processed_clause, given_clause, processed_idx, given_idx, literal_selector.as_ref()));
            
            // Superposition (if enabled)
            if config.use_superposition {
                new_inferences.extend(superposition(given_clause, processed_clause, given_idx, processed_idx, literal_selector.as_ref()));
                new_inferences.extend(superposition(processed_clause, given_clause, processed_idx, given_idx, literal_selector.as_ref()));
            }
        }
        
        // Add given clause to processed
        processed.insert(given_idx);
        
        // Process new inferences
        let mut new_clauses = Vec::new();
        for inference in new_inferences {
            if let Some(new_idx) = add_clause(inference, &mut clauses, &mut unprocessed, &mut proof_steps, &config) {
                new_clauses.push(new_idx);
                // Check if we derived empty clause
                if clauses[new_idx].is_empty() {
                    return SaturationResult::Proof(Proof {
                        steps: proof_steps,
                        empty_clause_idx: new_idx,
                    });
                }
            }
        }
        
        // Print new clauses generated
        if !new_clauses.is_empty() {
            println!("  Generated {} new clauses: {:?}", new_clauses.len(), new_clauses);
        }
    }
    
    // No more clauses to process
    SaturationResult::Saturated
}

/// Add a new clause if it's not redundant
fn add_clause(
    inference: InferenceResult, 
    clauses: &mut Vec<Clause>,
    unprocessed: &mut VecDeque<usize>,
    proof_steps: &mut Vec<ProofStep>,
    config: &SaturationConfig
) -> Option<usize> {
    // Check clause size limit
    if inference.conclusion.literals.len() > config.max_clause_size {
        return None;
    }
    
    // Check if tautology
    if inference.conclusion.is_tautology() {
        return None;
    }
    
    // Check for forward subsumption
    if is_subsumed(&inference.conclusion, clauses) {
        return None;
    }
    
    // Check for duplicates
    if has_duplicate(&inference.conclusion, clauses) {
        return None;
    }
    
    // Add the clause
    let new_idx = clauses.len();
    let mut clause_with_id = inference.conclusion.clone();
    clause_with_id.id = Some(new_idx);
    
    // Orient equalities before adding
    orient_clause_equalities(&mut clause_with_id);
    
    clauses.push(clause_with_id.clone());
    unprocessed.push_back(new_idx);
    
    // Record proof step with oriented clause
    let mut oriented_inference = inference;
    oriented_inference.conclusion = clause_with_id;
    proof_steps.push(ProofStep {
        inference: oriented_inference,
        clause_idx: new_idx,
    });
    
    Some(new_idx)
}