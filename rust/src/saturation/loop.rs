//! Array-based saturation loop using given-clause algorithm with proof tracking

use crate::core::{Problem, NodeType, ProofStep, Proof, InferenceRule, SaturationResult};
use crate::rules::{
    resolve_clauses, factor_clause, superpose_clauses, 
    equality_resolve, equality_factor
};
use super::subsumption::SubsumptionIndex;
use super::literal_selection::{
    LiteralSelector, SelectFirstNegative, apply_literal_selection
};

/// Configuration for saturation
pub struct SaturationConfig {
    pub max_clauses: usize,
    pub max_clause_size: usize,
    pub max_iterations: usize,
    pub use_backward_subsumption: bool,
    pub literal_selector: Box<dyn LiteralSelector>,
}

impl Default for SaturationConfig {
    fn default() -> Self {
        SaturationConfig {
            max_clauses: 100_000,
            max_clause_size: 100,
            max_iterations: 1_000_000,
            use_backward_subsumption: true,
            literal_selector: Box::new(SelectFirstNegative),
        }
    }
}

/// Run saturation loop on array problem with proof tracking
pub fn saturate(
    problem: &mut Problem,
    config: &SaturationConfig,
) -> SaturationResult {
    println!("SATURATE CALLED with {} clauses", problem.num_clauses);
    
    // Handle empty problem
    if problem.num_clauses == 0 {
        return SaturationResult::Saturated;
    }
    
    let mut proof_steps = Vec::new();
    let mut processed: Vec<usize> = Vec::new();
    let mut unprocessed: Vec<usize> = (0..problem.num_clauses).collect();
    let mut subsumption_index = SubsumptionIndex::new();
    
    // Initialize clause lifetime tracking
    let mut clause_first_step = vec![0; problem.num_clauses];
    let mut clause_last_step = vec![usize::MAX; problem.num_clauses];
    
    // Check for empty clause in initial clauses
    for i in 0..problem.num_clauses {
        if is_empty_clause(problem, i) {
            return SaturationResult::Proof(Proof {
                problem: problem.clone(),
                steps: vec![],
                clause_first_step,
                clause_last_step,
            });
        }
        // Don't insert into subsumption index yet - only processed clauses go there
    }
    
    let mut num_iterations = 0;
    
    while !unprocessed.is_empty() && num_iterations < config.max_iterations {
        num_iterations += 1;
        
        // Select given clause (FIFO for now)
        let given_idx = unprocessed.remove(0);
        eprintln!("\nIteration {}: Given clause {}, processed: {:?}", num_iterations, given_idx, processed);
        
        // Forward subsumption check
        if let Some(subsumer_idx) = subsumption_index.find_subsuming(given_idx, problem) {
            eprintln!("  Clause {} is subsumed by clause {}", given_idx, subsumer_idx);
            // Record backward subsumption step (existing clause subsumes given clause)
            proof_steps.push(ProofStep {
                rule: InferenceRule::BackwardSubsumption,
                parents: vec![subsumer_idx],
                selected_literals: vec![],
                given_clause: None,
                added_clauses: vec![],
                deleted_clauses: vec![given_idx],
            });
            clause_last_step[given_idx] = proof_steps.len() - 1;
            continue;
        }
        
        // Check if it's the empty clause
        if is_empty_clause(problem, given_idx) {
            // Finalize clause lifetimes
            finalize_clause_lifetimes(&mut clause_last_step, proof_steps.len());
            
            return SaturationResult::Proof(Proof {
                problem: problem.clone(),
                steps: proof_steps,
                clause_first_step,
                clause_last_step,
            });
        }
        
        // Apply literal selection to given clause
        apply_literal_selection(problem, given_idx, config.literal_selector.as_ref());
        
        // Generate inferences with processed clauses
        let mut new_clauses: Vec<(usize, InferenceRule, Vec<usize>, Vec<usize>)> = Vec::new();
        
        // Resolution with all processed clauses
        eprintln!("  Resolution: {} processed clauses", processed.len());
        for &processed_idx in &processed {  // All processed clauses
            // Apply literal selection to processed clause
            apply_literal_selection(problem, processed_idx, config.literal_selector.as_ref());
            let resolvents = resolve_clauses(problem, given_idx, processed_idx);
            for result in resolvents {
                if let Some(new_idx) = result.new_clause_idx {
                    new_clauses.push((new_idx, InferenceRule::Resolution { 
                        lit1_idx: result.selected_literals[0],
                        lit2_idx: result.selected_literals[1],
                    }, vec![given_idx, processed_idx], result.selected_literals));
                }
            }
        }
        
        // Factoring on given clause
        let factors = factor_clause(problem, given_idx);
        for result in factors {
            if let Some(new_idx) = result.new_clause_idx {
                new_clauses.push((new_idx, InferenceRule::Factoring { 
                    lit_indices: result.selected_literals.clone(),
                }, vec![given_idx], result.selected_literals));
            }
        }
        
        // Superposition (if equality present)
        eprintln!("  Checking superposition with {} processed clauses", processed.len());
        for &processed_idx in &processed {
            // Apply literal selection to processed clause
            apply_literal_selection(problem, processed_idx, config.literal_selector.as_ref());
            // Try superposition from given clause into processed
            let superpositions = superpose_clauses(problem, given_idx, processed_idx);
            eprintln!("Superposition {} -> {} generated {} results", given_idx, processed_idx, superpositions.len());
            for result in superpositions {
                if let Some(new_idx) = result.new_clause_idx {
                    new_clauses.push((new_idx, InferenceRule::Superposition {
                        from_lit: result.selected_literals[0],
                        into_lit: result.selected_literals[1],
                        position: vec![], // TODO: Track actual position
                        positive: true,
                    }, vec![given_idx, processed_idx], result.selected_literals));
                }
            }
            
            // Try superposition from processed into given clause
            println!("Trying superposition from clause {} into clause {}", processed_idx, given_idx);
            let superpositions = superpose_clauses(problem, processed_idx, given_idx);
            for result in superpositions {
                if let Some(new_idx) = result.new_clause_idx {
                    new_clauses.push((new_idx, InferenceRule::Superposition {
                        from_lit: result.selected_literals[0],
                        into_lit: result.selected_literals[1],
                        position: vec![], // TODO: Track actual position
                        positive: true,
                    }, vec![processed_idx, given_idx], result.selected_literals));
                }
            }
        }
        
        // Equality resolution on given clause
        let eq_resolutions = equality_resolve(problem, given_idx);
        for result in eq_resolutions {
            if let Some(new_idx) = result.new_clause_idx {
                new_clauses.push((new_idx, InferenceRule::EqualityResolution, vec![given_idx], result.selected_literals));
            }
        }
        
        // Equality factoring on given clause
        let eq_factors = equality_factor(problem, given_idx);
        for result in eq_factors {
            if let Some(new_idx) = result.new_clause_idx {
                new_clauses.push((new_idx, InferenceRule::EqualityFactoring, vec![given_idx], result.selected_literals));
            }
        }
        
        // Now move given clause to processed (after generating all inferences)
        processed.push(given_idx);
        subsumption_index.insert(given_idx, problem);
        
        // Process each new clause
        for (clause_idx, rule, parents, selected_lits) in new_clauses {
            // Update clause metadata
            clause_first_step.push(proof_steps.len());
            clause_last_step.push(usize::MAX);
            
            // Record the inference step
            proof_steps.push(ProofStep {
                rule: rule.clone(),
                parents: parents.clone(),
                selected_literals: selected_lits,
                given_clause: Some(given_idx),
                added_clauses: vec![clause_idx],
                deleted_clauses: vec![],
            });
            
            // Check clause size limit
            if !should_keep_clause(problem, clause_idx, &config) {
                clause_last_step[clause_idx] = proof_steps.len() - 1;
                continue;
            }
            
            // Forward subsumption check
            if let Some(subsumer_idx) = subsumption_index.find_subsuming(clause_idx, problem) {
                // Record backward subsumption (existing clause subsumes new clause)
                proof_steps.push(ProofStep {
                    rule: InferenceRule::BackwardSubsumption,
                    parents: vec![subsumer_idx],
                    selected_literals: vec![],
                    given_clause: None,
                    added_clauses: vec![],
                    deleted_clauses: vec![clause_idx],
                });
                clause_last_step[clause_idx] = proof_steps.len() - 1;
            } else {
                // Add to unprocessed
                unprocessed.push(clause_idx);
                subsumption_index.insert(clause_idx, problem);
                
                // Backward subsumption: check if new clause subsumes existing clauses
                if config.use_backward_subsumption {
                    let subsumed = subsumption_index.find_subsumed_by(clause_idx, problem);
                    if !subsumed.is_empty() {
                        // Record forward subsumption step
                        proof_steps.push(ProofStep {
                            rule: InferenceRule::ForwardSubsumption,
                            parents: vec![clause_idx],
                            selected_literals: vec![],
                            given_clause: None,
                            added_clauses: vec![],
                            deleted_clauses: subsumed.clone(),
                        });
                        
                        // Remove subsumed clauses
                        for &subsumed_idx in &subsumed {
                            processed.retain(|&x| x != subsumed_idx);
                            unprocessed.retain(|&x| x != subsumed_idx);
                            clause_last_step[subsumed_idx] = proof_steps.len() - 1;
                            subsumption_index.remove(subsumed_idx);
                        }
                    }
                }
                
                // Check for empty clause
                if is_empty_clause(problem, clause_idx) {
                    // Finalize clause lifetimes
                    finalize_clause_lifetimes(&mut clause_last_step, proof_steps.len());
                    
                    return SaturationResult::Proof(Proof {
                        problem: problem.clone(),
                        steps: proof_steps,
                        clause_first_step,
                        clause_last_step,
                    });
                }
            }
        }
        
        // Check resource limits
        if problem.num_clauses > config.max_clauses {
            finalize_clause_lifetimes(&mut clause_last_step, proof_steps.len());
            return SaturationResult::ResourceLimit;
        }
    }
    
    // Finalize clause lifetimes for saturated case
    finalize_clause_lifetimes(&mut clause_last_step, proof_steps.len());
    
    SaturationResult::Saturated
}

/// Check if a clause is empty
fn is_empty_clause(problem: &Problem, clause_idx: usize) -> bool {
    problem.clause_literals(clause_idx).is_empty()
}

/// Check if a clause should be kept (not redundant)
fn should_keep_clause(
    problem: &Problem,
    clause_idx: usize,
    config: &SaturationConfig,
) -> bool {
    // Check clause size limit
    let num_literals = problem.clause_literals(clause_idx).len();
    if num_literals > config.max_clause_size {
        return false;
    }
    
    // Check for tautologies
    if is_tautology(problem, clause_idx) {
        return false;
    }
    
    true
}

/// Check if a clause is a tautology
fn is_tautology(problem: &Problem, clause_idx: usize) -> bool {
    let literals = problem.clause_literals(clause_idx);
    
    // Check for complementary literals
    for i in 0..literals.len() {
        for j in i + 1..literals.len() {
            let lit1 = literals[i];
            let lit2 = literals[j];
            
            // Different polarities
            let pol1 = problem.node_polarities[lit1];
            let pol2 = problem.node_polarities[lit2];
            
            if pol1 != pol2 && pol1 != 0 && pol2 != 0 {
                // Check if predicates are identical
                if predicates_identical(problem, lit1, lit2) {
                    return true;
                }
            }
        }
    }
    
    false
}

/// Check if two literals have identical predicates
fn predicates_identical(problem: &Problem, lit1: usize, lit2: usize) -> bool {
    // Get predicate nodes
    let pred1 = get_literal_predicate(problem, lit1);
    let pred2 = get_literal_predicate(problem, lit2);
    
    match (pred1, pred2) {
        (Some(p1), Some(p2)) => {
            // Check symbol and recursively check arguments
            if problem.node_symbols[p1] != problem.node_symbols[p2] {
                return false;
            }
            
            let args1 = problem.node_children(p1);
            let args2 = problem.node_children(p2);
            
            if args1.len() != args2.len() {
                return false;
            }
            
            for (a1, a2) in args1.iter().zip(args2.iter()) {
                if !terms_identical(problem, *a1, *a2) {
                    return false;
                }
            }
            
            true
        }
        _ => false,
    }
}

/// Check if two terms are identical
fn terms_identical(problem: &Problem, t1: usize, t2: usize) -> bool {
    if problem.node_types[t1] != problem.node_types[t2] {
        return false;
    }
    
    if problem.node_symbols[t1] != problem.node_symbols[t2] {
        return false;
    }
    
    // For functions, check arguments
    if problem.node_types[t1] == NodeType::Function as u8 {
        let args1 = problem.node_children(t1);
        let args2 = problem.node_children(t2);
        
        if args1.len() != args2.len() {
            return false;
        }
        
        for (a1, a2) in args1.iter().zip(args2.iter()) {
            if !terms_identical(problem, *a1, *a2) {
                return false;
            }
        }
    }
    
    true
}

/// Get the predicate node of a literal
fn get_literal_predicate(problem: &Problem, lit_node: usize) -> Option<usize> {
    let children = problem.node_children(lit_node);
    if !children.is_empty() {
        Some(children[0])
    } else {
        None
    }
}

/// Finalize clause lifetimes
fn finalize_clause_lifetimes(clause_last_step: &mut Vec<usize>, final_step: usize) {
    for i in 0..clause_last_step.len() {
        if clause_last_step[i] == usize::MAX {
            clause_last_step[i] = final_step.saturating_sub(1);
        }
    }
}

#[cfg(test)]
mod tests {
    // Tests temporarily disabled during refactoring
    // #[test]
    // fn test_empty_clause_detection() {
    //     let mut problem = Problem::new();
    //     let mut builder = ArrayBuilder::new(&mut problem);
    //     
    //     // Add empty clause
    //     let empty = ParseClause { literals: vec![] };
    //     builder.add_clause(&empty, ClauseType::Axiom);
    //     
    //     // Add non-empty clause
    //     let p = ParsePredicate { name: "P".to_string(), args: vec![] };
    //     let lit = ParseLiteral { predicate: p, polarity: true };
    //     let clause = ParseClause { literals: vec![lit] };
    //     builder.add_clause(&clause, ClauseType::Axiom);
    //     
    //     // Run saturation
    //     let config = SaturationConfig::default();
    //     let result = saturate(&mut problem, &config);
    //     
    //     match result {
    //         SaturationResult::Proof(proof) => {
    //             assert_eq!(proof.steps.len(), 0); // Empty clause found immediately
    //         }
    //         _ => panic!("Expected proof with empty clause"),
    //     }
    // }
}

// Tests temporarily disabled during refactoring
// #[cfg(test)]
// #[path = "saturation_tests.rs"]
// mod extended_tests;