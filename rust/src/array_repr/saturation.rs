//! Array-based saturation loop using given-clause algorithm

use crate::array_repr::types::{ArrayProblem, NodeType};
use crate::array_repr::rules::{resolve_clauses, factor_clause};
use std::collections::HashSet;

/// Configuration for saturation
pub struct SaturationConfig {
    pub max_clauses: usize,
    pub max_clause_size: usize,
    pub max_iterations: usize,
}

impl Default for SaturationConfig {
    fn default() -> Self {
        SaturationConfig {
            max_clauses: 100_000,
            max_clause_size: 100,
            max_iterations: 1_000_000,
        }
    }
}

/// Result of saturation
pub struct SaturationResult {
    pub found_empty_clause: bool,
    pub num_clauses_generated: usize,
    pub num_iterations: usize,
    pub empty_clause_idx: Option<usize>,
}

/// Run saturation loop on array problem
pub fn saturate(
    problem: &mut ArrayProblem,
    config: &SaturationConfig,
) -> SaturationResult {
    let mut processed: HashSet<usize> = HashSet::new();
    let mut unprocessed: Vec<usize> = (0..problem.num_clauses).collect();
    
    let mut num_iterations = 0;
    let mut num_clauses_generated = 0;
    
    while !unprocessed.is_empty() && num_iterations < config.max_iterations {
        num_iterations += 1;
        
        // Select given clause (FIFO for now)
        let given_idx = unprocessed.remove(0);
        
        // Check if it's the empty clause
        if is_empty_clause(problem, given_idx) {
            return SaturationResult {
                found_empty_clause: true,
                num_clauses_generated,
                num_iterations,
                empty_clause_idx: Some(given_idx),
            };
        }
        
        // Generate inferences with processed clauses
        let mut new_clauses = Vec::new();
        
        // Resolution with all processed clauses
        for &processed_idx in &processed {
            let resolvents = resolve_clauses(problem, given_idx, processed_idx);
            for result in resolvents {
                if let Some(new_idx) = result.new_clause_idx {
                    new_clauses.push(new_idx);
                    num_clauses_generated += 1;
                }
            }
        }
        
        // Factoring on given clause
        let factors = factor_clause(problem, given_idx);
        for result in factors {
            if let Some(new_idx) = result.new_clause_idx {
                new_clauses.push(new_idx);
                num_clauses_generated += 1;
            }
        }
        
        // Forward simplification (basic redundancy check)
        let mut kept_clauses = Vec::new();
        for new_idx in new_clauses {
            if should_keep_clause(problem, new_idx, &processed, &config) {
                kept_clauses.push(new_idx);
            }
        }
        
        // Add kept clauses to unprocessed
        unprocessed.extend(kept_clauses);
        
        // Move given clause to processed
        processed.insert(given_idx);
        
        // Check resource limits
        if problem.num_clauses > config.max_clauses {
            break;
        }
    }
    
    SaturationResult {
        found_empty_clause: false,
        num_clauses_generated,
        num_iterations,
        empty_clause_idx: None,
    }
}

/// Check if a clause is empty
fn is_empty_clause(problem: &ArrayProblem, clause_idx: usize) -> bool {
    problem.clause_literals(clause_idx).is_empty()
}

/// Check if a clause should be kept (not redundant)
fn should_keep_clause(
    problem: &ArrayProblem,
    clause_idx: usize,
    processed: &HashSet<usize>,
    config: &SaturationConfig,
) -> bool {
    // Check clause size limit
    let num_literals = problem.clause_literals(clause_idx).len();
    if num_literals > config.max_clause_size {
        return false;
    }
    
    // Check for tautologies (simplified check)
    if is_tautology(problem, clause_idx) {
        return false;
    }
    
    // TODO: Add subsumption checking
    
    true
}

/// Check if a clause is a tautology
fn is_tautology(problem: &ArrayProblem, clause_idx: usize) -> bool {
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
fn predicates_identical(problem: &ArrayProblem, lit1: usize, lit2: usize) -> bool {
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
fn terms_identical(problem: &ArrayProblem, t1: usize, t2: usize) -> bool {
    if problem.node_types[t1] != problem.node_types[t2] {
        return false;
    }
    
    if problem.node_symbols[t1] != problem.node_symbols[t2] {
        return false;
    }
    
    // For functions, check arguments
    if problem.node_types[t1] == NodeType::Function {
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
fn get_literal_predicate(problem: &ArrayProblem, lit_node: usize) -> Option<usize> {
    use crate::array_repr::types::EdgeType;
    
    let start = problem.edge_row_offsets[lit_node];
    let end = problem.edge_row_offsets[lit_node + 1];
    
    for i in start..end {
        if problem.edge_types[i] == EdgeType::HasPredicate {
            return Some(problem.edge_col_indices[i] as usize);
        }
    }
    
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array_repr::builder::ArrayBuilder;
    use crate::core::logic::{Clause, Literal, Predicate, Term};
    
    #[test]
    fn test_empty_clause_detection() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // Add empty clause
        let empty = Clause::new(vec![]);
        builder.add_clause(&empty);
        
        // Add non-empty clause
        let p = Predicate::new("P".to_string(), vec![]);
        let lit = Literal::positive(p);
        let clause = Clause::new(vec![lit]);
        builder.add_clause(&clause);
        
        // Run saturation
        let config = SaturationConfig::default();
        let result = saturate(&mut problem, &config);
        
        assert!(result.found_empty_clause);
        assert_eq!(result.empty_clause_idx, Some(0));
    }
    
    #[test]
    fn test_simple_resolution() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // P(a)
        let a = Term::Constant("a".to_string());
        let p_a = Predicate::new("P".to_string(), vec![a]);
        let lit1 = Literal::positive(p_a);
        let clause1 = Clause::new(vec![lit1]);
        builder.add_clause(&clause1);
        
        // ~P(a)
        let a2 = Term::Constant("a".to_string());
        let p_a2 = Predicate::new("P".to_string(), vec![a2]);
        let lit2 = Literal::negative(p_a2);
        let clause2 = Clause::new(vec![lit2]);
        builder.add_clause(&clause2);
        
        // Run saturation
        let config = SaturationConfig::default();
        let result = saturate(&mut problem, &config);
        
        assert!(result.found_empty_clause);
    }
}

#[cfg(test)]
#[path = "saturation_tests.rs"]
mod extended_tests;