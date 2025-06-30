//! Equality resolution inference rule

use crate::core::{Problem, NodeType, ArraySubstitution};
use crate::saturation::unify_nodes;
use super::common::{InferenceResult, has_selected_literals, get_literal_predicate, is_equality_predicate, copy_literal_with_subst};

/// Apply equality resolution rule
pub fn equality_resolve(
    problem: &mut Problem,
    clause_idx: usize,
) -> Vec<InferenceResult> {
    let mut results = Vec::new();
    let literals = problem.clause_literals(clause_idx);
    
    // Look for negative equality literals s ≠ t
    let has_selection = has_selected_literals(problem, &literals);
    for (i, &lit) in literals.iter().enumerate() {
        // Check if literal is selected (or no selection active)
        if !has_selection || problem.node_selected[lit] {
            if problem.node_polarities[lit] != -1 {
                continue; // Only negative equalities
            }
        
        if let Some(pred) = get_literal_predicate(problem, lit) {
            if !is_equality_predicate(problem, pred) {
                continue;
            }
            
            // Get s and t from s ≠ t
            let args = problem.node_children(pred);
            if args.len() != 2 {
                continue;
            }
            
            // Try to unify s and t
            let mut subst = ArraySubstitution::new();
            if unify_nodes(problem, args[0], args[1], &mut subst) {
                // Build resolved clause (removing the inequality)
                if let Some(new_clause_idx) = build_equality_resolved_clause(
                    problem,
                    clause_idx,
                    i,
                    &subst,
                ) {
                    results.push(InferenceResult {
                        new_clause_idx: Some(new_clause_idx),
                        parent_clauses: vec![clause_idx],
                        applied_rule: "equality_resolution".to_string(),
                        selected_literals: vec![i],
                    });
                }
            }
        }
        }
    }
    
    results
}

/// Build clause after equality resolution
fn build_equality_resolved_clause(
    problem: &mut Problem,
    clause_idx: usize,
    removed_lit_idx: usize,
    subst: &ArraySubstitution,
) -> Option<usize> {
    let literals = problem.clause_literals(clause_idx);
    
    // Create new clause even if it will be empty (for deriving contradiction)
    
    // Check capacity
    let clause_node = problem.num_nodes;
    if clause_node >= problem.max_nodes {
        return None; // Can't create new clause
    }
    
    if problem.num_clauses >= problem.max_clauses {
        return None; // Can't create new clause
    }
    
    // Create new clause node in pre-allocated arrays
    problem.node_types[clause_node] = NodeType::Clause as u8;
    problem.node_symbols[clause_node] = 0;
    problem.node_polarities[clause_node] = 0;
    problem.node_arities[clause_node] = 0;
    problem.node_selected[clause_node] = false;
    
    // Update edge offsets
    if clause_node + 1 < problem.edge_row_offsets.len() {
        problem.edge_row_offsets[clause_node + 1] = problem.num_edges;
    }
    
    problem.num_nodes += 1;
    
    // Copy all literals except the resolved one
    let mut literal_count = 0;
    for (i, &lit) in literals.iter().enumerate() {
        if i != removed_lit_idx {
            if copy_literal_with_subst(problem, lit, clause_node, subst).is_err() {
                return None; // Capacity exceeded
            }
            literal_count += 1;
        }
    }
    
    // Update clause arity
    problem.node_arities[clause_node] = literal_count;
    
    // Update clause boundaries and types
    if problem.num_clauses + 1 < problem.clause_boundaries.len() {
        problem.clause_boundaries[problem.num_clauses + 1] = problem.num_nodes;
    }
    problem.clause_types[problem.num_clauses] = crate::core::ClauseType::Derived as u8;
    problem.num_clauses += 1;
    
    Some(problem.num_clauses - 1)
}