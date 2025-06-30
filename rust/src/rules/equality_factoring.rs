//! Equality factoring inference rule

use crate::core::{Problem, NodeType, ArraySubstitution};
use crate::saturation::unify_nodes;
use super::common::{InferenceResult, has_selected_literals, get_literal_predicate, is_equality_predicate, copy_literal_with_subst, copy_term_with_subst};

/// Apply equality factoring rule
pub fn equality_factor(
    problem: &mut Problem,
    clause_idx: usize,
) -> Vec<InferenceResult> {
    let mut results = Vec::new();
    let literals = problem.clause_literals(clause_idx);
    
    // Look for positive equality literals
    let has_selection = has_selected_literals(problem, &literals);
    for i in 0..literals.len() {
        // Check if literal is selected (or no selection active)
        if !has_selection || problem.node_selected[literals[i]] {
            if problem.node_polarities[literals[i]] != 1 {
                continue;
            }
        
        if let Some(pred1) = get_literal_predicate(problem, literals[i]) {
            if !is_equality_predicate(problem, pred1) {
                continue;
            }
            
            let args1 = problem.node_children(pred1);
            if args1.len() != 2 {
                continue;
            }
            
            // Look for another positive equality to factor with
            for j in i + 1..literals.len() {
                // Both literals should be selected if selection is active
                if !has_selection || problem.node_selected[literals[j]] {
                    if problem.node_polarities[literals[j]] != 1 {
                        continue;
                    }
                
                if let Some(pred2) = get_literal_predicate(problem, literals[j]) {
                    if !is_equality_predicate(problem, pred2) {
                        continue;
                    }
                    
                    let args2 = problem.node_children(pred2);
                    if args2.len() != 2 {
                        continue;
                    }
                    
                    // Try to unify left-hand sides
                    let mut subst = ArraySubstitution::new();
                    if unify_nodes(problem, args1[0], args2[0], &mut subst) {
                        // Build factored clause with inequality
                        if let Some(new_clause_idx) = build_equality_factored_clause(
                            problem,
                            clause_idx,
                            i,
                            j,
                            &subst,
                        ) {
                            results.push(InferenceResult {
                                new_clause_idx: Some(new_clause_idx),
                                parent_clauses: vec![clause_idx],
                                applied_rule: "equality_factoring".to_string(),
                                selected_literals: vec![i, j],
                            });
                        }
                    }
                }
            }
            }
        }
        }
    }
    
    results
}

/// Build clause after equality factoring
fn build_equality_factored_clause(
    problem: &mut Problem,
    clause_idx: usize,
    eq1_idx: usize,
    eq2_idx: usize,
    subst: &ArraySubstitution,
) -> Option<usize> {
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
    
    let literals = problem.clause_literals(clause_idx);
    let mut literal_count = 0;
    
    // Copy all literals except eq2
    for (i, &lit) in literals.iter().enumerate() {
        if i != eq2_idx {
            if copy_literal_with_subst(problem, lit, clause_node, subst).is_err() {
                return None; // Capacity exceeded
            }
            literal_count += 1;
        }
    }
    
    // Add the new inequality t1 ≠ t2
    if let (Some(pred1), Some(pred2)) = (
        get_literal_predicate(problem, literals[eq1_idx]),
        get_literal_predicate(problem, literals[eq2_idx]),
    ) {
        let args1 = problem.node_children(pred1);
        let args2 = problem.node_children(pred2);
        
        // Check capacity for new literal
        let new_lit_node = problem.num_nodes;
        if new_lit_node >= problem.max_nodes {
            return None;
        }
        
        // Create negative equality literal in pre-allocated arrays
        problem.node_types[new_lit_node] = NodeType::Literal as u8;
        problem.node_symbols[new_lit_node] = 0;
        problem.node_polarities[new_lit_node] = -1; // Negative
        problem.node_arities[new_lit_node] = 1;
        problem.node_selected[new_lit_node] = false;
        
        // Update edge offsets
        if new_lit_node + 1 < problem.edge_row_offsets.len() {
            problem.edge_row_offsets[new_lit_node + 1] = problem.num_edges;
        }
        
        problem.num_nodes += 1;
        
        // Add edge from clause to literal
        if problem.num_edges >= problem.max_edges {
            return None;
        }
        problem.edge_col_indices[problem.num_edges] = new_lit_node as u32;
        problem.num_edges += 1;
        
        // Update row offsets
        for i in (clause_node + 1)..=problem.num_nodes {
            if i < problem.edge_row_offsets.len() {
                problem.edge_row_offsets[i] = problem.num_edges;
            }
        }
        
        // Check capacity for predicate
        let new_pred_node = problem.num_nodes;
        if new_pred_node >= problem.max_nodes {
            return None;
        }
        
        // Create equality predicate t1 = t2 in pre-allocated arrays
        problem.node_types[new_pred_node] = NodeType::Predicate as u8;
        problem.node_symbols[new_pred_node] = problem.symbols.intern("=");
        problem.node_polarities[new_pred_node] = 0;
        problem.node_arities[new_pred_node] = 2;
        problem.node_selected[new_pred_node] = false;
        
        // Update edge offsets
        if new_pred_node + 1 < problem.edge_row_offsets.len() {
            problem.edge_row_offsets[new_pred_node + 1] = problem.num_edges;
        }
        
        problem.num_nodes += 1;
        
        // Add edge from literal to predicate
        if problem.num_edges >= problem.max_edges {
            return None;
        }
        problem.edge_col_indices[problem.num_edges] = new_pred_node as u32;
        problem.num_edges += 1;
        
        // Update row offsets
        for i in (new_lit_node + 1)..=problem.num_nodes {
            if i < problem.edge_row_offsets.len() {
                problem.edge_row_offsets[i] = problem.num_edges;
            }
        }
        
        // Copy right-hand sides as arguments
        if copy_term_with_subst(problem, args1[1], new_pred_node, subst).is_err() {
            return None;
        }
        if copy_term_with_subst(problem, args2[1], new_pred_node, subst).is_err() {
            return None;
        }
        
        literal_count += 1;
        problem.num_literals += 1;
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