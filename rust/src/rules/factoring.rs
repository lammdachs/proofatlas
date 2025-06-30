//! Factoring inference rule

use crate::core::{Problem, NodeType, ArraySubstitution};
use crate::saturation::unify_nodes;
use super::common::{InferenceResult, has_selected_literals, get_literal_predicate, copy_literal_with_subst};

/// Apply factoring to a clause
pub fn factor_clause(
    problem: &mut Problem,
    clause_idx: usize,
) -> Vec<InferenceResult> {
    let mut results = Vec::new();
    
    // Get literals from clause
    let literals = problem.clause_literals(clause_idx);
    let has_selection = has_selected_literals(problem, &literals);
    
    // Try to factor each pair of literals with same polarity
    for i in 0..literals.len() {
        // Check if first literal is selected (or no selection active)
        if !has_selection || problem.node_selected[literals[i]] {
            for j in i + 1..literals.len() {
                // For factoring, both literals should be selected if selection is active
                if !has_selection || problem.node_selected[literals[j]] {
                    let lit1_node = literals[i];
                    let lit2_node = literals[j];
                    
                    // Check if literals have same polarity
                    let pol1 = problem.node_polarities[lit1_node];
                    let pol2 = problem.node_polarities[lit2_node];
                    
                    if pol1 != pol2 || pol1 == 0 {
                        continue;
                    }
                    
                    // Get predicates
                    let pred1_node = get_literal_predicate(problem, lit1_node);
                    let pred2_node = get_literal_predicate(problem, lit2_node);
                    
                    if pred1_node.is_none() || pred2_node.is_none() {
                        continue;
                    }
                    
                    let pred1_node = pred1_node.unwrap();
                    let pred2_node = pred2_node.unwrap();
                    
                    // Try to unify predicates
                    let mut subst = ArraySubstitution::new();
                    if unify_nodes(problem, pred1_node, pred2_node, &mut subst) {
                        // Build factored clause
                        if let Some(new_clause_idx) = build_factored_clause(
                            problem,
                            clause_idx,
                            i,
                            j,
                            &subst,
                        ) {
                            results.push(InferenceResult {
                                new_clause_idx: Some(new_clause_idx),
                                parent_clauses: vec![clause_idx],
                                applied_rule: "factoring".to_string(),
                                selected_literals: vec![i, j],
                            });
                        }
                    }
                }
            }
        }
    }
    
    results
}

/// Build a factored clause
fn build_factored_clause(
    problem: &mut Problem,
    clause_idx: usize,
    _lit1_idx: usize,
    lit2_idx: usize,
    subst: &ArraySubstitution,
) -> Option<usize> {
    let literals = problem.clause_literals(clause_idx);
    
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
    problem.node_arities[clause_node] = 0; // Will update later
    problem.node_selected[clause_node] = false;
    
    // Update edge offsets
    if clause_node + 1 < problem.edge_row_offsets.len() {
        problem.edge_row_offsets[clause_node + 1] = problem.num_edges;
    }
    
    problem.num_nodes += 1;
    
    // Copy all literals except the second factored literal
    let mut literal_count = 0;
    for (i, &lit_node) in literals.iter().enumerate() {
        if i != lit2_idx {
            if copy_literal_with_subst(problem, lit_node, clause_node, subst).is_err() {
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