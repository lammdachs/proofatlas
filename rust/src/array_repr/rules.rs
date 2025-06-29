//! Array-based inference rules

use crate::array_repr::types::{ArrayProblem, NodeType, EdgeType, ArraySubstitution};
use crate::array_repr::unification::unify_nodes;

/// Result of an inference rule application
pub struct InferenceResult {
    pub new_clause_idx: Option<usize>,
    pub parent_clauses: Vec<usize>,
    pub applied_rule: String,
}

/// Apply binary resolution between two clauses
pub fn resolve_clauses(
    problem: &mut ArrayProblem,
    clause1_idx: usize,
    clause2_idx: usize,
) -> Vec<InferenceResult> {
    let mut results = Vec::new();
    
    // Get literals from both clauses
    let lits1 = problem.clause_literals(clause1_idx);
    let lits2 = problem.clause_literals(clause2_idx);
    
    // Try to resolve on each pair of complementary literals
    for (i, &lit1_node) in lits1.iter().enumerate() {
        for (j, &lit2_node) in lits2.iter().enumerate() {
            // Check if literals have opposite polarity
            let pol1 = problem.node_polarities[lit1_node];
            let pol2 = problem.node_polarities[lit2_node];
            
            if pol1 == pol2 || pol1 == 0 || pol2 == 0 {
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
                // Build resolvent
                if let Some(new_clause_idx) = build_resolvent(
                    problem,
                    clause1_idx,
                    clause2_idx,
                    i,
                    j,
                    &subst,
                ) {
                    results.push(InferenceResult {
                        new_clause_idx: Some(new_clause_idx),
                        parent_clauses: vec![clause1_idx, clause2_idx],
                        applied_rule: "resolution".to_string(),
                    });
                }
            }
        }
    }
    
    results
}

/// Apply factoring to a clause
pub fn factor_clause(
    problem: &mut ArrayProblem,
    clause_idx: usize,
) -> Vec<InferenceResult> {
    let mut results = Vec::new();
    
    // Get literals from clause
    let literals = problem.clause_literals(clause_idx);
    
    // Try to factor each pair of literals with same polarity
    for i in 0..literals.len() {
        for j in i + 1..literals.len() {
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
                    });
                }
            }
        }
    }
    
    results
}

/// Get the predicate node of a literal
fn get_literal_predicate(problem: &ArrayProblem, lit_node: usize) -> Option<usize> {
    // Find the predicate child of the literal
    let start = problem.edge_row_offsets[lit_node];
    let end = problem.edge_row_offsets[lit_node + 1];
    
    for i in start..end {
        if problem.edge_types[i] == EdgeType::HasPredicate {
            return Some(problem.edge_col_indices[i] as usize);
        }
    }
    
    None
}

/// Build a resolvent clause from two parent clauses
fn build_resolvent(
    problem: &mut ArrayProblem,
    clause1_idx: usize,
    clause2_idx: usize,
    lit1_idx: usize,
    lit2_idx: usize,
    subst: &ArraySubstitution,
) -> Option<usize> {
    let lits1 = problem.clause_literals(clause1_idx);
    let lits2 = problem.clause_literals(clause2_idx);
    
    // Create new clause node
    let clause_node = problem.num_nodes;
    problem.node_types.push(NodeType::Clause);
    problem.node_symbols.push(0);
    problem.node_polarities.push(0);
    problem.node_arities.push(0); // Will update later
    
    // Update edge offsets
    let last_offset = *problem.edge_row_offsets.last().unwrap();
    problem.edge_row_offsets.push(last_offset);
    
    problem.num_nodes += 1;
    
    // Track clause boundary
    let clause_start = problem.num_nodes - 1;
    
    // Copy literals from first clause (except resolved literal)
    let mut literal_count = 0;
    for (i, &lit_node) in lits1.iter().enumerate() {
        if i != lit1_idx {
            copy_literal_with_subst(problem, lit_node, clause_node, subst);
            literal_count += 1;
        }
    }
    
    // Copy literals from second clause (except resolved literal)
    for (j, &lit_node) in lits2.iter().enumerate() {
        if j != lit2_idx {
            copy_literal_with_subst(problem, lit_node, clause_node, subst);
            literal_count += 1;
        }
    }
    
    // Update clause arity
    problem.node_arities[clause_node] = literal_count;
    
    // Update clause boundaries
    problem.clause_boundaries.push(problem.num_nodes);
    problem.num_clauses += 1;
    
    Some(problem.num_clauses - 1)
}

/// Build a factored clause
fn build_factored_clause(
    problem: &mut ArrayProblem,
    clause_idx: usize,
    lit1_idx: usize,
    lit2_idx: usize,
    subst: &ArraySubstitution,
) -> Option<usize> {
    let literals = problem.clause_literals(clause_idx);
    
    // Create new clause node
    let clause_node = problem.num_nodes;
    problem.node_types.push(NodeType::Clause);
    problem.node_symbols.push(0);
    problem.node_polarities.push(0);
    problem.node_arities.push(0); // Will update later
    
    // Update edge offsets
    let last_offset = *problem.edge_row_offsets.last().unwrap();
    problem.edge_row_offsets.push(last_offset);
    
    problem.num_nodes += 1;
    
    // Copy all literals except the second factored literal
    let mut literal_count = 0;
    for (i, &lit_node) in literals.iter().enumerate() {
        if i != lit2_idx {
            copy_literal_with_subst(problem, lit_node, clause_node, subst);
            literal_count += 1;
        }
    }
    
    // Update clause arity
    problem.node_arities[clause_node] = literal_count;
    
    // Update clause boundaries
    problem.clause_boundaries.push(problem.num_nodes);
    problem.num_clauses += 1;
    
    Some(problem.num_clauses - 1)
}

/// Copy a literal with substitution applied
fn copy_literal_with_subst(
    problem: &mut ArrayProblem,
    lit_node: usize,
    parent_clause: usize,
    subst: &ArraySubstitution,
) {
    // For now, we'll do a simple copy without applying substitution
    // A full implementation would recursively copy and apply substitutions
    
    // Create literal node
    let new_lit_node = problem.num_nodes;
    problem.node_types.push(NodeType::Literal);
    problem.node_symbols.push(problem.node_symbols[lit_node]);
    problem.node_polarities.push(problem.node_polarities[lit_node]);
    problem.node_arities.push(1); // One predicate
    
    // Update edge offsets
    let last_offset = *problem.edge_row_offsets.last().unwrap();
    problem.edge_row_offsets.push(last_offset);
    
    problem.num_nodes += 1;
    
    // Connect clause to literal
    problem.edge_col_indices.push(new_lit_node as u32);
    problem.edge_types.push(EdgeType::HasLiteral);
    problem.edge_row_offsets[parent_clause] += 1;
    
    // Copy predicate (simplified - full implementation would apply substitution)
    if let Some(pred_node) = get_literal_predicate(problem, lit_node) {
        copy_predicate_with_subst(problem, pred_node, new_lit_node, subst);
    }
}

/// Copy a predicate with substitution applied
fn copy_predicate_with_subst(
    problem: &mut ArrayProblem,
    pred_node: usize,
    parent_lit: usize,
    subst: &ArraySubstitution,
) {
    // Create predicate node
    let new_pred_node = problem.num_nodes;
    problem.node_types.push(NodeType::Predicate);
    problem.node_symbols.push(problem.node_symbols[pred_node]);
    problem.node_polarities.push(0);
    problem.node_arities.push(problem.node_arities[pred_node]);
    
    // Update edge offsets
    let last_offset = *problem.edge_row_offsets.last().unwrap();
    problem.edge_row_offsets.push(last_offset);
    
    problem.num_nodes += 1;
    
    // Connect literal to predicate
    problem.edge_col_indices.push(new_pred_node as u32);
    problem.edge_types.push(EdgeType::HasPredicate);
    problem.edge_row_offsets[parent_lit] += 1;
    
    // Copy arguments (simplified - full implementation would apply substitution)
    let args = problem.node_children(pred_node);
    for arg in args {
        copy_term_with_subst(problem, arg, new_pred_node, subst);
    }
}

/// Copy a term with substitution applied
fn copy_term_with_subst(
    problem: &mut ArrayProblem,
    term_node: usize,
    parent_node: usize,
    subst: &ArraySubstitution,
) {
    // Check if this is a variable that should be substituted
    if problem.node_types[term_node] == NodeType::Variable {
        if let Some(replacement) = subst.get(term_node) {
            // Copy the replacement term instead
            copy_term_with_subst(problem, replacement, parent_node, subst);
            return;
        }
    }
    
    // Create term node
    let new_term_node = problem.num_nodes;
    problem.node_types.push(problem.node_types[term_node]);
    problem.node_symbols.push(problem.node_symbols[term_node]);
    problem.node_polarities.push(0);
    problem.node_arities.push(problem.node_arities[term_node]);
    
    // Update edge offsets
    let last_offset = *problem.edge_row_offsets.last().unwrap();
    problem.edge_row_offsets.push(last_offset);
    
    problem.num_nodes += 1;
    
    // Connect parent to term
    problem.edge_col_indices.push(new_term_node as u32);
    problem.edge_types.push(EdgeType::HasArgument);
    problem.edge_row_offsets[parent_node] += 1;
    
    // Copy children for functions
    if problem.node_types[term_node] == NodeType::Function {
        let children = problem.node_children(term_node);
        for child in children {
            copy_term_with_subst(problem, child, new_term_node, subst);
        }
    }
}

#[cfg(test)]
#[path = "rules_tests.rs"]
mod tests;