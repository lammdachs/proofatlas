//! Common utilities and types for inference rules

use crate::core::{Problem, NodeType, ArraySubstitution};

/// Result of an inference rule application
pub struct InferenceResult {
    pub new_clause_idx: Option<usize>,
    pub parent_clauses: Vec<usize>,
    pub applied_rule: String,
    pub selected_literals: Vec<usize>,
}

/// Check if any literals in the list are selected
pub fn has_selected_literals(problem: &Problem, literals: &[usize]) -> bool {
    literals.iter().any(|&lit| problem.node_selected[lit])
}

/// Get the predicate node of a literal
pub fn get_literal_predicate(problem: &Problem, lit_node: usize) -> Option<usize> {
    // Get the first child of the literal (which should be the predicate)
    let children = problem.node_children(lit_node);
    if !children.is_empty() && problem.node_types[children[0]] == NodeType::Predicate as u8 {
        Some(children[0])
    } else {
        None
    }
}

/// Check if a predicate represents equality
pub fn is_equality_predicate(problem: &Problem, pred_node: usize) -> bool {
    if problem.node_types[pred_node] != NodeType::Predicate as u8 {
        return false;
    }
    
    // Check if symbol is "="
    let symbol_id = problem.node_symbols[pred_node];
    if let Some(symbol) = problem.symbols.get(symbol_id) {
        symbol == "="
    } else {
        false
    }
}

/// Copy a literal with substitution applied
pub fn copy_literal_with_subst(
    problem: &mut Problem,
    lit_node: usize,
    parent_clause: usize,
    subst: &ArraySubstitution,
) -> Result<(), crate::core::CapacityError> {
    // For now, we'll do a simple copy without applying substitution
    // A full implementation would recursively copy and apply substitutions
    
    // Check capacity
    let new_lit_node = problem.num_nodes;
    if new_lit_node >= problem.max_nodes {
        return Err(crate::core::CapacityError {
            resource: "nodes",
            requested: new_lit_node + 1,
            capacity: problem.max_nodes,
        });
    }
    
    // Create literal node in pre-allocated arrays
    problem.node_types[new_lit_node] = NodeType::Literal as u8;
    problem.node_symbols[new_lit_node] = problem.node_symbols[lit_node];
    problem.node_polarities[new_lit_node] = problem.node_polarities[lit_node];
    problem.node_arities[new_lit_node] = 1; // One predicate
    problem.node_selected[new_lit_node] = false; // Default to not selected
    
    // Update edge offsets
    if new_lit_node + 1 < problem.edge_row_offsets.len() {
        problem.edge_row_offsets[new_lit_node + 1] = problem.num_edges;
    }
    
    problem.num_nodes += 1;
    
    // Add edge from clause to literal
    if problem.num_edges >= problem.max_edges {
        return Err(crate::core::CapacityError {
            resource: "edges",
            requested: problem.num_edges + 1,
            capacity: problem.max_edges,
        });
    }
    
    // Add edge in CSR format
    problem.edge_col_indices[problem.num_edges] = new_lit_node as u32;
    problem.num_edges += 1;
    
    // Update row offsets
    for i in (parent_clause + 1)..=problem.num_nodes {
        if i < problem.edge_row_offsets.len() {
            problem.edge_row_offsets[i] = problem.num_edges;
        }
    }
    
    // Copy predicate (simplified - full implementation would apply substitution)
    if let Some(pred_node) = get_literal_predicate(problem, lit_node) {
        copy_predicate_with_subst(problem, pred_node, new_lit_node, subst)?;
    }
    
    Ok(())
}

/// Copy a predicate with substitution applied
pub fn copy_predicate_with_subst(
    problem: &mut Problem,
    pred_node: usize,
    parent_lit: usize,
    subst: &ArraySubstitution,
) -> Result<(), crate::core::CapacityError> {
    // Check capacity
    let new_pred_node = problem.num_nodes;
    if new_pred_node >= problem.max_nodes {
        return Err(crate::core::CapacityError {
            resource: "nodes",
            requested: new_pred_node + 1,
            capacity: problem.max_nodes,
        });
    }
    
    // Create predicate node in pre-allocated arrays
    problem.node_types[new_pred_node] = NodeType::Predicate as u8;
    problem.node_symbols[new_pred_node] = problem.node_symbols[pred_node];
    problem.node_polarities[new_pred_node] = 0;
    problem.node_arities[new_pred_node] = problem.node_arities[pred_node];
    problem.node_selected[new_pred_node] = false;
    
    // Update edge offsets
    if new_pred_node + 1 < problem.edge_row_offsets.len() {
        problem.edge_row_offsets[new_pred_node + 1] = problem.num_edges;
    }
    
    problem.num_nodes += 1;
    
    // Add edge from literal to predicate
    if problem.num_edges >= problem.max_edges {
        return Err(crate::core::CapacityError {
            resource: "edges",
            requested: problem.num_edges + 1,
            capacity: problem.max_edges,
        });
    }
    
    problem.edge_col_indices[problem.num_edges] = new_pred_node as u32;
    problem.num_edges += 1;
    
    // Update row offsets
    for i in (parent_lit + 1)..=problem.num_nodes {
        if i < problem.edge_row_offsets.len() {
            problem.edge_row_offsets[i] = problem.num_edges;
        }
    }
    
    // Copy arguments (simplified - full implementation would apply substitution)
    let args = problem.node_children(pred_node);
    for arg in args {
        copy_term_with_subst(problem, arg, new_pred_node, subst)?;
    }
    
    Ok(())
}

/// Copy a term with substitution applied
pub fn copy_term_with_subst(
    problem: &mut Problem,
    term_node: usize,
    parent_node: usize,
    subst: &ArraySubstitution,
) -> Result<(), crate::core::CapacityError> {
    // Check if this is a variable that should be substituted
    if problem.node_types[term_node] == NodeType::Variable as u8 {
        if let Some(replacement) = subst.get(term_node) {
            // Copy the replacement term instead
            return copy_term_with_subst(problem, replacement, parent_node, subst);
        }
    }
    
    // Check capacity
    let new_term_node = problem.num_nodes;
    if new_term_node >= problem.max_nodes {
        return Err(crate::core::CapacityError {
            resource: "nodes",
            requested: new_term_node + 1,
            capacity: problem.max_nodes,
        });
    }
    
    // Create term node in pre-allocated arrays
    problem.node_types[new_term_node] = problem.node_types[term_node];
    problem.node_symbols[new_term_node] = problem.node_symbols[term_node];
    problem.node_polarities[new_term_node] = 0;
    problem.node_arities[new_term_node] = problem.node_arities[term_node];
    problem.node_selected[new_term_node] = false;
    
    // Update edge offsets
    if new_term_node + 1 < problem.edge_row_offsets.len() {
        problem.edge_row_offsets[new_term_node + 1] = problem.num_edges;
    }
    
    problem.num_nodes += 1;
    
    // Add edge from parent to term
    if problem.num_edges >= problem.max_edges {
        return Err(crate::core::CapacityError {
            resource: "edges",
            requested: problem.num_edges + 1,
            capacity: problem.max_edges,
        });
    }
    
    problem.edge_col_indices[problem.num_edges] = new_term_node as u32;
    problem.num_edges += 1;
    
    // Update row offsets
    for i in (parent_node + 1)..=problem.num_nodes {
        if i < problem.edge_row_offsets.len() {
            problem.edge_row_offsets[i] = problem.num_edges;
        }
    }
    
    // Copy children for functions
    if problem.node_types[term_node] == NodeType::Function as u8 {
        let children = problem.node_children(term_node);
        for child in children {
            copy_term_with_subst(problem, child, new_term_node, subst)?;
        }
    }
    
    Ok(())
}