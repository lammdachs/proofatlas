//! Superposition inference rule

use crate::core::{Problem, NodeType, ArraySubstitution};
use crate::core::ordering::kbo_compare;
use crate::saturation::unify_nodes;
use super::common::{InferenceResult, has_selected_literals, get_literal_predicate, is_equality_predicate, copy_literal_with_subst, copy_term_with_subst};
use std::cmp::Ordering;

/// Apply superposition rule
pub fn superpose_clauses(
    problem: &mut Problem,
    from_clause_idx: usize,  // Clause with equality
    into_clause_idx: usize,  // Clause to superpose into
) -> Vec<InferenceResult> {
    println!("superpose_clauses called: from_clause_idx={}, into_clause_idx={}", from_clause_idx, into_clause_idx);
    let mut results = Vec::new();
    
    // Get literals from both clauses
    let from_lits = problem.clause_literals(from_clause_idx);
    let into_lits = problem.clause_literals(into_clause_idx);
    
    eprintln!("  Superposition: {} literals from from_clause, {} from into_clause", from_lits.len(), into_lits.len());
    eprintln!("  from_lits: {:?}", from_lits);
    eprintln!("  into_lits: {:?}", into_lits);
    
    // Find positive equality literals in from_clause
    for (i, &from_lit) in from_lits.iter().enumerate() {
        eprintln!("  Checking from_lit {} (node {})", i, from_lit);
        
        // Check if literal is selected (or no selection active)
        let from_has_selection = has_selected_literals(problem, &from_lits);
        eprintln!("  from_has_selection: {}", from_has_selection);
        if !from_has_selection || problem.node_selected[from_lit] {
            if problem.node_polarities[from_lit] != 1 {
                eprintln!("  Skipping literal {} - not positive (pol={})", from_lit, problem.node_polarities[from_lit]);
                continue; // Only positive equalities
            }
        
        if let Some(from_pred) = get_literal_predicate(problem, from_lit) {
            if !is_equality_predicate(problem, from_pred) {
                continue;
            }
            
            
            // Get s and t from s = t
            let eq_args = problem.node_children(from_pred);
            if eq_args.len() != 2 {
                continue;
            }
            let s_term = eq_args[0];
            let t_term = eq_args[1];
            
            let s_sym = problem.symbols.get(problem.node_symbols[s_term]).unwrap_or("?");
            let t_sym = problem.symbols.get(problem.node_symbols[t_term]).unwrap_or("?");
            eprintln!("  Found equality: term {}:'{}' = term {}:'{}'", s_term, s_sym, t_term, t_sym);
            
            // Check if s > t using KBO ordering
            // Only proceed if s > t (or try both if they're incomparable)
            let ordering = kbo_compare(problem, s_term, t_term);
            
            // Skip if t > s (wrong orientation)
            if ordering == Ordering::Less {
                continue;
            }
            
            
            // Try superposing into each literal of into_clause
            let into_has_selection = has_selected_literals(problem, &into_lits);
            for (j, &into_lit) in into_lits.iter().enumerate() {
                // Check if literal is selected (or no selection active)
                if !into_has_selection || problem.node_selected[into_lit] {
                    if let Some(into_pred) = get_literal_predicate(problem, into_lit) {
                        eprintln!("    Looking in predicate {} for term {}", into_pred, s_term);
                        if is_equality_predicate(problem, into_pred) {
                            let args = problem.node_children(into_pred);
                            if args.len() == 2 {
                                let left_sym = problem.symbols.get(problem.node_symbols[args[0]]).unwrap_or("?");
                                let right_sym = problem.symbols.get(problem.node_symbols[args[1]]).unwrap_or("?");
                                eprintln!("      Target equality: {}:'{}' = {}:'{}'", args[0], left_sym, args[1], right_sym);
                            }
                        }
                        // Try to find positions where s occurs in into_pred
                        let positions = find_unifiable_positions(problem, into_pred, s_term);
                        eprintln!("Found {} positions for term {} in predicate {}", positions.len(), s_term, into_pred);
                        
                        for position in positions {
                            eprintln!("    Trying position {:?}", position);
                            let mut subst = ArraySubstitution::new();
                            if unify_at_position(problem, into_pred, &position, s_term, &mut subst) {
                                eprintln!("    Unification succeeded!");
                                // Build superposition result
                                if let Some(new_clause_idx) = build_superposition(
                                    problem,
                                    from_clause_idx,
                                    into_clause_idx,
                                    i,
                                    j,
                                    s_term,
                                    t_term,
                                    &position,
                                    &subst,
                                ) {
                                    eprintln!("    Generated new clause: {}", new_clause_idx);
                                    results.push(InferenceResult {
                                        new_clause_idx: Some(new_clause_idx),
                                        parent_clauses: vec![from_clause_idx, into_clause_idx],
                                        applied_rule: "superposition".to_string(),
                                        selected_literals: vec![i, j],
                                    });
                                } else {
                                    eprintln!("    Failed to build superposition");
                                }
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

/// Find positions in a term where another term could potentially unify
fn find_unifiable_positions(problem: &Problem, term: usize, pattern: usize) -> Vec<Vec<usize>> {
    let mut positions = Vec::new();
    find_positions_recursive(problem, term, pattern, Vec::new(), &mut positions);
    positions
}

fn find_positions_recursive(
    problem: &Problem,
    term: usize,
    pattern: usize,
    current_pos: Vec<usize>,
    positions: &mut Vec<Vec<usize>>,
) {
    // Check if current term could unify with pattern
    let mut subst = ArraySubstitution::new();
    if unify_nodes(problem, term, pattern, &mut subst) {
        positions.push(current_pos.clone());
    }
    
    // Recurse into subterms
    if problem.node_types[term] == NodeType::Function as u8 || problem.node_types[term] == NodeType::Predicate as u8 {
        let children = problem.node_children(term);
        for (i, &child) in children.iter().enumerate() {
            let mut child_pos = current_pos.clone();
            child_pos.push(i);
            find_positions_recursive(problem, child, pattern, child_pos, positions);
        }
    }
}

/// Try to unify at a specific position in a term
fn unify_at_position(
    problem: &Problem,
    term: usize,
    position: &[usize],
    pattern: usize,
    subst: &mut ArraySubstitution,
) -> bool {
    let subterm = get_subterm_at_position(problem, term, position);
    if let Some(subterm) = subterm {
        unify_nodes(problem, subterm, pattern, subst)
    } else {
        false
    }
}

/// Get subterm at a specific position
fn get_subterm_at_position(problem: &Problem, term: usize, position: &[usize]) -> Option<usize> {
    if position.is_empty() {
        return Some(term);
    }
    
    let children = problem.node_children(term);
    if position[0] < children.len() {
        get_subterm_at_position(problem, children[position[0]], &position[1..])
    } else {
        None
    }
}

/// Build the result of superposition
fn build_superposition(
    problem: &mut Problem,
    from_clause_idx: usize,
    into_clause_idx: usize,
    from_lit_idx: usize,
    into_lit_idx: usize,
    s_term: usize,
    t_term: usize,
    position: &[usize],
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
    
    let from_lits = problem.clause_literals(from_clause_idx);
    let into_lits = problem.clause_literals(into_clause_idx);
    
    let mut literal_count = 0;
    
    // Copy literals from from_clause (except the equality used)
    for (i, &lit) in from_lits.iter().enumerate() {
        if i != from_lit_idx {
            if copy_literal_with_subst(problem, lit, clause_node, subst).is_err() {
                return None; // Capacity exceeded
            }
            literal_count += 1;
        }
    }
    
    // Copy literals from into_clause, replacing at position
    for (j, &lit) in into_lits.iter().enumerate() {
        if j == into_lit_idx {
            // This is the literal we're superposing into
            // We need to copy it but replace s with t at the given position
            if copy_literal_with_replacement(problem, lit, clause_node, position, s_term, t_term, subst).is_err() {
                return None; // Capacity exceeded
            }
        } else {
            if copy_literal_with_subst(problem, lit, clause_node, subst).is_err() {
                return None; // Capacity exceeded
            }
        }
        literal_count += 1;
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

/// Copy a literal with a replacement at a specific position
fn copy_literal_with_replacement(
    problem: &mut Problem,
    lit_node: usize,
    parent_clause: usize,
    position: &[usize],
    _s_term: usize,
    t_term: usize,
    subst: &ArraySubstitution,
) -> Result<(), crate::core::CapacityError> {
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
    problem.node_symbols[new_lit_node] = 0;
    problem.node_polarities[new_lit_node] = problem.node_polarities[lit_node];
    problem.node_arities[new_lit_node] = 1;
    problem.node_selected[new_lit_node] = false;
    
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
    problem.edge_col_indices[problem.num_edges] = new_lit_node as u32;
    problem.num_edges += 1;
    
    // Update row offsets
    for i in (parent_clause + 1)..=problem.num_nodes {
        if i < problem.edge_row_offsets.len() {
            problem.edge_row_offsets[i] = problem.num_edges;
        }
    }
    
    // Copy predicate with replacement
    if let Some(pred_node) = get_literal_predicate(problem, lit_node) {
        copy_term_with_replacement(problem, pred_node, new_lit_node, position, t_term, subst)?;
    }
    
    problem.num_literals += 1;
    Ok(())
}

/// Copy a term, replacing at a specific position
fn copy_term_with_replacement(
    problem: &mut Problem,
    term_node: usize,
    parent_node: usize,
    position: &[usize],
    replacement: usize,
    subst: &ArraySubstitution,
) -> Result<(), crate::core::CapacityError> {
    if position.is_empty() {
        // This is the position to replace
        copy_term_with_subst(problem, replacement, parent_node, subst)?;
    } else {
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
        
        // Copy children
        let children = problem.node_children(term_node);
        for (i, &child) in children.iter().enumerate() {
            if i == position[0] {
                copy_term_with_replacement(problem, child, new_term_node, &position[1..], replacement, subst)?;
            } else {
                copy_term_with_subst(problem, child, new_term_node, subst)?;
            }
        }
    }
    Ok(())
}