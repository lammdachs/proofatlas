//! Binary resolution inference rule

use crate::core::{Problem, NodeType, ArraySubstitution, Builder};
use crate::saturation::unify_nodes;
use super::common::{InferenceResult, has_selected_literals, get_literal_predicate};

/// Apply binary resolution between two clauses
pub fn resolve_clauses(
    problem: &mut Problem,
    clause1_idx: usize,
    clause2_idx: usize,
) -> Vec<InferenceResult> {
    let mut results = Vec::new();
    
    // Get literals from both clauses
    let lits1 = problem.clause_literals(clause1_idx);
    let lits2 = problem.clause_literals(clause2_idx);
    
    println!("resolve_clauses: clause {} has {} lits, clause {} has {} lits", 
             clause1_idx, lits1.len(), clause2_idx, lits2.len());
    println!("  lits1: {:?}", lits1);
    println!("  lits2: {:?}", lits2);
    
    // Try to resolve on each pair of complementary literals
    for (i, &lit1_node) in lits1.iter().enumerate() {
        // Check if literal is selected (or no selection active)
        if problem.node_selected[lit1_node] || !has_selected_literals(problem, &lits1) {
            for (j, &lit2_node) in lits2.iter().enumerate() {
                // Check if literal is selected (or no selection active)
                if problem.node_selected[lit2_node] || !has_selected_literals(problem, &lits2) {
                    // Check if literals have opposite polarity
                    let pol1 = problem.node_polarities[lit1_node];
                    let pol2 = problem.node_polarities[lit2_node];
                    
                    if pol1 == pol2 || pol1 == 0 || pol2 == 0 {
                        continue;
                    }
                    
                    // Get predicates
                    let pred1_node = get_literal_predicate(problem, lit1_node);
                    let pred2_node = get_literal_predicate(problem, lit2_node);
                    
                    println!("  Checking lit {} (pol {}) vs lit {} (pol {}): pred1={:?}, pred2={:?}",
                             lit1_node, pol1, lit2_node, pol2, pred1_node, pred2_node);
                    
                    // Debug: check literal structure
                    if pred1_node.is_none() && i == 0 && j == 1 {
                        println!("    DEBUG: lit {} children = {:?}", lit1_node, problem.node_children(lit1_node));
                        println!("    DEBUG: lit {} children = {:?}", lit2_node, problem.node_children(lit2_node));
                    }
                    
                    if pred1_node.is_none() || pred2_node.is_none() {
                        continue;
                    }
                    
                    let pred1_node = pred1_node.unwrap();
                    let pred2_node = pred2_node.unwrap();
                    
                    // Try to unify predicates
                    let mut subst = ArraySubstitution::new();
                    if unify_nodes(problem, pred1_node, pred2_node, &mut subst) {
                        // Build resolvent
                        if let Some(new_clause_idx) = build_resolvent_graph(
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


/// Build a resolvent clause using Builder for proper edge management
fn build_resolvent_graph(
    problem: &mut Problem,
    clause1_idx: usize,
    clause2_idx: usize,
    lit1_idx: usize,
    lit2_idx: usize,
    subst: &ArraySubstitution,
) -> Option<usize> {
    let lits1 = problem.clause_literals(clause1_idx);
    let lits2 = problem.clause_literals(clause2_idx);
    
    // Check capacity
    if problem.num_clauses >= problem.max_clauses {
        return None;
    }
    
    // Calculate how many literals we'll need
    let literal_count = (lits1.len() - 1) + (lits2.len() - 1);
    
    // Start building with Builder
    let clause_idx = problem.num_clauses;
    let _start_node = problem.num_nodes;
    
    {
        let mut builder = Builder::new(problem);
        
        // Create clause node
        let clause_node = builder.add_node(NodeType::Clause, "", 0, literal_count as u32).ok()?;
        
        // Copy literals from first clause (except resolved literal)
        for (i, &lit_node) in lits1.iter().enumerate() {
            if i != lit1_idx {
                if let Some(new_lit) = copy_literal_graph(&mut builder, lit_node, clause_node, subst) {
                    builder.add_edge(clause_node, new_lit).ok()?;
                }
            }
        }
        
        // Copy literals from second clause (except resolved literal)
        for (j, &lit_node) in lits2.iter().enumerate() {
            if j != lit2_idx {
                if let Some(new_lit) = copy_literal_graph(&mut builder, lit_node, clause_node, subst) {
                    builder.add_edge(clause_node, new_lit).ok()?;
                }
            }
        }
        
        // Finalize the graph structure
        builder.finalize().ok()?;
    }
    
    // Update problem metadata
    problem.clause_types[clause_idx] = crate::core::ClauseType::Derived as u8;
    problem.clause_boundaries[clause_idx + 1] = problem.num_nodes;
    problem.num_clauses += 1;
    
    Some(clause_idx)
}

/// Copy a literal and its subgraph using Builder
fn copy_literal_graph(
    builder: &mut Builder,
    lit_node: usize,
    _parent_clause: usize,
    subst: &ArraySubstitution,
) -> Option<usize> {
    // Create new literal node
    let polarity = builder.problem.node_polarities[lit_node];
    let new_lit = builder.add_node(NodeType::Literal, "", polarity, 1).ok()?;
    
    // Copy the predicate and its arguments
    if let Some(pred_node) = get_literal_predicate(builder.problem, lit_node) {
        if let Some(new_pred) = copy_predicate_graph(builder, pred_node, new_lit, subst) {
            builder.add_edge(new_lit, new_pred).ok()?;
        }
    }
    
    // Update literal count
    let lit_idx = builder.problem.num_literals;
    if lit_idx + 1 < builder.problem.literal_boundaries.len() {
        builder.problem.literal_boundaries[lit_idx + 1] = builder.problem.num_nodes;
    }
    builder.problem.num_literals += 1;
    
    Some(new_lit)
}

/// Copy a predicate and its arguments using Builder
fn copy_predicate_graph(
    builder: &mut Builder,
    pred_node: usize,
    _parent_lit: usize,
    subst: &ArraySubstitution,
) -> Option<usize> {
    // Get predicate info
    let symbol_id = builder.problem.node_symbols[pred_node];
    let arity = builder.problem.node_arities[pred_node];
    
    // Get symbol as owned string to avoid borrow issues
    let symbol = builder.problem.symbols.get(symbol_id)
        .map(|s| s.to_string())
        .unwrap_or_else(|| String::new());
    
    // Create new predicate node
    let new_pred = builder.add_node(NodeType::Predicate, &symbol, 0, arity).ok()?;
    
    // Copy arguments
    let args = builder.problem.node_children(pred_node);
    for arg in args {
        if let Some(new_arg) = copy_term_graph(builder, arg, new_pred, subst) {
            builder.add_edge(new_pred, new_arg).ok()?;
        }
    }
    
    Some(new_pred)
}

/// Copy a term using Builder
fn copy_term_graph(
    builder: &mut Builder,
    term_node: usize,
    parent_node: usize,
    subst: &ArraySubstitution,
) -> Option<usize> {
    // Check if this is a variable that should be substituted
    if builder.problem.node_types[term_node] == NodeType::Variable as u8 {
        if let Some(replacement) = subst.get(term_node) {
            // Copy the replacement term instead
            return copy_term_graph(builder, replacement, parent_node, subst);
        }
    }
    
    // Get term info
    let node_type = match builder.problem.node_types[term_node] {
        0 => NodeType::Variable,
        1 => NodeType::Constant,
        2 => NodeType::Function,
        _ => return None,
    };
    
    let symbol_id = builder.problem.node_symbols[term_node];
    let arity = builder.problem.node_arities[term_node];
    
    // Get symbol as owned string to avoid borrow issues
    let symbol = builder.problem.symbols.get(symbol_id)
        .map(|s| s.to_string())
        .unwrap_or_else(|| String::new());
    
    // Create new term node
    let new_term = builder.add_node(node_type, &symbol, 0, arity).ok()?;
    
    // For functions, copy arguments
    if node_type == NodeType::Function {
        let children = builder.problem.node_children(term_node);
        for child in children {
            if let Some(new_child) = copy_term_graph(builder, child, new_term, subst) {
                builder.add_edge(new_term, new_child).ok()?;
            }
        }
    }
    
    Some(new_term)
}

