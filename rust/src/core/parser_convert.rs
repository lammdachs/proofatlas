//! Convert parser types directly to array representation

use crate::parsing::parse_types::{ParseProblem, ParseClause, ParseLiteral, ParsePredicate, ParseTerm};
use super::{Problem, NodeType, ClauseType, Builder};
use super::problem::CapacityError;
use std::collections::HashMap;

/// Helper type for parser conversion errors
type ParseResult<T> = Result<T, CapacityError>;

/// Convert a parsed problem directly to array representation
pub fn parse_problem_to_array(parse_problem: ParseProblem) -> Result<Problem, String> {
    // Estimate capacity based on parsed problem
    let (max_nodes, max_edges) = estimate_capacity(&parse_problem);
    let max_clauses = parse_problem.clauses.len() + 10000; // Allow room for derived clauses
    
    let mut array_problem = Problem::with_capacity(max_nodes, max_clauses, max_edges);
    
    {
        let mut builder = Builder::new(&mut array_problem);
        
        // Convert each clause
        for (idx, clause) in parse_problem.clauses.iter().enumerate() {
            let clause_type = if parse_problem.conjecture_indices.contains(&idx) {
                ClauseType::NegatedConjecture
            } else {
                ClauseType::Axiom
            };
            convert_clause_graph(&mut builder, clause, clause_type, idx)
                .map_err(|e| format!("Error converting clause {}: {}", idx, e))?;
        }
        
        // Finalize the graph structure
        builder.finalize()
            .map_err(|e| format!("Error finalizing graph: {}", e))?;
    }
    
    Ok(array_problem)
}

/// Estimate capacity needed for a parsed problem
fn estimate_capacity(problem: &ParseProblem) -> (usize, usize) {
    let mut node_count = 0;
    let mut edge_count = 0;
    
    for clause in &problem.clauses {
        node_count += 1; // Clause node
        edge_count += clause.literals.len(); // Clause->literal edges
        
        for literal in &clause.literals {
            node_count += 1; // Literal node
            edge_count += 1; // Literal->predicate edge
            
            node_count += 1; // Predicate node
            edge_count += literal.predicate.args.len(); // Predicate->arg edges
            
            // Count nodes in arguments
            for arg in &literal.predicate.args {
                let (n, e) = count_term_nodes(arg);
                node_count += n;
                edge_count += e;
            }
        }
    }
    
    // Add significant buffer for saturation (10x for nodes, 20x for edges)
    let max_nodes = (node_count * 10).max(10000);
    let max_edges = (edge_count * 20).max(50000);
    
    (max_nodes, max_edges)
}

/// Count nodes and edges in a term
fn count_term_nodes(term: &ParseTerm) -> (usize, usize) {
    match term {
        ParseTerm::Variable(_) | ParseTerm::Constant(_) => (1, 0),
        ParseTerm::Function { args, .. } => {
            let mut nodes = 1; // Function node
            let mut edges = args.len(); // Function->arg edges
            for arg in args {
                let (n, e) = count_term_nodes(arg);
                nodes += n;
                edges += e;
            }
            (nodes, edges)
        }
    }
}

/// Convert a parsed clause using Builder
fn convert_clause_graph(
    builder: &mut Builder, 
    clause: &ParseClause, 
    clause_type: ClauseType,
    clause_idx: usize,
) -> Result<(), String> {
    // Check clause capacity
    if clause_idx >= builder.problem.max_clauses {
        return Err(format!("Exceeded clause capacity: {} >= {}", clause_idx, builder.problem.max_clauses));
    }
    
    let clause_node = builder.add_node(NodeType::Clause, "", 0, clause.literals.len() as u32)
        .map_err(|e| e.to_string())?;
    
    // Set clause type in pre-allocated array
    builder.problem.clause_types[clause_idx] = clause_type as u8;
    
    // Create a variable map for this clause to ensure consistent variable nodes
    let mut var_map = std::collections::HashMap::new();
    
    // Convert all literals with shared variable context
    for literal in &clause.literals {
        let lit_node = convert_literal_graph(builder, literal, &mut var_map)
            .map_err(|e| format!("Error converting literal: {}", e))?;
        builder.add_edge(clause_node, lit_node)
            .map_err(|e| e.to_string())?;
    }
    
    // Update clause boundaries
    if clause_idx + 1 < builder.problem.clause_boundaries.len() {
        builder.problem.clause_boundaries[clause_idx + 1] = builder.problem.num_nodes;
    }
    
    builder.problem.num_clauses += 1;
    Ok(())
}

/// Convert a literal using Builder
fn convert_literal_graph(
    builder: &mut Builder,
    literal: &ParseLiteral,
    var_map: &mut HashMap<String, usize>,
) -> ParseResult<usize> {
    let polarity = if literal.polarity { 1 } else { -1 };
    let lit_node = builder.add_node(
        NodeType::Literal, 
        "", 
        polarity, 
        1  // Always has one predicate
    )?;
    
    // Update literal boundaries
    let lit_idx = builder.problem.num_literals;
    if lit_idx + 1 < builder.problem.literal_boundaries.len() {
        builder.problem.literal_boundaries[lit_idx + 1] = builder.problem.num_nodes;
    }
    
    // Add predicate with variable tracking
    let pred_node = convert_predicate_graph(builder, &literal.predicate, var_map)?;
    builder.add_edge(lit_node, pred_node)?;
    
    builder.problem.num_literals += 1;
    Ok(lit_node)
}

/// Convert a predicate using Builder
fn convert_predicate_graph(
    builder: &mut Builder,
    predicate: &ParsePredicate,
    var_map: &mut HashMap<String, usize>,
) -> ParseResult<usize> {
    let pred_node = builder.add_node(
        NodeType::Predicate,
        &predicate.name,
        0,
        predicate.args.len() as u32
    )?;
    
    // Add arguments with variable tracking
    for arg in &predicate.args {
        let arg_node = convert_term_graph(builder, arg, var_map)?;
        builder.add_edge(pred_node, arg_node)?;
    }
    
    Ok(pred_node)
}

/// Convert a term using Builder
fn convert_term_graph(
    builder: &mut Builder,
    term: &ParseTerm,
    var_map: &mut HashMap<String, usize>,
) -> ParseResult<usize> {
    match term {
        ParseTerm::Variable(name) => {
            // Check if we've seen this variable before in this clause
            if let Some(&node_id) = var_map.get(name) {
                // Reuse the existing node
                Ok(node_id)
            } else {
                // Create a new node and remember it
                let node_id = builder.add_node(NodeType::Variable, name, 0, 0)?;
                var_map.insert(name.clone(), node_id);
                Ok(node_id)
            }
        }
        ParseTerm::Constant(name) => {
            builder.add_node(NodeType::Constant, name, 0, 0)
        }
        ParseTerm::Function { name, args } => {
            let func_node = builder.add_node(
                NodeType::Function,
                name,
                0,
                args.len() as u32
            )?;
            
            // Add arguments with variable tracking
            for arg in args {
                let arg_node = convert_term_graph(builder, arg, var_map)?;
                builder.add_edge(func_node, arg_node)?;
            }
            
            Ok(func_node)
        }
    }
}
