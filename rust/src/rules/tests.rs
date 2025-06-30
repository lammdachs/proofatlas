//! Comprehensive tests for inference rules using parser

#[cfg(test)]
mod tests {
    use crate::rules::*;
    use crate::rules::common::get_literal_predicate;
    use crate::core::*;
    use crate::saturation::{apply_literal_selection, SelectNegative};
    use crate::parsing::tptp_parser::parse_string;
    
    /// Helper to parse a TPTP string into an Problem
    fn parse_problem(tptp_content: &str) -> Problem {
        parse_string(tptp_content).expect("Failed to parse TPTP content")
    }
    
    /// Helper to check if a clause contains a literal with given polarity and predicate
    fn clause_contains_literal(problem: &Problem, clause_idx: usize, polarity: i8, pred_name: &str) -> bool {
        let literals = problem.clause_literals(clause_idx);
        for &lit_idx in &literals {
            if problem.node_polarities[lit_idx] != polarity {
                continue;
            }
            
            // Find predicate node
            let start = problem.edge_row_offsets[lit_idx];
            let end = problem.edge_row_offsets[lit_idx + 1];
            
            for i in start..end {
                let pred_idx = problem.edge_col_indices[i] as usize;
                if problem.node_types[pred_idx] == NodeType::Predicate as u8 {
                    let symbol_id = problem.node_symbols[pred_idx];
                    if let Some(symbol) = problem.symbols.get(symbol_id) {
                        if symbol == pred_name {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
    
    
    #[test]
    fn test_basic_resolution() {
        let mut problem = parse_problem("
            cnf(clause1, axiom, p(X)).
            cnf(clause2, axiom, ~p(a)).
        ");
        
        println!("Problem has {} clauses, {} nodes", problem.num_clauses, problem.num_nodes);
        
        // Debug graph structure
        println!("\nGraph structure:");
        println!("Edge row offsets: {:?}", &problem.edge_row_offsets[0..=problem.num_nodes]);
        println!("Edge col indices: {:?}", &problem.edge_col_indices[0..problem.num_edges]);
        
        for i in 0..problem.num_nodes {
            let node_type = match problem.node_types[i] {
                0 => "Variable",
                1 => "Constant", 
                2 => "Function",
                3 => "Predicate",
                4 => "Literal",
                5 => "Clause",
                _ => "Unknown"
            };
            let symbol = problem.symbols.get(problem.node_symbols[i]).unwrap_or("<none>");
            let start = problem.edge_row_offsets[i];
            let end = problem.edge_row_offsets[i + 1];
            println!("  Node {}: type={} ({}), symbol='{}', polarity={}, edges[{}..{}]={:?}, children={:?}", 
                     i, problem.node_types[i], node_type, symbol, problem.node_polarities[i],
                     start, end, &problem.edge_col_indices[start..end],
                     problem.node_children(i));
        }
        
        let c0_lits = problem.clause_literals(0);
        let c1_lits = problem.clause_literals(1);
        println!("\nClause 0 literals: {:?}", c0_lits);
        println!("Clause 1 literals: {:?}", c1_lits);
        
        // Check polarities and structure
        for &lit in &c0_lits {
            println!("  Lit {}: polarity = {}, type = {}, children = {:?}", 
                     lit, problem.node_polarities[lit], problem.node_types[lit],
                     problem.node_children(lit));
        }
        for &lit in &c1_lits {
            println!("  Lit {}: polarity = {}, type = {}, children = {:?}", 
                     lit, problem.node_polarities[lit], problem.node_types[lit],
                     problem.node_children(lit));
        }
        
        // Apply resolution - the parser creates clauses 0 and 1
        let results = resolve_clauses(&mut problem, 0, 1);
        
        println!("Resolution results: {} results", results.len());
        for (i, result) in results.iter().enumerate() {
            println!("  Result {}: new_clause_idx = {:?}", i, result.new_clause_idx);
        }
        
        assert_eq!(results.len(), 1);
        let result = &results[0];
        assert!(result.new_clause_idx.is_some());
        
        // Check that we got empty clause
        let new_clause_idx = result.new_clause_idx.unwrap();
        let literals = problem.clause_literals(new_clause_idx);
        assert_eq!(literals.len(), 0); // Empty clause
    }
    
    #[test]
    fn test_resolution_with_multiple_literals() {
        let mut problem = parse_problem("
            cnf(clause1, axiom, (p(X) | q(X))).
            cnf(clause2, axiom, (~p(a) | r(a))).
        ");
        
        // Apply resolution
        let results = resolve_clauses(&mut problem, 0, 1);
        
        println!("Resolution produced {} results", results.len());
        assert_eq!(results.len(), 1);
        let result = &results[0];
        assert!(result.new_clause_idx.is_some());
        
        // Check resolvent contains Q(a) ∨ R(a)
        let new_clause_idx = result.new_clause_idx.unwrap();
        let literals = problem.clause_literals(new_clause_idx);
        println!("New clause has {} literals", literals.len());
        for &lit in &literals {
            println!("  Literal {}: polarity={}, children={:?}", 
                     lit, problem.node_polarities[lit], problem.node_children(lit));
            if let Some(pred) = get_literal_predicate(&problem, lit) {
                let symbol = problem.symbols.get(problem.node_symbols[pred]).unwrap_or("?");
                println!("    Predicate: {} (node {})", symbol, pred);
                let args = problem.node_children(pred);
                println!("    Args: {:?}", args);
            } else {
                println!("    No predicate found!");
            }
        }
        assert_eq!(literals.len(), 2);
        assert!(clause_contains_literal(&problem, new_clause_idx, 1, "q"));
        assert!(clause_contains_literal(&problem, new_clause_idx, 1, "r"));
    }
    
    #[test]
    fn test_resolution_with_literal_selection() {
        let mut problem = parse_problem("
            cnf(clause1, axiom, (p(X) | ~q(X))).
            cnf(clause2, axiom, (~p(a) | q(a))).
        ");
        
        // Apply negative literal selection
        let selector = SelectNegative;
        apply_literal_selection(&mut problem, 0, &selector);
        apply_literal_selection(&mut problem, 1, &selector);
        
        // Try resolution - should only resolve on selected literals
        let results = resolve_clauses(&mut problem, 0, 1);
        
        // With SelectNegative:
        // c1: P(X) ∨ ¬Q(X) - selects ¬Q(X) at index 1
        // c2: ¬P(a) ∨ Q(a) - selects ¬P(a) at index 0
        // 
        // Since selected literals must be used, and ¬Q(X) needs Q(a) which is not selected,
        // and ¬P(a) needs P(X) which is not selected, there should be no resolvents
        assert_eq!(results.len(), 0);
    }
    
    #[test]
    fn test_factoring() {
        let mut problem = parse_problem("
            cnf(clause1, axiom, (p(X) | p(a))).
        ");
        
        println!("\nFull graph structure for factoring test:");
        println!("Edge row offsets: {:?}", &problem.edge_row_offsets[0..=problem.num_nodes]);
        println!("Edge col indices: {:?}", &problem.edge_col_indices[0..problem.num_edges]);
        
        for i in 0..problem.num_nodes {
            let node_type = match problem.node_types[i] {
                0 => "Variable",
                1 => "Constant", 
                2 => "Function",
                3 => "Predicate",
                4 => "Literal",
                5 => "Clause",
                _ => "Unknown"
            };
            let symbol = problem.symbols.get(problem.node_symbols[i]).unwrap_or("<none>");
            let start = problem.edge_row_offsets[i];
            let end = problem.edge_row_offsets[i + 1];
            println!("  Node {}: type={} ({}), symbol='{}', edges[{}..{}]={:?}, children={:?}", 
                     i, problem.node_types[i], node_type, symbol, 
                     start, end, &problem.edge_col_indices[start..end],
                     problem.node_children(i));
        }
        
        println!("\nClause 0 literals: {:?}", problem.clause_literals(0));
        for (i, &lit) in problem.clause_literals(0).iter().enumerate() {
            println!("  Literal {}: node {}, polarity {}, children {:?}", 
                     i, lit, problem.node_polarities[lit], problem.node_children(lit));
            if let Some(pred) = get_literal_predicate(&problem, lit) {
                println!("    Predicate: node {}, symbol '{}'", pred, 
                         problem.symbols.get(problem.node_symbols[pred]).unwrap_or("?"));
            } else {
                println!("    No predicate found!");
            }
        }
        
        // Apply factoring
        let results = factor_clause(&mut problem, 0);
        
        println!("Factoring results: {}", results.len());
        assert_eq!(results.len(), 1);
        let result = &results[0];
        assert!(result.new_clause_idx.is_some());
        
        // Check factored clause contains only P(a)
        let new_clause_idx = result.new_clause_idx.unwrap();
        let literals = problem.clause_literals(new_clause_idx);
        assert_eq!(literals.len(), 1);
        assert!(clause_contains_literal(&problem, new_clause_idx, 1, "p"));
    }
    
    #[test]
    fn test_factoring_with_multiple_unifiable_literals() {
        let mut problem = parse_problem("
            cnf(clause1, axiom, (p(X) | p(Y) | p(a))).
        ");
        
        // Apply factoring
        let results = factor_clause(&mut problem, 0);
        
        // Should get multiple factors
        assert!(results.len() >= 3); // At least 3 different factorings possible
    }
    
    #[test]
    fn test_equality_resolution() {
        let mut problem = parse_problem("
            fof(clause1, axiom, X != X).
        ");
        
        // Apply equality resolution
        let results = equality_resolve(&mut problem, 0);
        
        assert_eq!(results.len(), 1);
        
        // Should derive empty clause
        let new_clause_idx = results[0].new_clause_idx.unwrap();
        let literals = problem.clause_literals(new_clause_idx);
        assert_eq!(literals.len(), 0);
    }
    
    #[test]
    fn test_no_resolution_same_polarity() {
        let mut problem = parse_problem("
            cnf(clause1, axiom, p(X)).
            cnf(clause2, axiom, p(a)).
        ");
        
        // Apply resolution
        let results = resolve_clauses(&mut problem, 0, 1);
        
        // Should get no resolvents (same polarity)
        assert_eq!(results.len(), 0);
    }
    
    #[test]
    fn test_no_factoring_different_polarity() {
        let mut problem = parse_problem("
            cnf(clause1, axiom, (p(X) | ~p(a))).
        ");
        
        // Apply factoring
        let results = factor_clause(&mut problem, 0);
        
        // Should get no factors (different polarities)
        assert_eq!(results.len(), 0);
    }
    
    #[test]
    fn test_superposition_basic() {
        let mut problem = parse_problem("
            fof(eq1, axiom, b = a).
            fof(has_b, axiom, p(b)).
        ");
        
        // Apply superposition
        let results = superpose_clauses(&mut problem, 0, 1);
        
        // Should derive P(a)
        assert!(results.len() > 0);
        let new_clause_idx = results[0].new_clause_idx.unwrap();
        assert!(clause_contains_literal(&problem, new_clause_idx, 1, "p"));
    }
    
    #[test]
    fn test_equality_factoring() {
        let mut problem = parse_problem("
            fof(clause1, axiom, (X = a | X = b)).
        ");
        
        // Apply equality factoring
        let results = equality_factor(&mut problem, 0);
        
        assert!(results.len() > 0);
        // Should produce clause with a != b
        let new_clause_idx = results[0].new_clause_idx.unwrap();
        let literals = problem.clause_literals(new_clause_idx);
        assert_eq!(literals.len(), 2); // Should have X = a and a != b
    }
    
    #[test]
    fn test_multiple_resolvents() {
        let mut problem = parse_problem("
            cnf(clause1, axiom, (p(X) | q(Y))).
            cnf(clause2, axiom, (~p(a) | ~q(b))).
        ");
        
        println!("\nClause structure:");
        for i in 0..2 {
            let lits = problem.clause_literals(i);
            println!("Clause {}: {} literals", i, lits.len());
            for (j, &lit) in lits.iter().enumerate() {
                println!("  Literal {}: node={}, polarity={}, children={:?}", 
                         j, lit, problem.node_polarities[lit], problem.node_children(lit));
                if let Some(pred) = get_literal_predicate(&problem, lit) {
                    let symbol = problem.symbols.get(problem.node_symbols[pred]).unwrap_or("?");
                    println!("    Predicate: {} (node {})", symbol, pred);
                } else {
                    println!("    No predicate found!");
                }
            }
        }
        
        // Apply resolution
        println!("\nCalling resolve_clauses...");
        let results = resolve_clauses(&mut problem, 0, 1);
        
        println!("\nGot {} resolvents", results.len());
        for (i, result) in results.iter().enumerate() {
            println!("  Result {}: parents={:?}, literals={:?}", 
                     i, result.parent_clauses, result.selected_literals);
        }
        
        // Should get two resolvents (one for each complementary pair)
        assert_eq!(results.len(), 2);
    }
}