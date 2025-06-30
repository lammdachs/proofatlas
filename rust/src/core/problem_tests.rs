//! Tests for Problem and related types

#[cfg(test)]
mod tests {
    use super::super::*;
    
    #[test]
    fn test_array_problem_new() {
        let problem = Problem::new();
        assert_eq!(problem.num_nodes, 0);
        assert_eq!(problem.num_clauses, 0);
        assert_eq!(problem.num_literals, 0);
        assert_eq!(problem.edge_row_offsets.len(), problem.max_nodes + 1);
        assert_eq!(problem.edge_row_offsets[0], 0);
        assert_eq!(problem.clause_boundaries.len(), problem.max_clauses + 1);
        assert_eq!(problem.clause_boundaries[0], 0);
    }
    
    #[test]
    fn test_array_problem_with_capacity() {
        let problem = Problem::with_capacity(100, 10, 200);
        assert_eq!(problem.max_nodes, 100);
        assert_eq!(problem.max_clauses, 10);
        assert_eq!(problem.max_edges, 200);
        assert_eq!(problem.node_types.len(), 100);
        assert_eq!(problem.clause_boundaries.len(), 11); // max_clauses + 1
        assert_eq!(problem.edge_col_indices.len(), 200);
    }
    
    #[test]
    fn test_clause_node() {
        // Use parser to create a simple clause
        let problem = crate::parsing::tptp_parser::parse_string("
            cnf(test, axiom, p(a)).
        ").expect("Failed to parse");
        
        // Test clause_node method
        assert_eq!(problem.clause_node(0), Some(0)); // First node is the clause
        assert_eq!(problem.clause_node(1), None); // Out of bounds
    }
    
    #[test]
    fn test_clause_literals_empty() {
        // Create empty clause using $false
        let problem = crate::parsing::tptp_parser::parse_string("
            cnf(empty, axiom, $false).
        ").expect("Failed to parse");
        
        let literals = problem.clause_literals(0);
        assert_eq!(literals.len(), 0);
    }
    
    #[test]
    fn test_clause_literals_with_literals() {
        // Create clause with two literals
        let problem = crate::parsing::tptp_parser::parse_string("
            cnf(test, axiom, (p(a) | q(b))).
        ").expect("Failed to parse");
        
        let literals = problem.clause_literals(0);
        assert_eq!(literals.len(), 2);
        
        // Verify both literals exist and have the expected polarities
        assert_eq!(problem.node_types[literals[0]], NodeType::Literal as u8);
        assert_eq!(problem.node_types[literals[1]], NodeType::Literal as u8);
        assert_eq!(problem.node_polarities[literals[0]], 1);  // positive
        assert_eq!(problem.node_polarities[literals[1]], 1);  // positive
    }
    
    #[test]
    fn test_has_empty_clause() {
        // Test with non-empty clause
        let problem1 = crate::parsing::tptp_parser::parse_string("
            cnf(test, axiom, p(a)).
        ").expect("Failed to parse");
        assert!(!problem1.has_empty_clause());
        
        // Test with empty clause
        let problem2 = crate::parsing::tptp_parser::parse_string("
            cnf(empty, axiom, $false).
        ").expect("Failed to parse");
        assert!(problem2.has_empty_clause());
    }
    
    #[test]
    fn test_node_children() {
        // Create a predicate with two arguments
        let problem = crate::parsing::tptp_parser::parse_string("
            cnf(test, axiom, p(a, b)).
        ").expect("Failed to parse");
        
        // Get the predicate node
        let clause_lits = problem.clause_literals(0);
        let pred_idx = problem.node_children(clause_lits[0])[0];
        
        let children = problem.node_children(pred_idx);
        assert_eq!(children.len(), 2);
        
        // Get the constants
        let const1_idx = children[0];
        let const2_idx = children[1];
        
        // Verify they are constants
        assert_eq!(problem.node_types[const1_idx], NodeType::Constant as u8);
        assert_eq!(problem.node_types[const2_idx], NodeType::Constant as u8);
        
        // Test leaf node
        let const_children = problem.node_children(const1_idx);
        assert!(const_children.is_empty());
        
        // Test out of bounds
        let oob_children = problem.node_children(1000);
        assert!(oob_children.is_empty());
    }
    
    #[test]
    fn test_array_substitution() {
        let mut subst = ArraySubstitution::new();
        
        // Test empty substitution
        assert_eq!(subst.get(0), None);
        
        // Add binding
        subst.bind(0, 5);
        assert_eq!(subst.get(0), Some(5));
        assert_eq!(subst.get(1), None);
        
        // Add another binding
        subst.bind(3, 7);
        assert_eq!(subst.get(3), Some(7));
        
        // Test clear
        subst.clear();
        assert_eq!(subst.get(0), None);
        assert_eq!(subst.get(3), None);
    }
    
    #[test]
    fn test_csr_array_invariants() {
        // Create a function with two arguments to test CSR
        let problem = crate::parsing::tptp_parser::parse_string("
            cnf(test, axiom, p(f(a, b))).
        ").expect("Failed to parse");
        
        // Verify CSR invariants
        // edge_row_offsets should be monotonic
        for i in 0..problem.num_nodes {
            assert!(problem.edge_row_offsets[i] <= problem.edge_row_offsets[i + 1]);
        }
        
        // edge_row_offsets length matches capacity
        assert_eq!(problem.edge_row_offsets.len(), problem.max_nodes + 1);
        
        // Check edge indices are in bounds
        for i in 0..problem.num_edges {
            assert!((problem.edge_col_indices[i] as usize) < problem.num_nodes);
        }
        
        // Verify specific structure: p(f(a, b))
        // Should have: clause -> literal -> predicate -> function -> two constants
        let clause_lits = problem.clause_literals(0);
        let pred = problem.node_children(clause_lits[0])[0];
        let func = problem.node_children(pred)[0];
        let func_children = problem.node_children(func);
        
        assert_eq!(func_children.len(), 2); // f has two arguments
        assert_eq!(problem.node_types[func], NodeType::Function as u8);
        assert_eq!(problem.node_types[func_children[0]], NodeType::Constant as u8);
        assert_eq!(problem.node_types[func_children[1]], NodeType::Constant as u8);
    }
}