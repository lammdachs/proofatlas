//! Comprehensive tests for array types

#[cfg(test)]
mod tests {
    use super::super::types::*;
    use super::super::symbol_table::SymbolTable;
    
    #[test]
    fn test_array_problem_creation() {
        let problem = ArrayProblem::new();
        
        assert_eq!(problem.num_nodes, 0);
        assert_eq!(problem.num_clauses, 0);
        assert_eq!(problem.num_literals, 0);
        assert_eq!(problem.node_types.len(), 0);
        assert_eq!(problem.edge_row_offsets.len(), 1);
        assert_eq!(problem.edge_row_offsets[0], 0);
    }
    
    #[test]
    fn test_node_type_values() {
        // Ensure enum values are stable for array storage
        assert_eq!(NodeType::Variable as u8, 0);
        assert_eq!(NodeType::Constant as u8, 1);
        assert_eq!(NodeType::Function as u8, 2);
        assert_eq!(NodeType::Predicate as u8, 3);
        assert_eq!(NodeType::Literal as u8, 4);
        assert_eq!(NodeType::Clause as u8, 5);
    }
    
    #[test]
    fn test_edge_type_values() {
        assert_eq!(EdgeType::HasArgument as u8, 0);
        assert_eq!(EdgeType::HasLiteral as u8, 1);
        assert_eq!(EdgeType::HasPredicate as u8, 2);
    }
    
    #[test]
    fn test_clause_node_access() {
        let mut problem = ArrayProblem::new();
        
        // Manually add some clause boundaries
        problem.clause_boundaries = vec![0, 5, 10, 15];
        problem.num_clauses = 3;
        
        assert_eq!(problem.clause_node(0), Some(0));
        assert_eq!(problem.clause_node(1), Some(5));
        assert_eq!(problem.clause_node(2), Some(10));
        assert_eq!(problem.clause_node(3), None); // Out of bounds
    }
    
    #[test]
    fn test_clause_node_range() {
        let mut problem = ArrayProblem::new();
        
        problem.clause_boundaries = vec![0, 5, 10, 15];
        problem.num_clauses = 3;
        
        assert_eq!(problem.clause_node_range(0), Some((0, 5)));
        assert_eq!(problem.clause_node_range(1), Some((5, 10)));
        assert_eq!(problem.clause_node_range(2), Some((10, 15)));
        assert_eq!(problem.clause_node_range(3), None);
    }
    
    #[test]
    fn test_clause_literals() {
        let mut problem = ArrayProblem::new();
        
        // Set up a simple clause structure
        problem.node_types = vec![
            NodeType::Clause,    // 0
            NodeType::Literal,   // 1
            NodeType::Predicate, // 2
            NodeType::Literal,   // 3
            NodeType::Predicate, // 4
            NodeType::Clause,    // 5
            NodeType::Literal,   // 6
            NodeType::Predicate, // 7
        ];
        
        problem.clause_boundaries = vec![0, 5, 8];
        problem.num_clauses = 2;
        problem.num_nodes = 8;
        
        let lits0 = problem.clause_literals(0);
        assert_eq!(lits0, vec![1, 3]);
        
        let lits1 = problem.clause_literals(1);
        assert_eq!(lits1, vec![6]);
    }
    
    #[test]
    fn test_node_children() {
        let mut problem = ArrayProblem::new();
        
        // Set up edge data
        problem.edge_row_offsets = vec![0, 2, 3, 5, 5, 7];
        problem.edge_col_indices = vec![1, 2, 3, 4, 5, 6, 7];
        problem.num_nodes = 5;
        
        assert_eq!(problem.node_children(0), vec![1, 2]);
        assert_eq!(problem.node_children(1), vec![3]);
        assert_eq!(problem.node_children(2), vec![4, 5]);
        assert_eq!(problem.node_children(3), vec![]); // No children
        assert_eq!(problem.node_children(4), vec![6, 7]);
        assert_eq!(problem.node_children(10), vec![]); // Out of bounds
    }
    
    #[test]
    fn test_has_empty_clause() {
        let mut problem = ArrayProblem::new();
        
        // Set up clauses - first is empty, second has literals
        problem.node_types = vec![
            NodeType::Clause,    // 0 (empty)
            NodeType::Clause,    // 1
            NodeType::Literal,   // 2
            NodeType::Predicate, // 3
        ];
        
        problem.clause_boundaries = vec![0, 1, 4];
        problem.num_clauses = 2;
        problem.num_nodes = 4;
        
        assert!(problem.has_empty_clause());
        
        // Test with no empty clauses
        let mut problem2 = ArrayProblem::new();
        problem2.node_types = vec![
            NodeType::Clause,
            NodeType::Literal,
            NodeType::Clause,
            NodeType::Literal,
        ];
        problem2.clause_boundaries = vec![0, 2, 4];
        problem2.num_clauses = 2;
        problem2.num_nodes = 4;
        
        assert!(!problem2.has_empty_clause());
    }
    
    #[test]
    fn test_array_substitution() {
        let mut subst = ArraySubstitution::new();
        
        // Test empty substitution
        assert_eq!(subst.get(0), None);
        assert_eq!(subst.var_indices.len(), 0);
        
        // Add bindings
        subst.bind(1, 10);
        subst.bind(3, 30);
        subst.bind(5, 50);
        
        assert_eq!(subst.get(1), Some(10));
        assert_eq!(subst.get(3), Some(30));
        assert_eq!(subst.get(5), Some(50));
        assert_eq!(subst.get(2), None);
        
        // Test clear
        subst.clear();
        assert_eq!(subst.var_indices.len(), 0);
        assert_eq!(subst.get(1), None);
    }
    
    #[test]
    fn test_array_substitution_overwrite() {
        let mut subst = ArraySubstitution::new();
        
        // Binding same variable multiple times
        subst.bind(1, 10);
        subst.bind(1, 20); // This creates a second binding
        
        // get() returns the first binding
        assert_eq!(subst.get(1), Some(10));
        
        // But we have two entries
        assert_eq!(subst.var_indices.len(), 2);
        assert_eq!(subst.var_indices, vec![1, 1]);
        assert_eq!(subst.term_indices, vec![10, 20]);
    }
    
    #[test]
    fn test_complex_graph_structure() {
        let mut problem = ArrayProblem::new();
        
        // Build a more complex structure:
        // Clause 0: P(f(a)) ∨ Q(X)
        problem.node_types = vec![
            NodeType::Clause,    // 0
            NodeType::Literal,   // 1 - positive
            NodeType::Predicate, // 2 - P
            NodeType::Function,  // 3 - f
            NodeType::Constant,  // 4 - a
            NodeType::Literal,   // 5 - positive
            NodeType::Predicate, // 6 - Q
            NodeType::Variable,  // 7 - X
        ];
        
        problem.node_polarities = vec![0, 1, 0, 0, 0, 1, 0, 0];
        problem.node_arities = vec![2, 1, 1, 1, 0, 1, 1, 0];
        
        // Edge structure
        problem.edge_row_offsets = vec![0, 2, 3, 4, 5, 5, 7, 8, 8];
        problem.edge_col_indices = vec![1, 5, 2, 3, 4, 6, 7];
        problem.edge_types = vec![
            EdgeType::HasLiteral,   // 0 -> 1
            EdgeType::HasLiteral,   // 0 -> 5
            EdgeType::HasPredicate, // 1 -> 2
            EdgeType::HasArgument,  // 2 -> 3
            EdgeType::HasArgument,  // 3 -> 4
            EdgeType::HasPredicate, // 5 -> 6
            EdgeType::HasArgument,  // 6 -> 7
        ];
        
        problem.clause_boundaries = vec![0, 8];
        problem.num_clauses = 1;
        problem.num_nodes = 8;
        problem.num_literals = 2;
        
        // Test navigation
        assert_eq!(problem.clause_literals(0), vec![1, 5]);
        assert_eq!(problem.node_children(0), vec![1, 5]);
        assert_eq!(problem.node_children(2), vec![3]); // P -> f
        assert_eq!(problem.node_children(3), vec![4]); // f -> a
    }
    
    #[test]
    fn test_memory_efficiency() {
        // Test that our array representation is memory efficient
        let mut problem = ArrayProblem::new();
        
        // Add 1000 nodes
        for i in 0..1000 {
            problem.node_types.push(if i % 2 == 0 { NodeType::Variable } else { NodeType::Constant });
            problem.node_symbols.push(i as u32);
            problem.node_polarities.push(0);
            problem.node_arities.push(0);
            problem.edge_row_offsets.push(problem.edge_row_offsets.last().unwrap() + 0);
        }
        
        problem.num_nodes = 1000;
        
        // Calculate approximate memory usage
        let node_memory = 
            problem.node_types.len() * std::mem::size_of::<NodeType>() +
            problem.node_symbols.len() * std::mem::size_of::<u32>() +
            problem.node_polarities.len() * std::mem::size_of::<i8>() +
            problem.node_arities.len() * std::mem::size_of::<u32>();
            
        let edge_memory = 
            problem.edge_row_offsets.len() * std::mem::size_of::<usize>() +
            problem.edge_col_indices.len() * std::mem::size_of::<u32>() +
            problem.edge_types.len() * std::mem::size_of::<EdgeType>();
            
        // Should be roughly 10KB for 1000 nodes (vs much more for object representation)
        assert!(node_memory < 15000, "Node memory usage too high: {}", node_memory);
        assert!(edge_memory < 10000, "Edge memory usage too high: {}", edge_memory);
    }
}