//! Comprehensive tests for array builder

#[cfg(test)]
mod tests {
    use super::super::builder::*;
    use super::super::types::*;
    use crate::core::logic::{Term, Literal, Clause, Predicate};
    
    #[test]
    fn test_builder_empty_clause() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // Add empty clause
        let empty = Clause::new(vec![]);
        let idx = builder.add_clause(&empty);
        
        assert_eq!(idx, 0);
        assert_eq!(problem.num_clauses, 1);
        assert_eq!(problem.num_nodes, 1); // Just the clause node
        assert_eq!(problem.node_types[0], NodeType::Clause);
        assert_eq!(problem.clause_boundaries, vec![0, 1]);
    }
    
    #[test]
    fn test_builder_propositional_clause() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // Create P ∨ ¬Q
        let p = Predicate::new("P".to_string(), vec![]);
        let q = Predicate::new("Q".to_string(), vec![]);
        
        let lit1 = Literal::positive(p);
        let lit2 = Literal::negative(q);
        
        let clause = Clause::new(vec![lit1, lit2]);
        builder.add_clause(&clause);
        
        // Verify structure
        assert_eq!(problem.num_clauses, 1);
        assert_eq!(problem.num_literals, 2);
        assert_eq!(problem.num_nodes, 5); // clause + 2 literals + 2 predicates
        
        // Check node types
        assert_eq!(problem.node_types[0], NodeType::Clause);
        assert_eq!(problem.node_types[1], NodeType::Literal);
        assert_eq!(problem.node_types[2], NodeType::Predicate);
        assert_eq!(problem.node_types[3], NodeType::Literal);
        assert_eq!(problem.node_types[4], NodeType::Predicate);
        
        // Check polarities
        assert_eq!(problem.node_polarities[1], 1);  // Positive literal
        assert_eq!(problem.node_polarities[3], -1); // Negative literal
        
        // Check symbols
        assert_eq!(problem.symbols.get(problem.node_symbols[2]), Some("P"));
        assert_eq!(problem.symbols.get(problem.node_symbols[4]), Some("Q"));
    }
    
    #[test]
    fn test_builder_first_order_clause() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // Create P(X, f(a))
        let x = Term::Variable("X".to_string());
        let a = Term::Constant("a".to_string());
        let f_a = Term::Function {
            name: "f".to_string(),
            args: vec![a],
        };
        
        let p = Predicate::new("P".to_string(), vec![x, f_a]);
        let lit = Literal::positive(p);
        let clause = Clause::new(vec![lit]);
        
        builder.add_clause(&clause);
        
        // Verify structure
        assert_eq!(problem.num_clauses, 1);
        assert_eq!(problem.num_literals, 1);
        
        // Count node types
        let vars = problem.node_types.iter().filter(|&&t| t == NodeType::Variable).count();
        let consts = problem.node_types.iter().filter(|&&t| t == NodeType::Constant).count();
        let funcs = problem.node_types.iter().filter(|&&t| t == NodeType::Function).count();
        let preds = problem.node_types.iter().filter(|&&t| t == NodeType::Predicate).count();
        
        assert_eq!(vars, 1);   // X
        assert_eq!(consts, 1); // a
        assert_eq!(funcs, 1);  // f
        assert_eq!(preds, 1);  // P
        
        // Verify symbols
        let symbols_vec: Vec<_> = (0..problem.symbols.len())
            .map(|i| problem.symbols.get(i as u32).unwrap())
            .collect();
        
        assert!(symbols_vec.contains(&"X"));
        assert!(symbols_vec.contains(&"a"));
        assert!(symbols_vec.contains(&"f"));
        assert!(symbols_vec.contains(&"P"));
    }
    
    #[test]
    fn test_builder_multiple_clauses() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // Add three clauses
        let p = Predicate::new("P".to_string(), vec![]);
        let q = Predicate::new("Q".to_string(), vec![]);
        let r = Predicate::new("R".to_string(), vec![]);
        
        // Clause 1: P
        let c1 = Clause::new(vec![Literal::positive(p.clone())]);
        let idx1 = builder.add_clause(&c1);
        
        // Clause 2: ¬P ∨ Q
        let c2 = Clause::new(vec![
            Literal::negative(p.clone()),
            Literal::positive(q.clone()),
        ]);
        let idx2 = builder.add_clause(&c2);
        
        // Clause 3: ¬Q ∨ R
        let c3 = Clause::new(vec![
            Literal::negative(q),
            Literal::positive(r),
        ]);
        let idx3 = builder.add_clause(&c3);
        
        assert_eq!(idx1, 0);
        assert_eq!(idx2, 1);
        assert_eq!(idx3, 2);
        assert_eq!(problem.num_clauses, 3);
        
        // Check clause boundaries
        assert_eq!(problem.clause_boundaries.len(), 4); // n+1 boundaries
        
        // Each clause should have correct literals
        assert_eq!(problem.clause_literals(0).len(), 1);
        assert_eq!(problem.clause_literals(1).len(), 2);
        assert_eq!(problem.clause_literals(2).len(), 2);
    }
    
    #[test]
    fn test_builder_complex_term_structure() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // Create P(f(g(X), a), h(b, Y))
        let x = Term::Variable("X".to_string());
        let y = Term::Variable("Y".to_string());
        let a = Term::Constant("a".to_string());
        let b = Term::Constant("b".to_string());
        
        let g_x = Term::Function {
            name: "g".to_string(),
            args: vec![x],
        };
        
        let f_ga = Term::Function {
            name: "f".to_string(),
            args: vec![g_x, a],
        };
        
        let h_by = Term::Function {
            name: "h".to_string(),
            args: vec![b, y],
        };
        
        let p = Predicate::new("P".to_string(), vec![f_ga, h_by]);
        let lit = Literal::positive(p);
        let clause = Clause::new(vec![lit]);
        
        builder.add_clause(&clause);
        
        // Verify arities
        let p_nodes: Vec<_> = problem.node_types.iter()
            .enumerate()
            .filter(|(_, &t)| t == NodeType::Predicate)
            .map(|(i, _)| i)
            .collect();
        
        assert_eq!(p_nodes.len(), 1);
        assert_eq!(problem.node_arities[p_nodes[0]], 2); // P has 2 args
        
        let f_nodes: Vec<_> = problem.node_types.iter()
            .enumerate()
            .filter(|(_, &t)| t == NodeType::Function)
            .filter(|(i, _)| problem.symbols.get(problem.node_symbols[*i]) == Some("f"))
            .map(|(i, _)| i)
            .collect();
        
        assert_eq!(f_nodes.len(), 1);
        assert_eq!(problem.node_arities[f_nodes[0]], 2); // f has 2 args
    }
    
    #[test]
    fn test_builder_edge_consistency() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // Create a simple clause
        let a = Term::Constant("a".to_string());
        let p = Predicate::new("P".to_string(), vec![a]);
        let lit = Literal::positive(p);
        let clause = Clause::new(vec![lit]);
        
        builder.add_clause(&clause);
        
        // Verify edge consistency
        assert_eq!(problem.edge_row_offsets.len(), problem.num_nodes + 1);
        assert_eq!(problem.edge_col_indices.len(), problem.edge_types.len());
        
        // Verify each edge offset is non-decreasing
        for i in 1..problem.edge_row_offsets.len() {
            assert!(problem.edge_row_offsets[i] >= problem.edge_row_offsets[i-1]);
        }
        
        // Last offset should equal number of edges
        assert_eq!(
            *problem.edge_row_offsets.last().unwrap(),
            problem.edge_col_indices.len()
        );
    }
    
    #[test]
    fn test_builder_shared_terms() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // Create clauses with shared constant
        let a = Term::Constant("a".to_string());
        
        let p = Predicate::new("P".to_string(), vec![a.clone()]);
        let q = Predicate::new("Q".to_string(), vec![a.clone()]);
        
        let c1 = Clause::new(vec![Literal::positive(p)]);
        let c2 = Clause::new(vec![Literal::positive(q)]);
        
        builder.add_clause(&c1);
        builder.add_clause(&c2);
        
        // Count constants - should have 2 (not deduplicated in array repr)
        let const_count = problem.node_types.iter()
            .filter(|&&t| t == NodeType::Constant)
            .count();
        
        assert_eq!(const_count, 2); // Each clause has its own copy
        
        // But symbol table should have only one entry for "a"
        let a_symbol_id = problem.symbols.get_id("a").unwrap();
        let a_occurrences = problem.node_symbols.iter()
            .filter(|&&id| id == a_symbol_id)
            .count();
        
        assert_eq!(a_occurrences, 2);
    }
    
    #[test]
    fn test_builder_equality_predicate() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // Create X = Y
        let x = Term::Variable("X".to_string());
        let y = Term::Variable("Y".to_string());
        
        let eq = Predicate::new("=".to_string(), vec![x, y]);
        let lit = Literal::positive(eq);
        let clause = Clause::new(vec![lit]);
        
        builder.add_clause(&clause);
        
        // Verify equality is treated as regular predicate
        let eq_nodes: Vec<_> = problem.node_types.iter()
            .enumerate()
            .filter(|(_, &t)| t == NodeType::Predicate)
            .filter(|(i, _)| problem.symbols.get(problem.node_symbols[*i]) == Some("="))
            .map(|(i, _)| i)
            .collect();
        
        assert_eq!(eq_nodes.len(), 1);
        assert_eq!(problem.node_arities[eq_nodes[0]], 2);
    }
    
    #[test]
    fn test_builder_large_clause() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // Create clause with many literals
        let mut literals = Vec::new();
        for i in 0..50 {
            let p = Predicate::new(format!("P{}", i), vec![]);
            literals.push(Literal::positive(p));
        }
        
        let clause = Clause::new(literals);
        builder.add_clause(&clause);
        
        assert_eq!(problem.num_literals, 50);
        assert_eq!(problem.clause_literals(0).len(), 50);
        
        // Verify all literals are connected to clause
        let clause_node = 0;
        let children = problem.node_children(clause_node);
        assert_eq!(children.len(), 50);
    }
    
    #[test]
    fn test_builder_deeply_nested_terms() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // Create f(f(f(f(a))))
        let mut term = Term::Constant("a".to_string());
        for _ in 0..4 {
            term = Term::Function {
                name: "f".to_string(),
                args: vec![term],
            };
        }
        
        let p = Predicate::new("P".to_string(), vec![term]);
        let lit = Literal::positive(p);
        let clause = Clause::new(vec![lit]);
        
        builder.add_clause(&clause);
        
        // Count function nodes
        let func_count = problem.node_types.iter()
            .filter(|&&t| t == NodeType::Function)
            .count();
        
        assert_eq!(func_count, 4); // 4 nested f's
        
        // Verify each has arity 1
        for (i, &node_type) in problem.node_types.iter().enumerate() {
            if node_type == NodeType::Function {
                assert_eq!(problem.node_arities[i], 1);
            }
        }
    }
}