//! Comprehensive tests for array-based unification

#[cfg(test)]
mod tests {
    use super::super::unification::*;
    use super::super::types::*;
    use super::super::builder::ArrayBuilder;
    use crate::core::logic::{Term, Predicate, Literal, Clause};
    
    /// Helper to create a test problem with two predicates
    fn create_test_predicates(
        pred1: Predicate,
        pred2: Predicate,
    ) -> (ArrayProblem, usize, usize) {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        let lit1 = Literal::positive(pred1);
        let lit2 = Literal::positive(pred2);
        
        let clause1 = Clause::new(vec![lit1]);
        let clause2 = Clause::new(vec![lit2]);
        
        builder.add_clause(&clause1);
        builder.add_clause(&clause2);
        
        // Get predicate nodes
        let pred1_node = problem.node_children(problem.clause_literals(0)[0])[0];
        let pred2_node = problem.node_children(problem.clause_literals(1)[0])[0];
        
        (problem, pred1_node, pred2_node)
    }
    
    #[test]
    fn test_unify_identical_constants() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        let a = Term::Constant("a".to_string());
        let p = Predicate::new("P".to_string(), vec![a.clone()]);
        let clause = Clause::new(vec![Literal::positive(p)]);
        
        builder.add_clause(&clause);
        builder.add_clause(&clause.clone());
        
        // Get the constant nodes
        let pred1 = problem.node_children(problem.clause_literals(0)[0])[0];
        let pred2 = problem.node_children(problem.clause_literals(1)[0])[0];
        let const1 = problem.node_children(pred1)[0];
        let const2 = problem.node_children(pred2)[0];
        
        let mut subst = ArraySubstitution::new();
        assert!(unify_nodes(&problem, const1, const2, &mut subst));
        assert_eq!(subst.var_indices.len(), 0); // No variables bound
    }
    
    #[test]
    fn test_unify_different_constants() {
        let (problem, pred1, pred2) = create_test_predicates(
            Predicate::new("P".to_string(), vec![Term::Constant("a".to_string())]),
            Predicate::new("P".to_string(), vec![Term::Constant("b".to_string())]),
        );
        
        let const1 = problem.node_children(pred1)[0];
        let const2 = problem.node_children(pred2)[0];
        
        let mut subst = ArraySubstitution::new();
        assert!(!unify_nodes(&problem, const1, const2, &mut subst));
    }
    
    #[test]
    fn test_unify_variable_with_constant() {
        let (problem, pred1, pred2) = create_test_predicates(
            Predicate::new("P".to_string(), vec![Term::Variable("X".to_string())]),
            Predicate::new("P".to_string(), vec![Term::Constant("a".to_string())]),
        );
        
        let var = problem.node_children(pred1)[0];
        let const_node = problem.node_children(pred2)[0];
        
        let mut subst = ArraySubstitution::new();
        assert!(unify_nodes(&problem, var, const_node, &mut subst));
        assert_eq!(subst.get(var), Some(const_node));
    }
    
    #[test]
    fn test_unify_two_variables() {
        let (problem, pred1, pred2) = create_test_predicates(
            Predicate::new("P".to_string(), vec![Term::Variable("X".to_string())]),
            Predicate::new("P".to_string(), vec![Term::Variable("Y".to_string())]),
        );
        
        let var1 = problem.node_children(pred1)[0];
        let var2 = problem.node_children(pred2)[0];
        
        let mut subst = ArraySubstitution::new();
        assert!(unify_nodes(&problem, var1, var2, &mut subst));
        
        // One should be bound to the other
        assert!(subst.get(var1) == Some(var2) || subst.get(var2) == Some(var1));
    }
    
    #[test]
    fn test_unify_function_terms() {
        let a = Term::Constant("a".to_string());
        let b = Term::Constant("b".to_string());
        
        let (problem, pred1, pred2) = create_test_predicates(
            Predicate::new("P".to_string(), vec![
                Term::Function { name: "f".to_string(), args: vec![a.clone()] }
            ]),
            Predicate::new("P".to_string(), vec![
                Term::Function { name: "f".to_string(), args: vec![b] }
            ]),
        );
        
        let func1 = problem.node_children(pred1)[0];
        let func2 = problem.node_children(pred2)[0];
        
        let mut subst = ArraySubstitution::new();
        // Should fail - different arguments
        assert!(!unify_nodes(&problem, func1, func2, &mut subst));
    }
    
    #[test]
    fn test_unify_function_with_variable_arg() {
        let x = Term::Variable("X".to_string());
        let a = Term::Constant("a".to_string());
        
        let (problem, pred1, pred2) = create_test_predicates(
            Predicate::new("P".to_string(), vec![
                Term::Function { name: "f".to_string(), args: vec![x] }
            ]),
            Predicate::new("P".to_string(), vec![
                Term::Function { name: "f".to_string(), args: vec![a] }
            ]),
        );
        
        let func1 = problem.node_children(pred1)[0];
        let func2 = problem.node_children(pred2)[0];
        
        let mut subst = ArraySubstitution::new();
        assert!(unify_nodes(&problem, func1, func2, &mut subst));
        
        // X should be bound to a
        let x_node = problem.node_children(func1)[0];
        let a_node = problem.node_children(func2)[0];
        assert_eq!(subst.get(x_node), Some(a_node));
    }
    
    #[test]
    fn test_unify_different_function_symbols() {
        let a = Term::Constant("a".to_string());
        
        let (problem, pred1, pred2) = create_test_predicates(
            Predicate::new("P".to_string(), vec![
                Term::Function { name: "f".to_string(), args: vec![a.clone()] }
            ]),
            Predicate::new("P".to_string(), vec![
                Term::Function { name: "g".to_string(), args: vec![a] }
            ]),
        );
        
        let func1 = problem.node_children(pred1)[0];
        let func2 = problem.node_children(pred2)[0];
        
        let mut subst = ArraySubstitution::new();
        assert!(!unify_nodes(&problem, func1, func2, &mut subst));
    }
    
    #[test]
    fn test_unify_different_arities() {
        let a = Term::Constant("a".to_string());
        
        let (problem, pred1, pred2) = create_test_predicates(
            Predicate::new("P".to_string(), vec![
                Term::Function { name: "f".to_string(), args: vec![a.clone()] }
            ]),
            Predicate::new("P".to_string(), vec![
                Term::Function { name: "f".to_string(), args: vec![a.clone(), a] }
            ]),
        );
        
        let func1 = problem.node_children(pred1)[0];
        let func2 = problem.node_children(pred2)[0];
        
        let mut subst = ArraySubstitution::new();
        assert!(!unify_nodes(&problem, func1, func2, &mut subst));
    }
    
    #[test]
    fn test_unify_nested_functions() {
        let x = Term::Variable("X".to_string());
        let y = Term::Variable("Y".to_string());
        
        // f(g(X)) with f(g(Y))
        let (problem, pred1, pred2) = create_test_predicates(
            Predicate::new("P".to_string(), vec![
                Term::Function {
                    name: "f".to_string(),
                    args: vec![Term::Function {
                        name: "g".to_string(),
                        args: vec![x],
                    }],
                }
            ]),
            Predicate::new("P".to_string(), vec![
                Term::Function {
                    name: "f".to_string(),
                    args: vec![Term::Function {
                        name: "g".to_string(),
                        args: vec![y],
                    }],
                }
            ]),
        );
        
        let func1 = problem.node_children(pred1)[0];
        let func2 = problem.node_children(pred2)[0];
        
        let mut subst = ArraySubstitution::new();
        assert!(unify_nodes(&problem, func1, func2, &mut subst));
        
        // Should have bound X to Y or vice versa
        assert_eq!(subst.var_indices.len(), 1);
    }
    
    #[test]
    fn test_unify_with_existing_substitution() {
        let (problem, pred1, pred2) = create_test_predicates(
            Predicate::new("P".to_string(), vec![
                Term::Variable("X".to_string()),
                Term::Variable("X".to_string()),
            ]),
            Predicate::new("P".to_string(), vec![
                Term::Constant("a".to_string()),
                Term::Constant("b".to_string()),
            ]),
        );
        
        let x1 = problem.node_children(pred1)[0];
        let x2 = problem.node_children(pred1)[1];
        let a = problem.node_children(pred2)[0];
        let b = problem.node_children(pred2)[1];
        
        let mut subst = ArraySubstitution::new();
        
        // First unify X with a
        assert!(unify_nodes(&problem, x1, a, &mut subst));
        assert_eq!(subst.get(x1), Some(a));
        
        // Now try to unify second X with b - should fail
        // because X is already bound to a
        assert!(!unify_nodes(&problem, x2, b, &mut subst));
    }
    
    #[test]
    fn test_unify_predicates() {
        let x = Term::Variable("X".to_string());
        let y = Term::Variable("Y".to_string());
        let a = Term::Constant("a".to_string());
        
        let (problem, pred1, pred2) = create_test_predicates(
            Predicate::new("P".to_string(), vec![x, a.clone()]),
            Predicate::new("P".to_string(), vec![y, a]),
        );
        
        let mut subst = ArraySubstitution::new();
        assert!(unify_nodes(&problem, pred1, pred2, &mut subst));
        
        // X and Y should be unified
        assert_eq!(subst.var_indices.len(), 1);
    }
    
    #[test]
    fn test_unify_complex_terms() {
        // Test unifying: f(X, g(a, Y)) with f(b, g(a, h(X)))
        let x = Term::Variable("X".to_string());
        let y = Term::Variable("Y".to_string());
        let a = Term::Constant("a".to_string());
        let b = Term::Constant("b".to_string());
        
        let g1 = Term::Function {
            name: "g".to_string(),
            args: vec![a.clone(), y],
        };
        
        let h_x = Term::Function {
            name: "h".to_string(),
            args: vec![x.clone()],
        };
        
        let g2 = Term::Function {
            name: "g".to_string(),
            args: vec![a, h_x],
        };
        
        let (problem, pred1, pred2) = create_test_predicates(
            Predicate::new("P".to_string(), vec![
                Term::Function { name: "f".to_string(), args: vec![x, g1] }
            ]),
            Predicate::new("P".to_string(), vec![
                Term::Function { name: "f".to_string(), args: vec![b, g2] }
            ]),
        );
        
        let func1 = problem.node_children(pred1)[0];
        let func2 = problem.node_children(pred2)[0];
        
        let mut subst = ArraySubstitution::new();
        assert!(unify_nodes(&problem, func1, func2, &mut subst));
        
        // Should have X -> b and Y -> h(b)
        assert_eq!(subst.var_indices.len(), 2);
    }
    
    #[test]
    fn test_unify_type_mismatch() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // Create a predicate and a function node
        let p = Predicate::new("P".to_string(), vec![]);
        let f = Term::Function {
            name: "f".to_string(),
            args: vec![Term::Constant("a".to_string())],
        };
        
        let clause = Clause::new(vec![
            Literal::positive(p),
            Literal::positive(Predicate::new("Q".to_string(), vec![f])),
        ]);
        
        builder.add_clause(&clause);
        
        let lits = problem.clause_literals(0);
        let pred_node = problem.node_children(lits[0])[0];
        let q_node = problem.node_children(lits[1])[0];
        let func_node = problem.node_children(q_node)[0];
        
        let mut subst = ArraySubstitution::new();
        // Predicate and function cannot unify
        assert!(!unify_nodes(&problem, pred_node, func_node, &mut subst));
    }
    
    #[test]
    fn test_unify_large_terms() {
        // Create deeply nested terms to test stack handling
        let mut term1 = Term::Variable("X".to_string());
        let mut term2 = Term::Constant("a".to_string());
        
        // Nest 20 levels deep
        for i in 0..20 {
            term1 = Term::Function {
                name: format!("f{}", i),
                args: vec![term1],
            };
            term2 = Term::Function {
                name: format!("f{}", i),
                args: vec![term2],
            };
        }
        
        let (problem, pred1, pred2) = create_test_predicates(
            Predicate::new("P".to_string(), vec![term1]),
            Predicate::new("P".to_string(), vec![term2]),
        );
        
        let t1 = problem.node_children(pred1)[0];
        let t2 = problem.node_children(pred2)[0];
        
        let mut subst = ArraySubstitution::new();
        assert!(unify_nodes(&problem, t1, t2, &mut subst));
        
        // Should have bound X to a (at the deepest level)
        assert_eq!(subst.var_indices.len(), 1);
    }
}