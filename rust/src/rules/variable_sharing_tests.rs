#[cfg(test)]
mod tests {
    use crate::parsing::tptp_parser::parse_string;
    use crate::rules::{resolve_clauses, factor_clause, equality_factor};
    use crate::saturation::unify_nodes;
    use crate::core::ArraySubstitution;
    
    #[test]
    fn test_resolution_different_variables() {
        // Test that P(a,X) from one clause can resolve with ~P(X,b) from another clause
        // The X's are different variables and can be renamed
        let input = r#"
            cnf(c1, axiom, p(a,X)).
            cnf(c2, axiom, ~p(Y,b)).
        "#;
        
        let mut problem = parse_string(input).expect("Failed to parse");
        
        // Apply resolution between clause 0 and clause 1
        let results = resolve_clauses(&mut problem, 0, 1);
        
        // Should be able to resolve: P(a,X) with ~P(Y,b) by X->b, Y->a
        assert!(!results.is_empty(), "P(a,X) should resolve with ~P(Y,b)");
        
        if let Some(new_clause_idx) = results[0].new_clause_idx {
            let literals = problem.clause_literals(new_clause_idx);
            assert_eq!(literals.len(), 0, "Resolution should produce empty clause");
        }
    }
    
    #[test]
    fn test_factoring_same_variable() {
        // Test that P(a,X) | P(X,b) in the same clause cannot be factored
        // because X cannot be both 'a' and 'b'
        let input = r#"
            cnf(c1, axiom, p(a,X) | p(X,b)).
        "#;
        
        let mut problem = parse_string(input).expect("Failed to parse");
        
        // Apply factoring to clause 0
        let results = factor_clause(&mut problem, 0);
        
        // Should not be able to factor because X can't unify with both 'a' and 'b'
        assert!(results.is_empty(), "P(a,X) | P(X,b) should NOT factor when X is the same variable");
    }
    
    #[test]
    fn test_factoring_different_variables() {
        // Test that P(a,X) | P(Y,b) in the same clause CAN be factored
        // because X and Y are different variables
        let input = r#"
            cnf(c1, axiom, p(a,X) | p(Y,b)).
        "#;
        
        let mut problem = parse_string(input).expect("Failed to parse");
        
        // Apply factoring to clause 0
        let results = factor_clause(&mut problem, 0);
        
        // Should be able to factor: P(a,X) with P(Y,b) by X->b, Y->a
        assert!(!results.is_empty(), "P(a,X) | P(Y,b) should factor when X and Y are different variables");
        
        if let Some(new_clause_idx) = results[0].new_clause_idx {
            let literals = problem.clause_literals(new_clause_idx);
            assert_eq!(literals.len(), 1, "Factoring should produce single literal P(a,b)");
        }
    }
    
    #[test]
    fn test_complex_variable_sharing() {
        // Test a more complex case: P(X,f(X)) | P(Y,f(Z))
        // These should factor only if we can make X=Y and X=Z, which means Y=Z
        let input = r#"
            cnf(c1, axiom, p(X,f(X)) | p(Y,f(Y))).
            cnf(c2, axiom, p(X,f(X)) | p(Y,f(Z))).
        "#;
        
        let mut problem = parse_string(input).expect("Failed to parse");
        
        // Test factoring clause 0: P(X,f(X)) | P(Y,f(Y))
        let results0 = factor_clause(&mut problem, 0);
        assert!(!results0.is_empty(), "P(X,f(X)) | P(Y,f(Y)) should factor");
        
        // Test factoring clause 1: P(X,f(X)) | P(Y,f(Z))
        // Let's trace through the unification manually:
        // 1. Unify P(X,f(X)) with P(Y,f(Z))
        // 2. This requires X = Y
        // 3. And f(X) = f(Z)
        // 4. Which requires X = Z
        // 5. But X is already bound to Y, so we need Y = Z
        // 6. Since Y and Z are different unbound variables in the same clause, they can be unified
        // 7. So the factoring SHOULD succeed, producing P(Y,f(Y))
        
        let results1 = factor_clause(&mut problem, 1);
        assert!(!results1.is_empty(), "P(X,f(X)) | P(Y,f(Z)) SHOULD factor because Y and Z can be unified");
    }
    
    #[test]
    fn test_unification_respects_variable_scope() {
        // Direct test of unification with variable scoping
        let input = r#"
            cnf(c1, axiom, p(X,X)).
            cnf(c2, axiom, p(X,Y)).
        "#;
        
        let problem = parse_string(input).expect("Failed to parse");
        
        // Get the predicates
        let pred1 = 2;  // P(X,X) from clause 0
        let pred2 = 7;  // P(X,Y) from clause 1
        
        // Test: P(X,X) from clause 0 should unify with P(X,Y) from clause 1
        // because the X's are from different clauses
        let mut subst = ArraySubstitution::new();
        let result = unify_nodes(&problem, pred1, pred2, &mut subst);
        assert!(result, "P(X,X) from one clause should unify with P(X,Y) from another clause");
    }
    
    #[test]
    fn test_equality_factoring() {
        // Test factoring with equality literals
        let input = r#"
            cnf(c1, axiom, X = a | X = b).
            cnf(c2, axiom, X = a | Y = b).
        "#;
        
        let mut problem = parse_string(input).expect("Failed to parse");
        
        // For equality literals, we need to use equality_factor, not regular factor
        
        // Clause 0: X=a | X=b - Regular factoring should not work
        let _results0_regular = factor_clause(&mut problem, 0);
        // Regular factoring doesn't apply to positive equality
        
        // Try equality factoring on clause 0
        let _results0_eq = equality_factor(&mut problem, 0);
        // Equality factoring also shouldn't work because X can't be both a and b
        
        // Clause 1: X=a | Y=b - Try equality factoring
        let _results1_eq = equality_factor(&mut problem, 1);
        // Note: Equality factoring has specific rules about when it applies
    }
    
    #[test]
    fn test_variable_consistency_in_factoring() {
        // Test that variable bindings are consistent during factoring
        // P(X,X) cannot factor with P(a,b) because X can't be both a and b
        let input = r#"
            cnf(c1, axiom, p(X,X) | p(a,b)).
        "#;
        
        let mut problem = parse_string(input).expect("Failed to parse");
        
        let results = factor_clause(&mut problem, 0);
        assert!(results.is_empty(), "P(X,X) | P(a,b) should NOT factor because X can't be both a and b");
    }
}