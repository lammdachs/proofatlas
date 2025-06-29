//! Comprehensive tests for array-based inference rules

#[cfg(test)]
mod tests {
    use super::super::rules::*;
    use super::super::types::*;
    use super::super::builder::ArrayBuilder;
    use crate::core::logic::{Term, Predicate, Literal, Clause};
    
    #[test]
    fn test_simple_propositional_resolution() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // Create P and ¬P
        let p = Predicate::new("P".to_string(), vec![]);
        
        let clause1 = Clause::new(vec![Literal::positive(p.clone())]);
        let clause2 = Clause::new(vec![Literal::negative(p)]);
        
        builder.add_clause(&clause1);
        builder.add_clause(&clause2);
        
        // Resolve
        let results = resolve_clauses(&mut problem, 0, 1);
        
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].parent_clauses, vec![0, 1]);
        assert_eq!(results[0].applied_rule, "resolution");
        
        // Should produce empty clause
        let new_idx = results[0].new_clause_idx.unwrap();
        assert!(problem.clause_literals(new_idx).is_empty());
    }
    
    #[test]
    fn test_resolution_with_multiple_literals() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // Create P ∨ Q and ¬P ∨ R
        let p = Predicate::new("P".to_string(), vec![]);
        let q = Predicate::new("Q".to_string(), vec![]);
        let r = Predicate::new("R".to_string(), vec![]);
        
        let clause1 = Clause::new(vec![
            Literal::positive(p.clone()),
            Literal::positive(q.clone()),
        ]);
        let clause2 = Clause::new(vec![
            Literal::negative(p),
            Literal::positive(r.clone()),
        ]);
        
        builder.add_clause(&clause1);
        builder.add_clause(&clause2);
        
        // Resolve
        let results = resolve_clauses(&mut problem, 0, 1);
        
        assert_eq!(results.len(), 1);
        
        // Should produce Q ∨ R
        let new_idx = results[0].new_clause_idx.unwrap();
        let new_literals = problem.clause_literals(new_idx);
        assert_eq!(new_literals.len(), 2);
    }
    
    #[test]
    fn test_resolution_with_variables() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // Create P(X) and ¬P(a)
        let x = Term::Variable("X".to_string());
        let a = Term::Constant("a".to_string());
        
        let p_x = Predicate::new("P".to_string(), vec![x]);
        let p_a = Predicate::new("P".to_string(), vec![a]);
        
        let clause1 = Clause::new(vec![Literal::positive(p_x)]);
        let clause2 = Clause::new(vec![Literal::negative(p_a)]);
        
        builder.add_clause(&clause1);
        builder.add_clause(&clause2);
        
        // Resolve
        let results = resolve_clauses(&mut problem, 0, 1);
        
        assert_eq!(results.len(), 1);
        
        // Should produce empty clause (after substitution X/a)
        let new_idx = results[0].new_clause_idx.unwrap();
        assert!(problem.clause_literals(new_idx).is_empty());
    }
    
    #[test]
    fn test_resolution_multiple_unifiable_pairs() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // Create P(X) ∨ Q(X) and ¬P(a) ∨ ¬Q(b)
        let x = Term::Variable("X".to_string());
        let a = Term::Constant("a".to_string());
        let b = Term::Constant("b".to_string());
        
        let clause1 = Clause::new(vec![
            Literal::positive(Predicate::new("P".to_string(), vec![x.clone()])),
            Literal::positive(Predicate::new("Q".to_string(), vec![x])),
        ]);
        
        let clause2 = Clause::new(vec![
            Literal::negative(Predicate::new("P".to_string(), vec![a])),
            Literal::negative(Predicate::new("Q".to_string(), vec![b])),
        ]);
        
        builder.add_clause(&clause1);
        builder.add_clause(&clause2);
        
        // Resolve - should get two different resolvents
        let results = resolve_clauses(&mut problem, 0, 1);
        
        assert_eq!(results.len(), 2);
    }
    
    #[test]
    fn test_resolution_no_unifiable_literals() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // Create P(a) and Q(b) - no complementary literals
        let a = Term::Constant("a".to_string());
        let b = Term::Constant("b".to_string());
        
        let clause1 = Clause::new(vec![
            Literal::positive(Predicate::new("P".to_string(), vec![a])),
        ]);
        
        let clause2 = Clause::new(vec![
            Literal::positive(Predicate::new("Q".to_string(), vec![b])),
        ]);
        
        builder.add_clause(&clause1);
        builder.add_clause(&clause2);
        
        // Resolve - should get no results
        let results = resolve_clauses(&mut problem, 0, 1);
        
        assert_eq!(results.len(), 0);
    }
    
    #[test]
    fn test_simple_factoring() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // Create P(X) ∨ P(a)
        let x = Term::Variable("X".to_string());
        let a = Term::Constant("a".to_string());
        
        let clause = Clause::new(vec![
            Literal::positive(Predicate::new("P".to_string(), vec![x])),
            Literal::positive(Predicate::new("P".to_string(), vec![a])),
        ]);
        
        builder.add_clause(&clause);
        
        // Factor
        let results = factor_clause(&mut problem, 0);
        
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].parent_clauses, vec![0]);
        assert_eq!(results[0].applied_rule, "factoring");
        
        // Should produce P(a) (after substitution X/a)
        let new_idx = results[0].new_clause_idx.unwrap();
        let new_literals = problem.clause_literals(new_idx);
        assert_eq!(new_literals.len(), 1);
    }
    
    #[test]
    fn test_factoring_multiple_pairs() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // Create P(X) ∨ P(Y) ∨ P(a)
        let x = Term::Variable("X".to_string());
        let y = Term::Variable("Y".to_string());
        let a = Term::Constant("a".to_string());
        
        let clause = Clause::new(vec![
            Literal::positive(Predicate::new("P".to_string(), vec![x])),
            Literal::positive(Predicate::new("P".to_string(), vec![y])),
            Literal::positive(Predicate::new("P".to_string(), vec![a])),
        ]);
        
        builder.add_clause(&clause);
        
        // Factor
        let results = factor_clause(&mut problem, 0);
        
        // Should get multiple factored clauses
        assert!(results.len() >= 3); // At least 3 pairs can be factored
    }
    
    #[test]
    fn test_factoring_negative_literals() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // Create ¬P(X) ∨ ¬P(a) ∨ Q(b)
        let x = Term::Variable("X".to_string());
        let a = Term::Constant("a".to_string());
        let b = Term::Constant("b".to_string());
        
        let clause = Clause::new(vec![
            Literal::negative(Predicate::new("P".to_string(), vec![x])),
            Literal::negative(Predicate::new("P".to_string(), vec![a])),
            Literal::positive(Predicate::new("Q".to_string(), vec![b])),
        ]);
        
        builder.add_clause(&clause);
        
        // Factor
        let results = factor_clause(&mut problem, 0);
        
        assert_eq!(results.len(), 1);
        
        // Should produce ¬P(a) ∨ Q(b)
        let new_idx = results[0].new_clause_idx.unwrap();
        let new_literals = problem.clause_literals(new_idx);
        assert_eq!(new_literals.len(), 2);
    }
    
    #[test]
    fn test_factoring_no_unifiable_pairs() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // Create P(a) ∨ Q(b) ∨ R(c) - no factorable pairs
        let a = Term::Constant("a".to_string());
        let b = Term::Constant("b".to_string());
        let c = Term::Constant("c".to_string());
        
        let clause = Clause::new(vec![
            Literal::positive(Predicate::new("P".to_string(), vec![a])),
            Literal::positive(Predicate::new("Q".to_string(), vec![b])),
            Literal::positive(Predicate::new("R".to_string(), vec![c])),
        ]);
        
        builder.add_clause(&clause);
        
        // Factor
        let results = factor_clause(&mut problem, 0);
        
        assert_eq!(results.len(), 0);
    }
    
    #[test]
    fn test_complex_resolution() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // Create P(f(X), Y) ∨ Q(Y) and ¬P(f(a), g(Z)) ∨ R(Z)
        let x = Term::Variable("X".to_string());
        let y = Term::Variable("Y".to_string());
        let z = Term::Variable("Z".to_string());
        let a = Term::Constant("a".to_string());
        
        let f_x = Term::Function {
            name: "f".to_string(),
            args: vec![x],
        };
        let f_a = Term::Function {
            name: "f".to_string(),
            args: vec![a],
        };
        let g_z = Term::Function {
            name: "g".to_string(),
            args: vec![z.clone()],
        };
        
        let clause1 = Clause::new(vec![
            Literal::positive(Predicate::new("P".to_string(), vec![f_x, y.clone()])),
            Literal::positive(Predicate::new("Q".to_string(), vec![y])),
        ]);
        
        let clause2 = Clause::new(vec![
            Literal::negative(Predicate::new("P".to_string(), vec![f_a, g_z])),
            Literal::positive(Predicate::new("R".to_string(), vec![z])),
        ]);
        
        builder.add_clause(&clause1);
        builder.add_clause(&clause2);
        
        // Resolve
        let results = resolve_clauses(&mut problem, 0, 1);
        
        assert_eq!(results.len(), 1);
        
        // Should produce Q(g(Z)) ∨ R(Z) after substitution X/a, Y/g(Z)
        let new_idx = results[0].new_clause_idx.unwrap();
        let new_literals = problem.clause_literals(new_idx);
        assert_eq!(new_literals.len(), 2);
    }
    
    #[test]
    fn test_resolution_preserves_node_structure() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // Create structured clauses
        let x = Term::Variable("X".to_string());
        let f_x = Term::Function {
            name: "f".to_string(),
            args: vec![x],
        };
        
        let clause1 = Clause::new(vec![
            Literal::positive(Predicate::new("P".to_string(), vec![f_x])),
        ]);
        
        let a = Term::Constant("a".to_string());
        let f_a = Term::Function {
            name: "f".to_string(),
            args: vec![a],
        };
        
        let clause2 = Clause::new(vec![
            Literal::negative(Predicate::new("P".to_string(), vec![f_a])),
            Literal::positive(Predicate::new("Q".to_string(), vec![])),
        ]);
        
        let initial_nodes = problem.num_nodes;
        builder.add_clause(&clause1);
        builder.add_clause(&clause2);
        
        // Resolve
        let results = resolve_clauses(&mut problem, 0, 1);
        
        // Verify node count increased appropriately
        assert!(problem.num_nodes > initial_nodes);
        
        // Verify edge consistency
        assert_eq!(
            problem.edge_row_offsets.len(),
            problem.num_nodes + 1
        );
    }
    
    #[test]
    fn test_multiple_resolutions_consistency() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // Create a small set of clauses
        let p = Predicate::new("P".to_string(), vec![]);
        let q = Predicate::new("Q".to_string(), vec![]);
        let r = Predicate::new("R".to_string(), vec![]);
        
        // P ∨ Q
        builder.add_clause(&Clause::new(vec![
            Literal::positive(p.clone()),
            Literal::positive(q.clone()),
        ]));
        
        // ¬P ∨ R
        builder.add_clause(&Clause::new(vec![
            Literal::negative(p),
            Literal::positive(r.clone()),
        ]));
        
        // ¬Q
        builder.add_clause(&Clause::new(vec![
            Literal::negative(q),
        ]));
        
        // ¬R
        builder.add_clause(&Clause::new(vec![
            Literal::negative(r),
        ]));
        
        let initial_clauses = problem.num_clauses;
        
        // Perform multiple resolutions
        let res1 = resolve_clauses(&mut problem, 0, 1); // Should give Q ∨ R
        let res2 = resolve_clauses(&mut problem, 0, 2); // Should give P
        
        assert!(!res1.is_empty());
        assert!(!res2.is_empty());
        
        // Verify clause count increased correctly
        assert!(problem.num_clauses > initial_clauses);
        
        // Verify all new clauses are properly bounded
        for i in initial_clauses..problem.num_clauses {
            assert!(problem.clause_node_range(i).is_some());
        }
    }
}