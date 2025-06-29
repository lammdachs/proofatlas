//! Comprehensive tests for array-based saturation

#[cfg(test)]
mod tests {
    use super::super::saturation::*;
    use super::super::types::*;
    use super::super::builder::ArrayBuilder;
    use crate::core::logic::{Clause, Literal, Predicate, Term};
    
    #[test]
    fn test_saturate_empty_problem() {
        let mut problem = ArrayProblem::new();
        let config = SaturationConfig::default();
        
        let result = saturate(&mut problem, &config);
        
        assert!(!result.found_empty_clause);
        assert_eq!(result.num_clauses_generated, 0);
        assert_eq!(result.num_iterations, 0);
    }
    
    #[test]
    fn test_saturate_single_clause() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        let p = Predicate::new("P".to_string(), vec![]);
        let clause = Clause::new(vec![Literal::positive(p)]);
        builder.add_clause(&clause);
        
        let config = SaturationConfig::default();
        let result = saturate(&mut problem, &config);
        
        assert!(!result.found_empty_clause);
        assert_eq!(result.num_clauses_generated, 0); // No inferences possible
        assert_eq!(result.num_iterations, 1);
    }
    
    #[test]
    fn test_saturate_immediate_contradiction() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // Add empty clause
        builder.add_clause(&Clause::new(vec![]));
        
        let config = SaturationConfig::default();
        let result = saturate(&mut problem, &config);
        
        assert!(result.found_empty_clause);
        assert_eq!(result.empty_clause_idx, Some(0));
        assert_eq!(result.num_iterations, 1);
    }
    
    #[test]
    fn test_saturate_simple_propositional() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        let p = Predicate::new("P".to_string(), vec![]);
        
        // P
        builder.add_clause(&Clause::new(vec![Literal::positive(p.clone())]));
        // ¬P
        builder.add_clause(&Clause::new(vec![Literal::negative(p)]));
        
        let config = SaturationConfig::default();
        let result = saturate(&mut problem, &config);
        
        assert!(result.found_empty_clause);
        assert_eq!(result.num_clauses_generated, 1); // Empty clause
    }
    
    #[test]
    fn test_saturate_propositional_chain() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        let p = Predicate::new("P".to_string(), vec![]);
        let q = Predicate::new("Q".to_string(), vec![]);
        let r = Predicate::new("R".to_string(), vec![]);
        
        // P
        builder.add_clause(&Clause::new(vec![Literal::positive(p.clone())]));
        // ¬P ∨ Q
        builder.add_clause(&Clause::new(vec![
            Literal::negative(p),
            Literal::positive(q.clone()),
        ]));
        // ¬Q ∨ R
        builder.add_clause(&Clause::new(vec![
            Literal::negative(q),
            Literal::positive(r.clone()),
        ]));
        // ¬R
        builder.add_clause(&Clause::new(vec![Literal::negative(r)]));
        
        let config = SaturationConfig::default();
        let result = saturate(&mut problem, &config);
        
        assert!(result.found_empty_clause);
        assert!(result.num_clauses_generated >= 3); // Q, R, empty
    }
    
    #[test]
    fn test_saturate_with_factoring() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        let x = Term::Variable("X".to_string());
        let a = Term::Constant("a".to_string());
        let p = Predicate::new("P".to_string(), vec![x.clone()]);
        let p_a = Predicate::new("P".to_string(), vec![a]);
        
        // P(X) ∨ P(a)
        builder.add_clause(&Clause::new(vec![
            Literal::positive(p),
            Literal::positive(p_a.clone()),
        ]));
        // ¬P(a)
        builder.add_clause(&Clause::new(vec![Literal::negative(p_a)]));
        
        let config = SaturationConfig::default();
        let result = saturate(&mut problem, &config);
        
        assert!(result.found_empty_clause);
        // Should factor first clause to P(a), then resolve with ¬P(a)
    }
    
    #[test]
    fn test_saturate_first_order() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        let x = Term::Variable("X".to_string());
        let a = Term::Constant("a".to_string());
        let b = Term::Constant("b".to_string());
        
        // ∀X. P(X) → Q(X)  as  ¬P(X) ∨ Q(X)
        builder.add_clause(&Clause::new(vec![
            Literal::negative(Predicate::new("P".to_string(), vec![x.clone()])),
            Literal::positive(Predicate::new("Q".to_string(), vec![x])),
        ]));
        
        // P(a)
        builder.add_clause(&Clause::new(vec![
            Literal::positive(Predicate::new("P".to_string(), vec![a.clone()])),
        ]));
        
        // P(b)
        builder.add_clause(&Clause::new(vec![
            Literal::positive(Predicate::new("P".to_string(), vec![b])),
        ]));
        
        // ¬Q(a)
        builder.add_clause(&Clause::new(vec![
            Literal::negative(Predicate::new("Q".to_string(), vec![a])),
        ]));
        
        let config = SaturationConfig::default();
        let result = saturate(&mut problem, &config);
        
        assert!(result.found_empty_clause);
    }
    
    #[test]
    fn test_saturate_clause_limit() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // Create a problem that generates many clauses
        for i in 0..5 {
            let p = Predicate::new(format!("P{}", i), vec![]);
            let q = Predicate::new(format!("Q{}", i), vec![]);
            
            builder.add_clause(&Clause::new(vec![
                Literal::positive(p.clone()),
                Literal::positive(q.clone()),
            ]));
            builder.add_clause(&Clause::new(vec![
                Literal::negative(p),
                Literal::positive(q),
            ]));
        }
        
        let mut config = SaturationConfig::default();
        config.max_clauses = 20; // Limit clauses
        
        let result = saturate(&mut problem, &config);
        
        // Should stop due to clause limit
        assert!(problem.num_clauses <= config.max_clauses);
    }
    
    #[test]
    fn test_saturate_iteration_limit() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // Create clauses that don't lead to contradiction
        for i in 0..10 {
            let p = Predicate::new(format!("P{}", i), vec![]);
            builder.add_clause(&Clause::new(vec![Literal::positive(p)]));
        }
        
        let mut config = SaturationConfig::default();
        config.max_iterations = 5;
        
        let result = saturate(&mut problem, &config);
        
        assert!(!result.found_empty_clause);
        assert_eq!(result.num_iterations, 5);
    }
    
    #[test]
    fn test_saturate_tautology_elimination() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        let p = Predicate::new("P".to_string(), vec![]);
        let q = Predicate::new("Q".to_string(), vec![]);
        
        // P ∨ ¬P (tautology)
        builder.add_clause(&Clause::new(vec![
            Literal::positive(p.clone()),
            Literal::negative(p.clone()),
        ]));
        
        // Q
        builder.add_clause(&Clause::new(vec![Literal::positive(q.clone())]));
        
        // ¬Q
        builder.add_clause(&Clause::new(vec![Literal::negative(q)]));
        
        let config = SaturationConfig::default();
        let result = saturate(&mut problem, &config);
        
        // Should still find contradiction despite tautology
        assert!(result.found_empty_clause);
    }
    
    #[test]
    fn test_saturate_complex_problem() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        let x = Term::Variable("X".to_string());
        let y = Term::Variable("Y".to_string());
        let a = Term::Constant("a".to_string());
        let f_x = Term::Function { name: "f".to_string(), args: vec![x.clone()] };
        let f_a = Term::Function { name: "f".to_string(), args: vec![a.clone()] };
        
        // Complex first-order clauses
        builder.add_clause(&Clause::new(vec![
            Literal::positive(Predicate::new("P".to_string(), vec![x.clone(), f_x])),
        ]));
        
        builder.add_clause(&Clause::new(vec![
            Literal::negative(Predicate::new("P".to_string(), vec![y.clone(), f_a.clone()])),
            Literal::positive(Predicate::new("Q".to_string(), vec![y])),
        ]));
        
        builder.add_clause(&Clause::new(vec![
            Literal::negative(Predicate::new("Q".to_string(), vec![a])),
        ]));
        
        let config = SaturationConfig::default();
        let result = saturate(&mut problem, &config);
        
        assert!(result.found_empty_clause);
    }
    
    #[test]
    fn test_saturate_preserves_invariants() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // Add some test clauses
        let p = Predicate::new("P".to_string(), vec![]);
        let q = Predicate::new("Q".to_string(), vec![]);
        
        builder.add_clause(&Clause::new(vec![
            Literal::positive(p.clone()),
            Literal::positive(q.clone()),
        ]));
        builder.add_clause(&Clause::new(vec![Literal::negative(p)]));
        builder.add_clause(&Clause::new(vec![Literal::negative(q)]));
        
        let config = SaturationConfig::default();
        let result = saturate(&mut problem, &config);
        
        // Verify invariants
        assert_eq!(problem.clause_boundaries.len(), problem.num_clauses + 1);
        assert_eq!(problem.edge_row_offsets.len(), problem.num_nodes + 1);
        assert_eq!(problem.node_types.len(), problem.num_nodes);
        
        // Verify all clause boundaries are valid
        for i in 0..problem.num_clauses {
            let (start, end) = problem.clause_node_range(i).unwrap();
            assert!(start < end);
            assert!(end <= problem.num_nodes);
        }
    }
    
    #[test]
    fn test_saturate_performance() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // Create a larger problem
        let vars = vec![
            Term::Variable("X".to_string()),
            Term::Variable("Y".to_string()),
            Term::Variable("Z".to_string()),
        ];
        
        let consts = vec![
            Term::Constant("a".to_string()),
            Term::Constant("b".to_string()),
            Term::Constant("c".to_string()),
        ];
        
        // Add various clauses
        for (i, var) in vars.iter().enumerate() {
            for (j, c) in consts.iter().enumerate() {
                if i != j {
                    let p = Predicate::new("P".to_string(), vec![var.clone(), c.clone()]);
                    builder.add_clause(&Clause::new(vec![Literal::positive(p)]));
                }
            }
        }
        
        // Add some negative clauses
        for c in &consts {
            let p = Predicate::new("P".to_string(), vec![c.clone(), c.clone()]);
            builder.add_clause(&Clause::new(vec![Literal::negative(p)]));
        }
        
        let mut config = SaturationConfig::default();
        config.max_iterations = 100;
        config.max_clauses = 1000;
        
        let result = saturate(&mut problem, &config);
        
        // Should complete within limits
        assert!(result.num_iterations <= config.max_iterations);
        assert!(problem.num_clauses <= config.max_clauses);
    }
}