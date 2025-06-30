//! Tests for proof tracking types

#[cfg(test)]
mod tests {
    use super::super::*;
    
    #[test]
    fn test_inference_rule_equality() {
        let rule1 = InferenceRule::Resolution { lit1_idx: 0, lit2_idx: 1 };
        let rule2 = InferenceRule::Resolution { lit1_idx: 0, lit2_idx: 1 };
        let rule3 = InferenceRule::Resolution { lit1_idx: 1, lit2_idx: 0 };
        let rule4 = InferenceRule::Factoring { lit_indices: vec![0, 1] };
        
        assert_eq!(rule1, rule2);
        assert_ne!(rule1, rule3);
        assert_ne!(rule1, rule4);
    }
    
    #[test]
    fn test_proof_step_creation() {
        let step = ProofStep {
            rule: InferenceRule::Resolution { lit1_idx: 0, lit2_idx: 1 },
            parents: vec![0, 1],
            selected_literals: vec![0, 1],
            given_clause: Some(0),
            added_clauses: vec![2],
            deleted_clauses: vec![],
        };
        
        assert_eq!(step.parents.len(), 2);
        assert_eq!(step.given_clause, Some(0));
        assert_eq!(step.added_clauses, vec![2]);
        assert!(step.deleted_clauses.is_empty());
    }
    
    #[test]
    fn test_superposition_rule() {
        let rule = InferenceRule::Superposition {
            from_lit: 0,
            into_lit: 1,
            position: vec![0, 1],
            positive: true,
        };
        
        match rule {
            InferenceRule::Superposition { from_lit, into_lit, position, positive } => {
                assert_eq!(from_lit, 0);
                assert_eq!(into_lit, 1);
                assert_eq!(position, vec![0, 1]);
                assert!(positive);
            }
            _ => panic!("Wrong rule type"),
        }
    }
    
    #[test]
    fn test_subsumption_rules() {
        let forward = InferenceRule::ForwardSubsumption;
        let backward = InferenceRule::BackwardSubsumption;
        
        assert_ne!(forward, backward);
        
        // Test in proof steps
        let step1 = ProofStep {
            rule: forward,
            parents: vec![5],
            selected_literals: vec![],
            given_clause: None,
            added_clauses: vec![],
            deleted_clauses: vec![3, 7],
        };
        
        assert_eq!(step1.deleted_clauses.len(), 2);
    }
    
    #[test]
    fn test_proof_structure() {
        let mut problem = Problem::with_capacity(10, 5, 20);
        
        // Add some dummy clauses
        problem.num_clauses = 3;
        problem.clause_boundaries[0] = 0;
        problem.clause_boundaries[1] = 1;
        problem.clause_boundaries[2] = 2;
        problem.clause_boundaries[3] = 3;
        
        let proof = Proof {
            problem,
            steps: vec![
                ProofStep {
                    rule: InferenceRule::Resolution { lit1_idx: 0, lit2_idx: 1 },
                    parents: vec![0, 1],
                    selected_literals: vec![0, 1],
                    given_clause: Some(0),
                    added_clauses: vec![3],
                    deleted_clauses: vec![],
                },
                ProofStep {
                    rule: InferenceRule::BackwardSubsumption,
                    parents: vec![2],
                    selected_literals: vec![],
                    given_clause: None,
                    added_clauses: vec![],
                    deleted_clauses: vec![3],
                },
            ],
            clause_first_step: vec![0, 0, 0, 1],
            clause_last_step: vec![1, 1, 1, 1],
        };
        
        assert_eq!(proof.steps.len(), 2);
        assert_eq!(proof.problem.num_clauses, 3);
        assert_eq!(proof.clause_first_step.len(), 4);
    }
    
    #[test]
    fn test_saturation_result() {
        let problem = Problem::new();
        let proof = Proof {
            problem,
            steps: vec![],
            clause_first_step: vec![],
            clause_last_step: vec![],
        };
        
        // Test different result types
        let result1 = SaturationResult::Proof(proof);
        let result2 = SaturationResult::Saturated;
        let result3 = SaturationResult::ResourceLimit;
        
        // Verify we can match on them
        match result1 {
            SaturationResult::Proof(_) => (),
            _ => panic!("Wrong result type"),
        }
        
        match result2 {
            SaturationResult::Saturated => (),
            _ => panic!("Wrong result type"),
        }
        
        match result3 {
            SaturationResult::ResourceLimit => (),
            _ => panic!("Wrong result type"),
        }
    }
    
    #[test]
    fn test_equality_rules() {
        let eq_res = InferenceRule::EqualityResolution;
        let eq_fact = InferenceRule::EqualityFactoring;
        
        assert_ne!(eq_res, eq_fact);
        
        // Test factoring with lit indices
        let fact = InferenceRule::Factoring { lit_indices: vec![1, 3, 5] };
        match fact {
            InferenceRule::Factoring { lit_indices } => {
                assert_eq!(lit_indices.len(), 3);
                assert_eq!(lit_indices, vec![1, 3, 5]);
            }
            _ => panic!("Wrong rule type"),
        }
    }
}