//! Tests for Proof and ProofStep

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::core::logic::{Clause, Literal, Predicate};
    use std::collections::HashMap;
    use serde_json::json;
    
    fn make_test_state() -> ProofState {
        ProofState::new(
            vec![Clause::new(vec![
                Literal {
                    predicate: Predicate::new("p".to_string(), vec![]),
                    polarity: true,
                }
            ])],
            vec![Clause::new(vec![
                Literal {
                    predicate: Predicate::new("q".to_string(), vec![]),
                    polarity: false,
                }
            ])],
        )
    }
    
    #[test]
    fn test_rule_application_creation() {
        let rule = RuleApplication::new("resolution".to_string(), vec![0, 1]);
        assert_eq!(rule.rule_name, "resolution");
        assert_eq!(rule.parents, vec![0, 1]);
        assert!(rule.generated_clauses.is_empty());
        assert!(rule.deleted_clause_indices.is_empty());
        assert!(rule.metadata.is_empty());
    }
    
    #[test]
    fn test_rule_application_builder() {
        let clause = Clause::new(vec![
            Literal {
                predicate: Predicate::new("r".to_string(), vec![]),
                polarity: true,
            }
        ]);
        
        let rule = RuleApplication::new("factoring".to_string(), vec![0])
            .with_generated(vec![clause])
            .with_deleted(vec![1, 2])
            .with_metadata("factor_count".to_string(), json!(2));
        
        assert_eq!(rule.generated_clauses.len(), 1);
        assert_eq!(rule.deleted_clause_indices, vec![1, 2]);
        assert_eq!(rule.metadata.get("factor_count"), Some(&json!(2)));
    }
    
    #[test]
    fn test_proofstep_creation() {
        let state = make_test_state();
        let step = ProofStep::new(state.clone());
        
        assert_eq!(step.state, state);
        assert_eq!(step.selected_clause, None);
        assert!(step.applied_rules.is_empty());
        assert!(step.metadata.is_empty());
    }
    
    #[test]
    fn test_proofstep_builder() {
        let state = make_test_state();
        let rule = RuleApplication::new("resolution".to_string(), vec![0, 1]);
        
        let step = ProofStep::new(state)
            .with_selected_clause(0)
            .with_applied_rules(vec![rule])
            .with_metadata("time_ms".to_string(), json!(150));
        
        assert_eq!(step.selected_clause, Some(0));
        assert_eq!(step.applied_rules.len(), 1);
        assert_eq!(step.metadata.get("time_ms"), Some(&json!(150)));
    }
    
    #[test]
    fn test_proof_creation() {
        let proof = Proof::empty();
        assert_eq!(proof.steps.len(), 1);
        assert_eq!(proof.length(), 0);
        
        let state = make_test_state();
        let proof = Proof::new(state);
        assert_eq!(proof.steps.len(), 1);
        assert_eq!(proof.length(), 0);
    }
    
    #[test]
    fn test_proof_add_step() {
        let mut proof = Proof::empty();
        let state = make_test_state();
        
        // Proof::empty() creates a proof with one initial step
        assert_eq!(proof.steps.len(), 1);
        assert_eq!(proof.length(), 0);
        
        // Add step with selection
        let step = ProofStep::new(state.clone()).with_selected_clause(0);
        proof.add_step(step);
        
        // Since the initial step has no selection, it gets replaced
        assert_eq!(proof.steps.len(), 1);
        assert_eq!(proof.length(), 1);
        
        // Add another step
        let step2 = ProofStep::new(state).with_selected_clause(1);
        proof.add_step(step2);
        
        assert_eq!(proof.steps.len(), 2);
        assert_eq!(proof.length(), 2);
    }
    
    #[test]
    fn test_proof_finalize() {
        let mut proof = Proof::empty();
        let state = make_test_state();
        
        // Add some steps
        proof.add_step(ProofStep::new(state.clone()).with_selected_clause(0));
        proof.add_step(ProofStep::new(state.clone()).with_selected_clause(1));
        
        // Finalize
        let final_state = ProofState::new(vec![Clause::new(vec![])], vec![]);
        proof.finalize(final_state);
        
        // Check last step has no selection
        assert_eq!(proof.steps.last().unwrap().selected_clause, None);
        assert!(proof.found_contradiction());
    }
    
    #[test]
    fn test_proof_properties() {
        let mut proof = Proof::empty();
        let state = make_test_state();
        
        // Test initial/final state
        assert!(proof.initial_state().is_some());
        assert!(proof.final_state().is_some());
        
        // Add steps with metadata
        let step = ProofStep::new(state.clone())
            .with_selected_clause(0)
            .with_metadata("score".to_string(), json!(0.5));
        proof.add_step(step);
        
        let step2 = ProofStep::new(state)
            .with_selected_clause(1)
            .with_metadata("score".to_string(), json!(0.7));
        proof.add_step(step2);
        
        // Test get_step
        assert!(proof.get_step(0).is_some());
        assert!(proof.get_step(1).is_some());
        assert!(proof.get_step(10).is_none());
        
        // Test get_selected_clauses
        let selected = proof.get_selected_clauses();
        assert_eq!(selected, vec![0, 1]);
        
        // Test metadata history
        let scores = proof.get_metadata_history("score");
        assert_eq!(scores.len(), 2);
        assert_eq!(scores[0], &json!(0.5));
        assert_eq!(scores[1], &json!(0.7));
    }
    
    #[test]
    fn test_proof_serialization() {
        let mut proof = Proof::empty();
        let state = make_test_state();
        
        proof.add_step(ProofStep::new(state).with_selected_clause(0));
        
        // Test that proof can be serialized
        let json = serde_json::to_string(&proof).unwrap();
        assert!(json.contains("steps"));
        
        // Test deserialization
        let proof2: Proof = serde_json::from_str(&json).unwrap();
        assert_eq!(proof2.steps.len(), proof.steps.len());
    }
}