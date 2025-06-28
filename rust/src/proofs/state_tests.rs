//! Tests for ProofState

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::core::logic::{Clause, Literal, Predicate, Term};
    
    fn make_test_clause(literals: Vec<(&str, bool)>) -> Clause {
        let lits: Vec<Literal> = literals.into_iter()
            .map(|(name, polarity)| {
                Literal {
                    predicate: Predicate::new(name.to_string(), Vec::new()),
                    polarity,
                }
            })
            .collect();
        Clause::new(lits)
    }
    
    #[test]
    fn test_proofstate_creation() {
        let state = ProofState::empty();
        assert_eq!(state.num_processed(), 0);
        assert_eq!(state.num_unprocessed(), 0);
        assert_eq!(state.num_clauses(), 0);
    }
    
    #[test]
    fn test_proofstate_with_clauses() {
        let processed = vec![
            make_test_clause(vec![("p", true)]),
            make_test_clause(vec![("q", false)]),
        ];
        let unprocessed = vec![
            make_test_clause(vec![("r", true), ("s", false)]),
        ];
        
        let state = ProofState::new(processed, unprocessed);
        assert_eq!(state.num_processed(), 2);
        assert_eq!(state.num_unprocessed(), 1);
        assert_eq!(state.num_clauses(), 3);
    }
    
    #[test]
    fn test_add_clauses() {
        let mut state = ProofState::empty();
        
        state.add_processed(make_test_clause(vec![("p", true)]));
        assert_eq!(state.num_processed(), 1);
        
        state.add_unprocessed(make_test_clause(vec![("q", false)]));
        assert_eq!(state.num_unprocessed(), 1);
        
        state.add_unprocessed_many(vec![
            make_test_clause(vec![("r", true)]),
            make_test_clause(vec![("s", false)]),
        ]);
        assert_eq!(state.num_unprocessed(), 3);
    }
    
    #[test]
    fn test_move_to_processed() {
        let mut state = ProofState::empty();
        state.add_unprocessed(make_test_clause(vec![("p", true)]));
        state.add_unprocessed(make_test_clause(vec![("q", false)]));
        
        assert_eq!(state.num_processed(), 0);
        assert_eq!(state.num_unprocessed(), 2);
        
        // Move first clause
        let moved = state.move_to_processed(0);
        assert!(moved.is_some());
        assert_eq!(state.num_processed(), 1);
        assert_eq!(state.num_unprocessed(), 1);
        
        // Try invalid index
        let moved = state.move_to_processed(5);
        assert!(moved.is_none());
    }
    
    #[test]
    fn test_contains_empty_clause() {
        let mut state = ProofState::empty();
        assert!(!state.contains_empty_clause());
        
        // Add non-empty clause
        state.add_processed(make_test_clause(vec![("p", true)]));
        assert!(!state.contains_empty_clause());
        
        // Add empty clause
        state.add_unprocessed(Clause::new(vec![]));
        assert!(state.contains_empty_clause());
    }
    
    #[test]
    fn test_all_clauses() {
        let mut state = ProofState::empty();
        state.add_processed(make_test_clause(vec![("p", true)]));
        state.add_processed(make_test_clause(vec![("q", false)]));
        state.add_unprocessed(make_test_clause(vec![("r", true)]));
        
        let all = state.all_clauses();
        assert_eq!(all.len(), 3);
    }
}