//! Re-exports from crate::state for backward compatibility
pub use crate::state::{ProofResult, SaturationState, Proof, ProofStep};
pub use crate::config::{LiteralSelectionStrategy, ProverConfig};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::{CNFFormula, Clause, Constant, Interner, Literal, PredicateSymbol, Term, Variable};
    use crate::selection::{AgeWeightSelector, ClauseSelector};

    fn create_selector() -> Box<dyn ClauseSelector> {
        Box::new(AgeWeightSelector::default())
    }

    /// Helper to create a simple proof problem with interner
    fn create_simple_problem() -> (CNFFormula, Interner) {
        let mut interner = Interner::new();

        let p_id = interner.intern_predicate("P");
        let q_id = interner.intern_predicate("Q");
        let a_id = interner.intern_constant("a");
        let x_id = interner.intern_variable("X");

        let p = PredicateSymbol::new(p_id, 1);
        let q = PredicateSymbol::new(q_id, 1);
        let a = Term::Constant(Constant::new(a_id));
        let x = Term::Variable(Variable::new(x_id));

        let clauses = vec![
            Clause::new(vec![Literal::positive(p, vec![a.clone()])]),
            Clause::new(vec![
                Literal::negative(p, vec![x.clone()]),
                Literal::positive(q, vec![x.clone()]),
            ]),
            Clause::new(vec![Literal::negative(q, vec![a.clone()])]),
        ];

        (CNFFormula { clauses }, interner)
    }

    #[test]
    fn test_simple_proof() {
        let (formula, interner) = create_simple_problem();
        let (result, profile, _, _) = crate::saturation::saturate(formula, ProverConfig::default(), create_selector(), interner);

        match result {
            ProofResult::Proof(_) => {} // Expected
            _ => panic!("Expected to find proof"),
        }
        assert!(profile.is_none(), "Profiling should be disabled by default");
    }

    #[test]
    fn test_profiling_enabled() {
        let (formula, interner) = create_simple_problem();
        let mut config = ProverConfig::default();
        config.enable_profiling = true;
        let (result, profile, _, _) = crate::saturation::saturate(formula, config, create_selector(), interner);

        match result {
            ProofResult::Proof(_) => {}
            _ => panic!("Expected to find proof"),
        }

        let profile = profile.expect("Profile should be present when enabled");
        assert!(profile.total_time.as_nanos() > 0, "total_time should be non-zero");
        assert!(profile.iterations > 0, "iterations should be non-zero");
        assert!(
            profile.generating_rules.get("Resolution").map_or(0, |s| s.count) > 0,
            "should have resolution inferences"
        );
        assert_eq!(profile.selector_name, "AgeWeight");

        // Verify JSON serialization works
        let json = serde_json::to_string(&profile).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(value["total_time"].as_f64().unwrap() > 0.0);
        assert!(value["iterations"].as_u64().unwrap() > 0);
        assert_eq!(value["selector_name"].as_str().unwrap(), "AgeWeight");
        assert!(value["generating_rules"]["Resolution"]["count"].as_u64().unwrap() > 0);
    }

    #[test]
    fn test_event_log_populated() {
        use crate::state::StateChange;

        let (formula, interner) = create_simple_problem();
        let (result, _, event_log, _) = crate::saturation::saturate(formula, ProverConfig::default(), create_selector(), interner);

        assert!(matches!(result, ProofResult::Proof(_)));

        assert!(!event_log.is_empty(), "Should have events in the log");

        let new_count = event_log.iter().filter(|e| matches!(e, StateChange::Add { .. })).count();
        assert!(new_count >= 3, "Should have at least 3 New events for initial clauses");

        let json = serde_json::to_string(&event_log).unwrap();
        let parsed: Vec<StateChange> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.len(), event_log.len());
    }
}
