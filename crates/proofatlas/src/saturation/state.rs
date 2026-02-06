//! Saturation state: clause sets and configuration types.
//!
//! `SaturationState` is a lean data container holding the three clause sets
//! (N, U, P), the clause storage, and the event log. The saturation algorithm
//! itself lives in `ProofAtlas` (`prover.rs`).

use super::rule::StateChange;
use crate::fol::{Clause, Interner};
use crate::inference::{Proof, ProofStep};
use indexmap::IndexSet;
use std::collections::VecDeque;
use std::time::Duration;

/// Configuration for the saturation loop
#[derive(Debug, Clone)]
pub struct ProverConfig {
    pub max_clauses: usize,
    pub max_iterations: usize,
    pub max_clause_size: usize,
    pub timeout: Duration,
    pub literal_selection: LiteralSelectionStrategy,
    /// Memory limit for clause storage in MB (directly comparable across provers)
    pub max_clause_memory_mb: Option<usize>,
    /// Enable structured profiling (zero overhead when false)
    pub enable_profiling: bool,
}

/// Literal selection strategies (numbers match Vampire's --selection option)
///
/// From Hoder et al. "Selecting the selection" (2016):
/// - Sel0: Select all literals
/// - Sel20: Select all maximal literals
/// - Sel21: Select unique maximal, else max-weight negative, else all maximal
/// - Sel22: Select max-weight negative literal, else all maximal
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiteralSelectionStrategy {
    /// Selection 0: Select all literals (no selection)
    Sel0,
    /// Selection 20: Select all maximal literals
    Sel20,
    /// Selection 21: Unique maximal, else max-weight negative, else all maximal
    Sel21,
    /// Selection 22: Max-weight negative literal, else all maximal
    Sel22,
}

impl Default for ProverConfig {
    fn default() -> Self {
        ProverConfig {
            max_clauses: 0,      // 0 means no limit
            max_iterations: 0,   // 0 means no limit
            max_clause_size: 100,
            timeout: Duration::from_secs(60),
            literal_selection: LiteralSelectionStrategy::Sel21,
            max_clause_memory_mb: None,
            enable_profiling: false,
        }
    }
}

/// Result of saturation
#[derive(Debug, Clone)]
pub enum ProofResult {
    /// Empty clause derived - proof found
    Proof(Proof),
    /// Saturated without finding empty clause
    Saturated(Vec<ProofStep>, Vec<Clause>),
    /// Resource limit reached
    ResourceLimit(Vec<ProofStep>, Vec<Clause>),
    /// Timeout reached
    Timeout(Vec<ProofStep>, Vec<Clause>),
}

impl ProofResult {
    /// Convert to JSON representation
    pub fn to_json(&self, time_seconds: f64, interner: &Interner) -> crate::json::ProofResultJson {
        use crate::json::{ClauseJson, ProofResultJson};

        match self {
            ProofResult::Proof(proof) => ProofResultJson::Proof {
                proof: proof.to_json(interner),
                time_seconds,
            },
            ProofResult::Saturated(steps, clauses) => ProofResultJson::Saturated {
                final_clauses: clauses.iter().map(|c| ClauseJson::from_clause(c, interner)).collect(),
                proof_steps: steps.iter().map(|s| s.into()).collect(),
                time_seconds,
            },
            ProofResult::ResourceLimit(steps, clauses) => {
                ProofResultJson::ResourceLimit {
                    reason: "Clause or iteration limit exceeded".to_string(),
                    final_clauses: clauses.iter().map(|c| ClauseJson::from_clause(c, interner)).collect(),
                    proof_steps: steps.iter().map(|s| s.into()).collect(),
                    time_seconds,
                }
            }
            ProofResult::Timeout(steps, clauses) => ProofResultJson::Timeout {
                final_clauses: clauses.iter().map(|c| ClauseJson::from_clause(c, interner)).collect(),
                proof_steps: steps.iter().map(|s| s.into()).collect(),
                time_seconds,
            },
        }
    }
}

/// Lean data container for the three-set given-clause algorithm.
///
/// Holds clause storage, the N/U/P sets, and the event log.
/// The saturation algorithm lives in `ProofAtlas`.
pub struct SaturationState {
    /// Storage for all clauses, indexed by clause ID
    pub clauses: Vec<Clause>,
    /// Set P: Processed/active clauses (used for generating inferences)
    pub processed: IndexSet<usize>,
    /// Set U: Unprocessed/passive clauses (awaiting selection)
    pub unprocessed: IndexSet<usize>,
    /// Set N: New clauses (awaiting forward simplification)
    pub new: VecDeque<usize>,
    /// Raw event log capturing all state changes
    pub event_log: Vec<StateChange>,
    /// Tracked clause memory usage in bytes
    pub clause_memory_bytes: usize,
    /// Current iteration (used for clause age)
    pub current_iteration: usize,
    /// Number of initial input clauses
    pub initial_clause_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fol::{CNFFormula, Constant, Literal, PredicateSymbol, Term, Variable};
    use crate::selection::{AgeWeightSelector, ClauseSelector};

    fn create_selector() -> Box<dyn ClauseSelector> {
        Box::new(AgeWeightSelector::default())
    }

    /// Helper to create a simple proof problem with interner
    fn create_simple_problem() -> (CNFFormula, Interner) {
        // P(a)
        // ~P(X) | Q(X)
        // ~Q(a)
        // Should derive empty clause

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
        // Verify generating_rules structure
        assert!(value["generating_rules"]["Resolution"]["count"].as_u64().unwrap() > 0);
    }

    #[test]
    fn test_event_log_populated() {
        use crate::saturation::rule::StateChange;

        let (formula, interner) = create_simple_problem();
        let (result, _, event_log, _) = crate::saturation::saturate(formula, ProverConfig::default(), create_selector(), interner);

        assert!(matches!(result, ProofResult::Proof(_)));

        // Event log should have content
        assert!(!event_log.is_empty(), "Should have events in the log");

        // Should have at least the initial 3 New events
        let new_count = event_log.iter().filter(|e| matches!(e, StateChange::Add { .. })).count();
        assert!(new_count >= 3, "Should have at least 3 New events for initial clauses");

        // Verify serde serialization round-trips
        let json = serde_json::to_string(&event_log).unwrap();
        let parsed: Vec<StateChange> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.len(), event_log.len());
    }
}
