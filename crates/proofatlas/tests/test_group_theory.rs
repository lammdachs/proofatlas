//! End-to-end tests for group theory problems

use proofatlas::{
    parse_tptp_file, AgeWeightSelector, ClauseSelector, LiteralSelectionStrategy,
    ProverConfig, ProofResult, ProofAtlas,
};
use std::time::Duration;

fn create_selector() -> Box<dyn ClauseSelector> {
    Box::new(AgeWeightSelector::default())
}

/// Run a group theory problem and return the result
fn run_group_problem(problem_file: &str, timeout_secs: u64) -> ProofResult {
    let parsed = parse_tptp_file(problem_file, &[], None, None).expect("Failed to parse TPTP file");

    let config = ProverConfig {
        max_clauses: 10000,
        max_iterations: 10000,
        timeout: Duration::from_secs(timeout_secs),
        literal_selection: LiteralSelectionStrategy::Sel0,
        ..Default::default()
    };

    let mut prover = ProofAtlas::new(parsed.formula.clauses, config, create_selector(), parsed.interner);
    prover.prove()
}

#[test]
fn test_right_identity() {
    let result = run_group_problem("tests/problems/right_identity.p", 10);
    assert!(matches!(result, ProofResult::Proof { .. }), "Failed to prove right identity");
}

#[test]
fn test_uniqueness_of_identity() {
    let result = run_group_problem("tests/problems/uniqueness_of_identity.p", 10);
    assert!(matches!(result, ProofResult::Proof { .. }), "Failed to prove uniqueness of identity");
}

#[test]
fn test_right_inverse() {
    let result = run_group_problem("tests/problems/right_inverse.p", 10);
    assert!(matches!(result, ProofResult::Proof { .. }), "Failed to prove right inverse");
}

#[test]
fn test_uniqueness_of_inverse() {
    let result = run_group_problem("tests/problems/uniqueness_of_inverse.p", 10);
    assert!(matches!(result, ProofResult::Proof { .. }), "Failed to prove uniqueness of inverse");
}

#[test]
fn test_inverse_of_identity() {
    let result = run_group_problem("tests/problems/inverse_of_identity.p", 10);
    assert!(matches!(result, ProofResult::Proof { .. }), "Failed to prove inverse of identity is identity");
}

#[test]
fn test_inverse_of_inverse() {
    let result = run_group_problem("tests/problems/inverse_of_inverse.p", 10);
    assert!(matches!(result, ProofResult::Proof { .. }), "Failed to prove inverse of inverse");
}
