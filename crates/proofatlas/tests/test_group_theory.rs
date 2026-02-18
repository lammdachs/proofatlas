//! End-to-end tests for group theory problems

use proofatlas::{
    parse_tptp_file, AgeWeightSink, ProverSink, LiteralSelectionStrategy,
    ProverConfig, ProofResult, Prover,
};
use std::time::Duration;

fn create_sink() -> Box<dyn ProverSink> {
    Box::new(AgeWeightSink::new(0.5))
}

/// Run a group theory problem and return (result, prover)
fn run_group_problem(problem_file: &str, timeout_secs: u64) -> (ProofResult, Prover) {
    let parsed = parse_tptp_file(problem_file, &[], None, None).expect("Failed to parse TPTP file");

    let config = ProverConfig {
        max_clauses: 10000,
        max_iterations: 10000,
        timeout: Duration::from_secs(timeout_secs),
        literal_selection: LiteralSelectionStrategy::Sel0,
        ..Default::default()
    };

    let mut prover = Prover::new(parsed.formula.clauses, config, create_sink(), parsed.interner);
    let result = prover.prove();
    (result, prover)
}

#[test]
fn test_right_identity() {
    let (result, prover) = run_group_problem("tests/problems/right_identity.p", 10);
    match result {
        ProofResult::Proof { empty_clause_idx } => {
            prover.verify_proof(empty_clause_idx).expect("proof verification failed");
        }
        _ => panic!("Failed to prove right identity: {:?}", result),
    }
}

#[test]
fn test_uniqueness_of_identity() {
    let (result, prover) = run_group_problem("tests/problems/uniqueness_of_identity.p", 10);
    match result {
        ProofResult::Proof { empty_clause_idx } => {
            prover.verify_proof(empty_clause_idx).expect("proof verification failed");
        }
        _ => panic!("Failed to prove uniqueness of identity: {:?}", result),
    }
}

#[test]
fn test_right_inverse() {
    let (result, prover) = run_group_problem("tests/problems/right_inverse.p", 10);
    match result {
        ProofResult::Proof { empty_clause_idx } => {
            prover.verify_proof(empty_clause_idx).expect("proof verification failed");
        }
        _ => panic!("Failed to prove right inverse: {:?}", result),
    }
}

#[test]
fn test_uniqueness_of_inverse() {
    let (result, prover) = run_group_problem("tests/problems/uniqueness_of_inverse.p", 10);
    match result {
        ProofResult::Proof { empty_clause_idx } => {
            prover.verify_proof(empty_clause_idx).expect("proof verification failed");
        }
        _ => panic!("Failed to prove uniqueness of inverse: {:?}", result),
    }
}

#[test]
fn test_inverse_of_identity() {
    let (result, prover) = run_group_problem("tests/problems/inverse_of_identity.p", 10);
    match result {
        ProofResult::Proof { empty_clause_idx } => {
            prover.verify_proof(empty_clause_idx).expect("proof verification failed");
        }
        _ => panic!("Failed to prove inverse of identity is identity: {:?}", result),
    }
}

#[test]
fn test_inverse_of_inverse() {
    let (result, prover) = run_group_problem("tests/problems/inverse_of_inverse.p", 10);
    match result {
        ProofResult::Proof { empty_clause_idx } => {
            prover.verify_proof(empty_clause_idx).expect("proof verification failed");
        }
        _ => panic!("Failed to prove inverse of inverse: {:?}", result),
    }
}
