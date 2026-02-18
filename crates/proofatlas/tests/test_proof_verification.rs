//! Integration tests for proof verification.
//!
//! Every test that expects a proof also runs verify_proof() to check soundness.

use proofatlas::{
    parse_tptp, parse_tptp_file, saturate, AgeWeightSink, LiteralSelectionStrategy, ProofResult,
    Prover, ProverConfig, ProverSink,
};
use std::time::Duration;

fn create_sink() -> Box<dyn ProverSink> {
    Box::new(AgeWeightSink::new(0.5))
}

fn prove_tptp(tptp: &str) -> (ProofResult, Prover) {
    let parsed = parse_tptp(tptp, &[], None, None).unwrap();
    let config = ProverConfig::default();
    saturate(parsed.formula, config, create_sink(), parsed.interner)
}

fn prove_file(path: &str, timeout_secs: u64) -> (ProofResult, Prover) {
    let parsed = parse_tptp_file(path, &[], None, None).expect("Failed to parse TPTP file");
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

fn assert_proof_verified(result: ProofResult, prover: &Prover, name: &str) {
    match result {
        ProofResult::Proof { empty_clause_idx } => {
            prover.verify_proof(empty_clause_idx)
                .unwrap_or_else(|e| panic!("{}: proof verification failed: {}", name, e));
        }
        other => panic!("{}: expected proof, got {:?}", name, other),
    }
}

// =========================================================================
// Pure propositional (resolution only)
// =========================================================================

#[test]
fn test_verify_simple_resolution() {
    let (result, prover) = prove_tptp(r#"
        cnf(c1, axiom, p(a)).
        cnf(c2, axiom, ~p(X) | q(X)).
        cnf(c3, negated_conjecture, ~q(a)).
    "#);
    assert_proof_verified(result, &prover, "simple resolution");
}

#[test]
fn test_verify_chain_resolution() {
    let (result, prover) = prove_tptp(r#"
        cnf(c1, axiom, p(a)).
        cnf(c2, axiom, ~p(X) | q(X)).
        cnf(c3, axiom, ~q(X) | r(X)).
        cnf(c4, negated_conjecture, ~r(a)).
    "#);
    assert_proof_verified(result, &prover, "chain resolution");
}

#[test]
fn test_verify_ground_propositional() {
    let (result, prover) = prove_tptp(r#"
        cnf(c1, axiom, p | q).
        cnf(c2, axiom, ~p | q).
        cnf(c3, axiom, p | ~q).
        cnf(c4, axiom, ~p | ~q).
    "#);
    assert_proof_verified(result, &prover, "ground propositional");
}

// =========================================================================
// Equality reasoning (superposition, demodulation, equality resolution)
// =========================================================================

#[test]
fn test_verify_equality_reflexivity() {
    let (result, prover) = prove_tptp(r#"
        cnf(c1, negated_conjecture, a != a).
    "#);
    assert_proof_verified(result, &prover, "equality reflexivity");
}

#[test]
fn test_verify_equality_symmetry() {
    let (result, prover) = prove_tptp(r#"
        cnf(c1, axiom, a = b).
        cnf(c2, negated_conjecture, b != a).
    "#);
    assert_proof_verified(result, &prover, "equality symmetry");
}

#[test]
fn test_verify_equality_transitivity() {
    let (result, prover) = prove_tptp(r#"
        cnf(c1, axiom, a = b).
        cnf(c2, axiom, b = c).
        cnf(c3, negated_conjecture, a != c).
    "#);
    assert_proof_verified(result, &prover, "equality transitivity");
}

#[test]
fn test_verify_function_congruence() {
    let (result, prover) = prove_tptp(r#"
        cnf(c1, axiom, a = b).
        cnf(c2, negated_conjecture, f(a) != f(b)).
    "#);
    assert_proof_verified(result, &prover, "function congruence");
}

#[test]
fn test_verify_superposition_basic() {
    let (result, prover) = prove_tptp(r#"
        cnf(c1, axiom, f(a) = b).
        cnf(c2, axiom, p(f(a))).
        cnf(c3, negated_conjecture, ~p(b)).
    "#);
    assert_proof_verified(result, &prover, "superposition basic");
}

// =========================================================================
// Group theory problems (all 7)
// =========================================================================

#[test]
fn test_verify_right_identity() {
    let (result, prover) = prove_file("tests/problems/right_identity.p", 10);
    assert_proof_verified(result, &prover, "right identity");
}

#[test]
fn test_verify_uniqueness_of_identity() {
    let (result, prover) = prove_file("tests/problems/uniqueness_of_identity.p", 10);
    assert_proof_verified(result, &prover, "uniqueness of identity");
}

#[test]
fn test_verify_right_inverse() {
    let (result, prover) = prove_file("tests/problems/right_inverse.p", 10);
    assert_proof_verified(result, &prover, "right inverse");
}

#[test]
fn test_verify_uniqueness_of_inverse() {
    let (result, prover) = prove_file("tests/problems/uniqueness_of_inverse.p", 10);
    assert_proof_verified(result, &prover, "uniqueness of inverse");
}

#[test]
fn test_verify_inverse_of_identity() {
    let (result, prover) = prove_file("tests/problems/inverse_of_identity.p", 10);
    assert_proof_verified(result, &prover, "inverse of identity");
}

#[test]
fn test_verify_inverse_of_inverse() {
    let (result, prover) = prove_file("tests/problems/inverse_of_inverse.p", 10);
    assert_proof_verified(result, &prover, "inverse of inverse");
}

#[test]
fn test_verify_order_2_implies_abelian() {
    let (result, prover) = prove_file("tests/problems/order_2_implies_abelian.p", 30);
    assert_proof_verified(result, &prover, "order 2 implies abelian");
}

// =========================================================================
// Mixed (equality + predicates)
// =========================================================================

#[test]
fn test_verify_mixed_equality_predicates() {
    let (result, prover) = prove_tptp(r#"
        cnf(c1, axiom, f(X) = g(X)).
        cnf(c2, axiom, p(f(a))).
        cnf(c3, negated_conjecture, ~p(g(a))).
    "#);
    assert_proof_verified(result, &prover, "mixed equality and predicates");
}

// =========================================================================
// Soundness check: satisfiable formulas must NOT produce proofs
// =========================================================================

#[test]
fn test_no_proof_satisfiable() {
    let (result, _) = prove_tptp(r#"
        cnf(c1, axiom, p(a)).
        cnf(c2, axiom, q(b)).
    "#);
    assert!(
        !matches!(result, ProofResult::Proof { .. }),
        "satisfiable formula must not produce a proof"
    );
}

#[test]
fn test_no_proof_satisfiable_equality() {
    let (result, _) = prove_tptp(r#"
        cnf(c1, axiom, a = b).
        cnf(c2, axiom, p(a)).
    "#);
    assert!(
        !matches!(result, ProofResult::Proof { .. }),
        "satisfiable formula with equality must not produce a proof"
    );
}
