//! Integration tests for the theorem prover

use proofatlas::{parse_tptp, saturate, AgeWeightSink, ProverSink, ProverConfig, ProofResult};

fn create_sink() -> Box<dyn ProverSink> {
    Box::new(AgeWeightSink::new(0.5))
}

#[test]
fn test_simple_resolution() {
    let tptp = r#"
        cnf(p_a, axiom, p(a)).
        cnf(p_implies_q, axiom, ~p(X) | q(X)).
        cnf(not_q_a, negated_conjecture, ~q(a)).
    "#;

    let parsed = parse_tptp(tptp, &[], None, None).unwrap();
    let config = ProverConfig::default();
    let (result, prover) = saturate(parsed.formula, config, create_sink(), parsed.interner);

    match result {
        ProofResult::Proof { empty_clause_idx } => {
            prover.verify_proof(empty_clause_idx)
                .expect("proof verification failed");
        }
        _ => panic!("Expected proof, got {:?}", result),
    }
}

#[test]
fn test_equality_reflexivity() {
    let tptp = r#"
        cnf(not_self_equal, negated_conjecture, a != a).
    "#;

    let parsed = parse_tptp(tptp, &[], None, None).unwrap();
    let config = ProverConfig::default();
    let (result, prover) = saturate(parsed.formula, config, create_sink(), parsed.interner);

    match result {
        ProofResult::Proof { empty_clause_idx } => {
            prover.verify_proof(empty_clause_idx)
                .expect("proof verification failed");
        }
        _ => panic!("Expected proof, got {:?}", result),
    }
}

#[test]
fn test_satisfiable_formula() {
    let tptp = r#"
        cnf(p_a, axiom, p(a)).
        cnf(q_b, axiom, q(b)).
    "#;

    let parsed = parse_tptp(tptp, &[], None, None).unwrap();
    let mut config = ProverConfig::default();
    config.max_clauses = 100; // Small limit to force saturation
    let (result, _) = saturate(parsed.formula, config, create_sink(), parsed.interner);

    match result {
        ProofResult::Saturated => {
            // Expected - no contradiction, formula is satisfiable
        }
        ProofResult::Proof { .. } => {
            panic!("Unexpected proof for satisfiable formula");
        }
        _ => {
            // Resource limit is also acceptable for this test
        }
    }
}
