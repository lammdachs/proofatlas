//! Integration tests for the theorem prover

use proofatlas::{parse_tptp, saturate, AgeWeightSelector, ClauseSelector, SaturationConfig, SaturationResult};

fn create_selector() -> Box<dyn ClauseSelector> {
    Box::new(AgeWeightSelector::default())
}

#[test]
fn test_simple_resolution() {
    let tptp = r#"
        cnf(p_a, axiom, p(a)).
        cnf(p_implies_q, axiom, ~p(X) | q(X)).
        cnf(not_q_a, negated_conjecture, ~q(a)).
    "#;

    let formula = parse_tptp(tptp, &[], None).unwrap();
    let config = SaturationConfig::default();
    let (result, _, _) = saturate(formula, config, create_selector());

    match result {
        SaturationResult::Proof(_) => {
            // Expected - proof found
        }
        _ => panic!("Expected proof, got {:?}", result),
    }
}

#[test]
fn test_equality_reflexivity() {
    let tptp = r#"
        cnf(not_self_equal, negated_conjecture, a != a).
    "#;

    let formula = parse_tptp(tptp, &[], None).unwrap();
    let config = SaturationConfig::default();
    let (result, _, _) = saturate(formula, config, create_selector());

    match result {
        SaturationResult::Proof(_) => {
            // Expected - contradiction found
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

    let formula = parse_tptp(tptp, &[], None).unwrap();
    let mut config = SaturationConfig::default();
    config.max_clauses = 100; // Small limit to force saturation
    let (result, _, _) = saturate(formula, config, create_selector());

    match result {
        SaturationResult::Saturated(_, _) => {
            // Expected - no contradiction, formula is satisfiable
        }
        SaturationResult::Proof(_) => {
            panic!("Unexpected proof for satisfiable formula");
        }
        _ => {
            // Resource limit is also acceptable for this test
        }
    }
}
