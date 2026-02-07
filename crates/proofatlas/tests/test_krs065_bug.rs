//! Test for KRS065+1 bug: saturation loop not generating inferences with duplicates

use proofatlas::{
    parse_tptp, saturate, AgeWeightSelector, ClauseSelector,
    ProverConfig, ProofResult, ProofAtlas,
};

#[test]
fn test_simple_resolution_proof() {
    // Simulating KRS065+1: cowlNothing(sk0) and ~cowlNothing(V0)
    let tptp = "
        cnf(c1, axiom, ~cowlNothing(V)).
        cnf(c2, axiom, cowlNothing(sk0)).
    ";

    let parsed = parse_tptp(tptp, &[], None).expect("parse failed");
    println!("Parsed {} clauses", parsed.formula.clauses.len());
    for (i, c) in parsed.formula.clauses.iter().enumerate() {
        println!("  [{}] {}", i, c);
    }

    let config = ProverConfig::default();
    let selector: Box<dyn ClauseSelector> = Box::new(AgeWeightSelector::new(0.5));

    let (result, _, _, _) = saturate(parsed.formula, config, selector, parsed.interner);

    match &result {
        ProofResult::Proof(proof) => {
            println!("PROOF FOUND! {} steps", proof.steps.len());
        }
        ProofResult::Saturated(_, clauses) => {
            println!("SATURATED with {} clauses", clauses.len());
            for (i, c) in clauses.iter().enumerate() {
                println!("  [{}] {}", i, c);
            }
        }
        ProofResult::ResourceLimit(_, clauses) | ProofResult::Timeout(_, clauses) => {
            println!("RESOURCE LIMIT/TIMEOUT with {} clauses", clauses.len());
        }
    }

    assert!(matches!(result, ProofResult::Proof(_)), "Should find proof!");
}

#[test]
fn test_with_duplicate_clause() {
    // The bug: adding a duplicate clause causes saturation to fail
    let tptp = "
        cnf(c1, axiom, ~cowlNothing(V)).
        cnf(c5, axiom, cowlNothing(sk0)).
        cnf(c7, axiom, cowlNothing(sk0)).
    ";

    let parsed = parse_tptp(tptp, &[], None).expect("parse failed");
    println!("Parsed {} clauses", parsed.formula.clauses.len());
    for (i, c) in parsed.formula.clauses.iter().enumerate() {
        println!("  [{}] {}", i, c);
    }

    let config = ProverConfig::default();
    let selector: Box<dyn ClauseSelector> = Box::new(AgeWeightSelector::new(0.5));

    let (result, _, _, _) = saturate(parsed.formula, config, selector, parsed.interner);

    match &result {
        ProofResult::Proof(proof) => {
            println!("PROOF FOUND! {} steps", proof.steps.len());
            for step in &proof.steps {
                println!("  [{:?}] {} <- {:?}", step.rule_name, step.conclusion, proofatlas::clause_indices(&step.premises));
            }
        }
        ProofResult::Saturated(steps, clauses) => {
            println!("SATURATED with {} clauses, {} steps", clauses.len(), steps.len());
            for (i, c) in clauses.iter().enumerate() {
                println!("  [{}] {}", i, c);
            }
            println!("Steps:");
            for step in steps {
                println!("  [{:?}] {} <- {:?}", step.rule_name, step.conclusion, proofatlas::clause_indices(&step.premises));
            }
        }
        ProofResult::ResourceLimit(_, clauses) | ProofResult::Timeout(_, clauses) => {
            println!("RESOURCE LIMIT/TIMEOUT with {} clauses", clauses.len());
        }
    }

    assert!(matches!(result, ProofResult::Proof(_)), "Should find proof even with duplicate!");
}

#[test]
fn test_with_precloned_clauses() {
    // Simulate what Python does: parse, set IDs, clone, then run saturation
    let tptp = "
        cnf(c1, axiom, ~cowlNothing(V)).
        cnf(c5, axiom, cowlNothing(sk0)).
        cnf(c7, axiom, cowlNothing(sk0)).
    ";

    let mut parsed = parse_tptp(tptp, &[], None).expect("parse failed");

    // Assign IDs like Python's add_clauses_from_tptp does
    for (i, clause) in parsed.formula.clauses.iter_mut().enumerate() {
        clause.id = Some(i);
    }

    println!("After assigning IDs:");
    for (i, c) in parsed.formula.clauses.iter().enumerate() {
        println!("  [{}] {} (id={:?})", i, c, c.id);
    }

    // Clone like Python does
    let cloned_clauses: Vec<_> = parsed.formula.clauses.clone();
    println!("Cloned {} clauses", cloned_clauses.len());

    let config = ProverConfig::default();
    let selector: Box<dyn ClauseSelector> = Box::new(AgeWeightSelector::new(0.5));

    let prover = ProofAtlas::new(cloned_clauses, config, selector, parsed.interner);
    let (result, _, _, _) = prover.prove();

    match &result {
        ProofResult::Proof(proof) => {
            println!("PROOF FOUND! {} steps", proof.steps.len());
        }
        ProofResult::Saturated(steps, clauses) => {
            println!("SATURATED with {} clauses, {} steps", clauses.len(), steps.len());
            for step in steps {
                println!("  [{}] idx={}", step.rule_name, step.clause_idx);
            }
        }
        _ => {}
    }

    assert!(matches!(result, ProofResult::Proof(_)), "Should find proof with pre-assigned IDs!");
}
