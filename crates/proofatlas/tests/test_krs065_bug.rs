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

    let parsed = parse_tptp(tptp, &[], None, None).expect("parse failed");
    println!("Parsed {} clauses", parsed.formula.clauses.len());
    for (i, c) in parsed.formula.clauses.iter().enumerate() {
        println!("  [{}] {}", i, c);
    }

    let config = ProverConfig::default();
    let selector: Box<dyn ClauseSelector> = Box::new(AgeWeightSelector::new(0.5));

    let (result, prover) = saturate(parsed.formula, config, selector, parsed.interner);

    match &result {
        ProofResult::Proof { empty_clause_idx } => {
            let steps = prover.extract_proof(*empty_clause_idx);
            println!("PROOF FOUND! {} steps", steps.len());
        }
        ProofResult::Saturated => {
            let clauses = prover.clauses();
            println!("SATURATED with {} clauses", clauses.len());
            for (i, c) in clauses.iter().enumerate() {
                println!("  [{}] {}", i, c);
            }
        }
        ProofResult::ResourceLimit => {
            println!("RESOURCE LIMIT with {} clauses", prover.clauses().len());
        }
    }

    assert!(matches!(result, ProofResult::Proof { .. }), "Should find proof!");
}

#[test]
fn test_with_duplicate_clause() {
    // The bug: adding a duplicate clause causes saturation to fail
    let tptp = "
        cnf(c1, axiom, ~cowlNothing(V)).
        cnf(c5, axiom, cowlNothing(sk0)).
        cnf(c7, axiom, cowlNothing(sk0)).
    ";

    let parsed = parse_tptp(tptp, &[], None, None).expect("parse failed");
    println!("Parsed {} clauses", parsed.formula.clauses.len());
    for (i, c) in parsed.formula.clauses.iter().enumerate() {
        println!("  [{}] {}", i, c);
    }

    let config = ProverConfig::default();
    let selector: Box<dyn ClauseSelector> = Box::new(AgeWeightSelector::new(0.5));

    let (result, prover) = saturate(parsed.formula, config, selector, parsed.interner);

    match &result {
        ProofResult::Proof { empty_clause_idx } => {
            let steps = prover.extract_proof(*empty_clause_idx);
            println!("PROOF FOUND! {} steps", steps.len());
            for step in &steps {
                println!("  [{:?}] {} <- {:?}", step.rule_name, step.conclusion, proofatlas::clause_indices(&step.premises));
            }
        }
        ProofResult::Saturated => {
            println!("SATURATED with {} clauses", prover.clauses().len());
            for (i, c) in prover.clauses().iter().enumerate() {
                println!("  [{}] {}", i, c);
            }
        }
        ProofResult::ResourceLimit => {
            println!("RESOURCE LIMIT with {} clauses", prover.clauses().len());
        }
    }

    assert!(matches!(result, ProofResult::Proof { .. }), "Should find proof even with duplicate!");
}

#[test]
fn test_with_precloned_clauses() {
    // Simulate what Python does: parse, set IDs, clone, then run saturation
    let tptp = "
        cnf(c1, axiom, ~cowlNothing(V)).
        cnf(c5, axiom, cowlNothing(sk0)).
        cnf(c7, axiom, cowlNothing(sk0)).
    ";

    let mut parsed = parse_tptp(tptp, &[], None, None).expect("parse failed");

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

    let mut prover = ProofAtlas::new(cloned_clauses, config, selector, parsed.interner);
    let result = prover.prove();

    match &result {
        ProofResult::Proof { .. } => {
            println!("PROOF FOUND!");
        }
        ProofResult::Saturated => {
            println!("SATURATED with {} clauses", prover.clauses().len());
        }
        _ => {}
    }

    assert!(matches!(result, ProofResult::Proof { .. }), "Should find proof with pre-assigned IDs!");
}
