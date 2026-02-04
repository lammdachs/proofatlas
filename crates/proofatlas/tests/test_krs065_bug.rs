//! Test for KRS065+1 bug: saturation loop not generating inferences with duplicates

use proofatlas::{
    parse_tptp, saturate, AgeWeightSelector, ClauseSelector,
    SaturationConfig, SaturationResult, SaturationState,
};

#[test]
fn test_simple_resolution_proof() {
    // Simulating KRS065+1: cowlNothing(sk0) and ~cowlNothing(V0)
    let tptp = "
        cnf(c1, axiom, ~cowlNothing(V)).
        cnf(c2, axiom, cowlNothing(sk0)).
    ";
    
    let formula = parse_tptp(tptp, &[], None).expect("parse failed");
    println!("Parsed {} clauses", formula.clauses.len());
    for (i, c) in formula.clauses.iter().enumerate() {
        println!("  [{}] {}", i, c);
    }
    
    let config = SaturationConfig::default();
    let selector: Box<dyn ClauseSelector> = Box::new(AgeWeightSelector::new(0.5));
    
    let (result, _, _) = saturate(formula, config, selector);

    match &result {
        SaturationResult::Proof(proof) => {
            println!("PROOF FOUND! {} steps", proof.steps.len());
        }
        SaturationResult::Saturated(_, clauses) => {
            println!("SATURATED with {} clauses", clauses.len());
            for (i, c) in clauses.iter().enumerate() {
                println!("  [{}] {}", i, c);
            }
        }
        SaturationResult::ResourceLimit(_, clauses) | SaturationResult::Timeout(_, clauses) => {
            println!("RESOURCE LIMIT/TIMEOUT with {} clauses", clauses.len());
        }
    }

    assert!(matches!(result, SaturationResult::Proof(_)), "Should find proof!");
}

#[test]
fn test_with_duplicate_clause() {
    // The bug: adding a duplicate clause causes saturation to fail
    let tptp = "
        cnf(c1, axiom, ~cowlNothing(V)).
        cnf(c5, axiom, cowlNothing(sk0)).
        cnf(c7, axiom, cowlNothing(sk0)).
    ";

    let formula = parse_tptp(tptp, &[], None).expect("parse failed");
    println!("Parsed {} clauses", formula.clauses.len());
    for (i, c) in formula.clauses.iter().enumerate() {
        println!("  [{}] {}", i, c);
    }

    let config = SaturationConfig::default();
    let selector: Box<dyn ClauseSelector> = Box::new(AgeWeightSelector::new(0.5));

    let (result, _, _) = saturate(formula, config, selector);

    match &result {
        SaturationResult::Proof(proof) => {
            println!("PROOF FOUND! {} steps", proof.steps.len());
            for step in &proof.steps {
                println!("  [{:?}] {} <- {:?}", step.derivation, step.conclusion, step.derivation.premises);
            }
        }
        SaturationResult::Saturated(steps, clauses) => {
            println!("SATURATED with {} clauses, {} steps", clauses.len(), steps.len());
            for (i, c) in clauses.iter().enumerate() {
                println!("  [{}] {}", i, c);
            }
            println!("Steps:");
            for step in steps {
                println!("  [{:?}] {} <- {:?}", step.derivation, step.conclusion, step.derivation.premises);
            }
        }
        SaturationResult::ResourceLimit(_, clauses) | SaturationResult::Timeout(_, clauses) => {
            println!("RESOURCE LIMIT/TIMEOUT with {} clauses", clauses.len());
        }
    }

    assert!(matches!(result, SaturationResult::Proof(_)), "Should find proof even with duplicate!");
}

#[test]
fn test_with_precloned_clauses() {
    // Simulate what Python does: parse, set IDs, clone, then run saturation
    let tptp = "
        cnf(c1, axiom, ~cowlNothing(V)).
        cnf(c5, axiom, cowlNothing(sk0)).
        cnf(c7, axiom, cowlNothing(sk0)).
    ";
    
    let mut formula = parse_tptp(tptp, &[], None).expect("parse failed");
    
    // Assign IDs like Python's add_clauses_from_tptp does
    for (i, clause) in formula.clauses.iter_mut().enumerate() {
        clause.id = Some(i);
    }
    
    println!("After assigning IDs:");
    for (i, c) in formula.clauses.iter().enumerate() {
        println!("  [{}] {} (id={:?})", i, c, c.id);
    }
    
    // Clone like Python does
    let cloned_clauses: Vec<_> = formula.clauses.clone();
    println!("Cloned {} clauses", cloned_clauses.len());
    
    let config = SaturationConfig::default();
    let selector: Box<dyn ClauseSelector> = Box::new(AgeWeightSelector::new(0.5));
    
    let state = SaturationState::new(cloned_clauses, config, selector);
    let (result, _, _) = state.saturate();
    
    match &result {
        SaturationResult::Proof(proof) => {
            println!("PROOF FOUND! {} steps", proof.steps.len());
        }
        SaturationResult::Saturated(steps, clauses) => {
            println!("SATURATED with {} clauses, {} steps", clauses.len(), steps.len());
            for step in steps {
                println!("  [{:?}] idx={}", step.derivation, step.clause_idx);
            }
        }
        _ => {}
    }

    assert!(matches!(result, SaturationResult::Proof(_)), "Should find proof with pre-assigned IDs!");
}
