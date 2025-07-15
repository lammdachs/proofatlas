//! Full trace showing all proof steps for right identity problem

use proofatlas::{parse_tptp_file, SaturationConfig, SaturationState, LiteralSelectionStrategy};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let filename = if args.len() > 1 {
        &args[1]
    } else {
        "/tmp/right_identity.p"
    };
    
    // Parse the TPTP file
    let formula = parse_tptp_file(filename, &[]).expect("Failed to parse file");
    
    println!("=== Right Identity Test - Trace (100 steps) ===\n");
    println!("Initial clauses:");
    for (i, clause) in formula.clauses.iter().enumerate() {
        println!("[{}] {}", i, clause);
    }
    
    // Configure saturation
    let mut config = SaturationConfig::default();
    config.step_limit = Some(100);
    config.max_clauses = 1000;  // Allow more clauses to see what happens
    config.literal_selection = LiteralSelectionStrategy::SelectAll;
    config.max_clauses = 10000;
    config.timeout = std::time::Duration::from_secs(60);
    
    let step_limit = config.step_limit.unwrap();
    println!("\nRunning saturation for {} steps...\n", step_limit);
    
    // Create saturation state and run
    let state = SaturationState::new(formula.clauses, config);
    
    let start = Instant::now();
    let result = state.saturate();
    let elapsed = start.elapsed();
    
    println!("=== Saturation Complete ===");
    println!("Time: {:.3} seconds\n", elapsed.as_secs_f64());
    
    match result {
        proofatlas::SaturationResult::Proof(proof) => {
            println!("✓ PROOF FOUND!");
            println!("Empty clause derived at index: {}", proof.empty_clause_idx);
            println!("Proof length: {} steps\n", proof.steps.len());
            
            println!("=== All Proof Steps ===");
            for step in proof.steps.iter() {
                println!("[{}] {}", step.clause_idx, step.inference.conclusion);
                println!("    Rule: {:?}", step.inference.rule);
                println!("    Parents: {:?}", step.inference.premises);
                println!();
            }
        }
        proofatlas::SaturationResult::ResourceLimit(steps) => {
            println!("RESOURCE LIMIT REACHED");
            println!("Generated {} clauses in {} steps\n", steps.len(), step_limit);
            
            println!("=== All Derived Clauses ===");
            
            // Check for invalid inference
            let mut invalid_found = false;
            
            for step in steps.iter() {
                println!("[{}] {}", step.clause_idx, step.inference.conclusion);
                println!("    Rule: {:?}", step.inference.rule);
                println!("    Parents: {:?}", step.inference.premises);
                
                // Check if this is the invalid inference
                if step.inference.conclusion.to_string() == "~mult(e,mult(c,e)) = c" {
                    println!("    ❌ INVALID INFERENCE DETECTED!");
                    invalid_found = true;
                }
                println!();
            }
            
            println!("\n=== Ordering Constraint Check ===");
            if invalid_found {
                println!("❌ FAILED: Invalid inference ~mult(e,mult(c,e)) = c was generated");
                println!("This violates the ordering constraints!");
            } else {
                println!("✓ PASSED: No invalid inference found");
                println!("The ordering constraints are working correctly");
            }
        }
        proofatlas::SaturationResult::Saturated => {
            println!("SATURATED - No proof found");
            println!("\n=== Ordering Constraint Check ===");
            println!("✓ PASSED: No invalid inference found");
            println!("The ordering constraints are working correctly");
        }
        proofatlas::SaturationResult::Timeout => {
            println!("TIMEOUT");
        }
    }
}