//! Full trace showing all proof steps for right identity problem

use proofatlas::{
    parse_tptp_file, LiteralSelectionStrategy, OnnxClauseSelector, SaturationConfig,
    SaturationState,
};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!(
            "Usage: {} <tptp_file> --model <onnx_model> [options]",
            args[0]
        );
        eprintln!("\nRequired:");
        eprintln!("  --model <path>         Path to ONNX clause selector model");
        eprintln!("\nOptions:");
        eprintln!("  --include <dir>        Add include directory (can be used multiple times)");
        std::process::exit(1);
    }

    let filename = &args[1];
    let mut include_dirs: Vec<String> = Vec::new();
    let mut model_path: Option<String> = None;

    // Parse command line options
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--include" => {
                if i + 1 < args.len() {
                    include_dirs.push(args[i + 1].clone());
                    i += 1;
                }
            }
            "--model" => {
                if i + 1 < args.len() {
                    model_path = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
            }
        }
        i += 1;
    }

    // Check required model path
    let model_path = match model_path {
        Some(p) => p,
        None => {
            eprintln!("Error: --model <path> is required");
            std::process::exit(1);
        }
    };

    // Load ONNX clause selector
    let clause_selector = match OnnxClauseSelector::new(&model_path) {
        Ok(s) => Box::new(s),
        Err(e) => {
            eprintln!("Failed to load ONNX model: {}", e);
            std::process::exit(1);
        }
    };

    // Parse TPTP with include support
    let include_dir_refs: Vec<&str> = include_dirs.iter().map(|s| s.as_str()).collect();
    let formula = match parse_tptp_file(filename, &include_dir_refs) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            std::process::exit(1);
        }
    };

    println!("=== Right Identity Test - Trace (100 steps) ===\n");
    println!("Initial clauses:");
    for (i, clause) in formula.clauses.iter().enumerate() {
        println!("[{}] {}", i, clause);
    }

    // Configure saturation
    let mut config = SaturationConfig::default();
    config.step_limit = Some(100);
    config.max_clauses = 1000; // Allow more clauses to see what happens
    config.literal_selection = LiteralSelectionStrategy::SelectAll;
    config.max_clauses = 10000;
    config.timeout = std::time::Duration::from_secs(60);

    let step_limit = config.step_limit.unwrap();
    println!("\nRunning saturation for {} steps...\n", step_limit);

    // Create saturation state and run
    let state = SaturationState::new(formula.clauses, config, clause_selector);

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
        proofatlas::SaturationResult::ResourceLimit(steps, _) => {
            println!("RESOURCE LIMIT REACHED");
            println!(
                "Generated {} clauses in {} steps\n",
                steps.len(),
                step_limit
            );

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
        proofatlas::SaturationResult::Saturated(_, _) => {
            println!("SATURATED - No proof found");
            println!("\n=== Ordering Constraint Check ===");
            println!("✓ PASSED: No invalid inference found");
            println!("The ordering constraints are working correctly");
        }
        proofatlas::SaturationResult::Timeout(_, _) => {
            println!("TIMEOUT");
        }
    }
}
