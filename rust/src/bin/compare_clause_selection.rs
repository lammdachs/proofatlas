//! Compare performance of different clause selection strategies

use std::env;
use std::time::Instant;
use proofatlas::{parse_tptp_file, SaturationConfig, SaturationState, SaturationResult, LiteralSelectionStrategy};
use proofatlas::selection::{SizeBasedSelector, AgeBasedSelector, AgeWeightRatioSelector};

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Usage: {} <tptp_file> [--include <dir>]", args[0]);
        std::process::exit(1);
    }
    
    let filename = &args[1];
    let mut include_dirs: Vec<&str> = Vec::new();
    
    // Parse include directories
    let mut i = 2;
    while i < args.len() {
        if args[i] == "--include" && i + 1 < args.len() {
            include_dirs.push(&args[i + 1]);
            i += 2;
        } else {
            i += 1;
        }
    }
    
    // Parse the TPTP file
    let cnf_formula = match parse_tptp_file(filename, &include_dirs) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            std::process::exit(1);
        }
    };
    
    println!("Parsed {} clauses from '{}'", cnf_formula.clauses.len(), filename);
    println!("\n========================================");
    
    // Test different clause selection strategies
    let strategies: Vec<(&str, Box<dyn Fn() -> Box<dyn proofatlas::selection::ClauseSelector>>)> = vec![
        ("Age-Based (FIFO)", Box::new(|| Box::new(AgeBasedSelector))),
        ("Size-Based", Box::new(|| Box::new(SizeBasedSelector))),
        ("Age-Weight 1:5", Box::new(|| Box::new(AgeWeightRatioSelector::new(1, 5)))),
        ("Age-Weight 1:1", Box::new(|| Box::new(AgeWeightRatioSelector::new(1, 1)))),
        ("Age-Weight 1:10", Box::new(|| Box::new(AgeWeightRatioSelector::new(1, 10)))),
    ];
    
    // Use SelectAll as the literal selection strategy
    let config = SaturationConfig {
        max_clauses: 10000,
        max_iterations: 100000,
        max_clause_size: 20,
        timeout: std::time::Duration::from_secs(10),
        use_superposition: true,
        literal_selection: LiteralSelectionStrategy::SelectAll,
        step_limit: None,
    };
    
    for (name, selector_factory) in strategies {
        println!("\nTesting with {}:", name);
        println!("----------------------------------------");
        
        // Create a fresh state with the selected clause selector
        let mut state = SaturationState::new(cnf_formula.clauses.clone(), config.clone());
        state.set_clause_selector(selector_factory());
        
        let start = Instant::now();
        let result = state.saturate();
        let elapsed = start.elapsed();
        
        match result {
            SaturationResult::Proof(proof) => {
                println!("✓ THEOREM PROVED in {:.3}s", elapsed.as_secs_f64());
                println!("  Proof length: {} steps", proof.steps.len());
                println!("  Empty clause at index: {}", proof.empty_clause_idx);
            }
            SaturationResult::Saturated => {
                println!("✗ SATURATED in {:.3}s (no proof found)", elapsed.as_secs_f64());
            }
            SaturationResult::ResourceLimit(_) => {
                println!("✗ RESOURCE LIMIT in {:.3}s", elapsed.as_secs_f64());
                println!("  Exceeded clause limit or iteration limit");
            }
            SaturationResult::Timeout => {
                println!("✗ TIMEOUT after {:.3}s", elapsed.as_secs_f64());
            }
        }
    }
    
    println!("\n========================================");
}