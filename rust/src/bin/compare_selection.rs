//! Compare performance of different literal selection strategies

use proofatlas::{
    parse_tptp_file, saturate, LiteralSelectionStrategy, SaturationConfig, SaturationResult,
};
use std::env;
use std::time::Instant;

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

    println!(
        "Parsed {} clauses from '{}'",
        cnf_formula.clauses.len(),
        filename
    );
    println!("\n========================================");

    // Test different selection strategies
    let strategies = vec![
        ("Select All", LiteralSelectionStrategy::SelectAll),
        (
            "Select Max Weight",
            LiteralSelectionStrategy::SelectMaxWeight,
        ),
        (
            "Select Largest Negative",
            LiteralSelectionStrategy::SelectLargestNegative,
        ),
    ];

    for (name, strategy) in strategies {
        println!("\nTesting with {}:", name);
        println!("----------------------------------------");

        let mut config = SaturationConfig::default();
        config.literal_selection = strategy;
        config.timeout = std::time::Duration::from_secs(10); // Shorter timeout for comparison

        let start = Instant::now();
        let result = saturate(cnf_formula.clone(), config);
        let elapsed = start.elapsed();

        match result {
            SaturationResult::Proof(proof) => {
                println!("✓ THEOREM PROVED in {:.3}s", elapsed.as_secs_f64());
                println!("  Proof length: {} steps", proof.steps.len());
                println!("  Empty clause at index: {}", proof.empty_clause_idx);
            }
            SaturationResult::Saturated(_, _) => {
                println!(
                    "✗ SATURATED in {:.3}s (no proof found)",
                    elapsed.as_secs_f64()
                );
            }
            SaturationResult::ResourceLimit(_, _) => {
                println!("✗ RESOURCE LIMIT in {:.3}s", elapsed.as_secs_f64());
            }
            SaturationResult::Timeout(_, _) => {
                println!("✗ TIMEOUT after {:.3}s", elapsed.as_secs_f64());
            }
        }
    }

    println!("\n========================================");
}
