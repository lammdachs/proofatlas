//! Test if improved subsumption reduces redundancy

use proofatlas::{
    parse_tptp_file,
    SaturationConfig,
    SaturationState,
    SaturationResult,
    LiteralSelectionStrategy,
};
use std::fs::File;
use std::io::Write;
use std::time::Duration;
use std::collections::HashMap;

fn main() {
    // Write right identity problem to temp file
    let tptp_content = r#"
cnf(right_identity, axiom, mult(e,X) = X).
cnf(right_inverse, axiom, mult(inv(X),X) = e).
cnf(associativity, axiom, mult(mult(X,Y),Z) = mult(X,mult(Y,Z))).
cnf(goal, negated_conjecture, mult(c,e) != c).
"#;
    
    let temp_filename = "/tmp/right_identity.p";
    let mut file = File::create(temp_filename).expect("Failed to create temp file");
    writeln!(file, "{}", tptp_content).expect("Failed to write temp file");

    // Parse the formula
    let formula = match parse_tptp_file(temp_filename, &[]) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            std::process::exit(1);
        }
    };

    // Configure saturation
    let config = SaturationConfig {
        max_clauses: 10000,
        max_iterations: 10000,
        max_clause_size: 100,
        timeout: Duration::from_secs(300),
        use_superposition: true,
        literal_selection: LiteralSelectionStrategy::SelectAll,
        step_limit: Some(30), // Just 30 steps
    };

    let state = SaturationState::new(formula.clauses.clone(), config);

    match state.saturate() {
        SaturationResult::ResourceLimit(steps) => {
            // Track clause strings and their indices
            let mut clause_map: HashMap<String, Vec<usize>> = HashMap::new();
            
            // Add initial clauses
            for (i, clause) in formula.clauses.iter().enumerate() {
                let clause_str = format!("{}", clause);
                clause_map.entry(clause_str).or_insert(vec![]).push(i);
            }
            
            // Add derived clauses
            for step in steps.iter() {
                let clause_str = format!("{}", step.inference.conclusion);
                clause_map.entry(clause_str).or_insert(vec![]).push(step.clause_idx);
            }
            
            // Count duplicates
            let mut duplicate_count = 0;
            let mut duplicate_instances = 0;
            
            println!("=== Duplicate Analysis with Improved Subsumption ===");
            for (clause_str, indices) in clause_map.iter() {
                if indices.len() > 1 {
                    duplicate_count += 1;
                    duplicate_instances += indices.len() - 1;
                    println!("\"{}\" appears {} times: {:?}", clause_str, indices.len(), indices);
                }
            }
            
            println!("\nSummary:");
            println!("Total clauses generated: {}", 4 + steps.len());
            println!("Unique clause strings: {}", clause_map.len());
            println!("Clauses with duplicates: {}", duplicate_count);
            println!("Total duplicate instances: {}", duplicate_instances);
            
            // Also check for specific problematic clause
            println!("\n=== Checking mult(e,e) = e ===");
            let mut mult_ee_count = 0;
            for step in steps.iter() {
                let clause_str = format!("{}", step.inference.conclusion);
                if clause_str == "mult(e,e) = e" {
                    mult_ee_count += 1;
                    println!("[{}] from {:?}", step.clause_idx, step.inference.premises);
                }
            }
            println!("Total occurrences of 'mult(e,e) = e': {}", mult_ee_count);
        }
        SaturationResult::Proof(proof) => {
            println!("Found proof in {} steps!", proof.steps.len());
        }
        _ => println!("Unexpected result"),
    }
}