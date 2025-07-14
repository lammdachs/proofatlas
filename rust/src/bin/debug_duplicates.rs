//! Debug why duplicate clauses aren't being detected

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
        step_limit: Some(30), // Just 30 steps to see duplicates
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
            
            // Find duplicates
            println!("=== Duplicate Clauses ===");
            for (clause_str, indices) in clause_map.iter() {
                if indices.len() > 1 {
                    println!("\n\"{}\" appears {} times:", clause_str, indices.len());
                    for &idx in indices {
                        // Find which step generated this
                        if idx < 4 {
                            println!("  [{}] - initial clause", idx);
                        } else {
                            for step in steps.iter() {
                                if step.clause_idx == idx {
                                    println!("  [{}] - from {:?} by {:?}", 
                                        idx, step.inference.premises, step.inference.rule);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            
            // Also check for variants (same clause with renamed variables)
            println!("\n=== Checking specific clause: mult(e,e) = e ===");
            for step in steps.iter() {
                let clause_str = format!("{}", step.inference.conclusion);
                if clause_str.contains("mult(e,e)") && clause_str.contains("= e") {
                    println!("[{}] {} from {:?}", 
                        step.clause_idx, clause_str, step.inference.premises);
                }
            }
        }
        _ => println!("Unexpected result"),
    }
}