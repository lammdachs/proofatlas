//! Generate a correct trace of saturation with proper clause indexing

use proofatlas::{
    parse_tptp_file,
    SaturationConfig,
    SaturationState,
    SaturationResult,
    LiteralSelectionStrategy,
    selection::AgeWeightRatioSelector,
};
use std::fs::File;
use std::io::Write;
use std::time::Duration;

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

    println!("=== Right Identity Problem - Correct Trace ===\n");
    println!("Initial clauses:");
    for (i, clause) in formula.clauses.iter().enumerate() {
        println!("[{}] {}", i, clause);
    }

    // Configure saturation with age-weight selection
    let config = SaturationConfig {
        max_clauses: 10000,
        max_iterations: 10000,
        max_clause_size: 100,
        timeout: Duration::from_secs(300),
        use_superposition: true,
        literal_selection: LiteralSelectionStrategy::SelectAll,
        step_limit: Some(100), // 100 steps
    };

    // Create saturation state with age-weight clause selector
    let mut state = SaturationState::new(formula.clauses.clone(), config);
    state.set_clause_selector(Box::new(AgeWeightRatioSelector::new(1, 5)));

    println!("\nRunning saturation with Age-Weight (1:5) selection for 100 steps...\n");

    // Run saturation
    match state.saturate() {
        SaturationResult::Proof(proof) => {
            println!("✓ THEOREM PROVED!");
            println!("Proof length: {} steps", proof.steps.len());
            println!("Empty clause at index: {}", proof.empty_clause_idx);
            
            // Build complete clause list
            let mut all_clauses = vec![];
            
            // Add initial clauses
            for (i, clause) in formula.clauses.iter().enumerate() {
                all_clauses.push((i, clause.clone(), None));
            }
            
            // Add derived clauses with their derivation info
            for step in proof.steps.iter() {
                all_clauses.push((
                    step.clause_idx,
                    step.inference.conclusion.clone(),
                    Some((step.inference.premises.clone(), step.inference.rule.clone()))
                ));
            }
            
            // Print all clauses
            println!("\n=== All clauses in proof ===");
            for (idx, clause, derivation) in all_clauses.iter() {
                println!("[{}] {}", idx, clause);
                if let Some((premises, rule)) = derivation {
                    println!("    from {:?} by {:?}", premises, rule);
                }
            }
        }
        SaturationResult::ResourceLimit(steps) => {
            println!("Reached step limit of 100");
            println!("Total clauses generated: {}", steps.len());
            
            // Build complete clause list
            let mut all_clauses = vec![];
            
            // Add initial clauses
            for (i, clause) in formula.clauses.iter().enumerate() {
                all_clauses.push((i, clause.clone(), None));
            }
            
            // Add derived clauses with their derivation info
            for step in steps.iter() {
                all_clauses.push((
                    step.clause_idx,
                    step.inference.conclusion.clone(),
                    Some((step.inference.premises.clone(), step.inference.rule.clone()))
                ));
            }
            
            // Print all clauses
            println!("\n=== All clauses after 100 steps ===");
            for (idx, clause, derivation) in all_clauses.iter() {
                println!("[{}] {}", idx, clause);
                if let Some((premises, rule)) = derivation {
                    println!("    from {:?} by {:?}", premises, rule);
                }
            }
            
            // Also show which clauses involve 33 or 94
            println!("\n=== Clauses involving indices 33 or 94 ===");
            for (idx, clause, derivation) in all_clauses.iter() {
                if let Some((premises, _)) = derivation {
                    if premises.contains(&33) || premises.contains(&94) {
                        println!("[{}] {} from {:?}", idx, clause, premises);
                    }
                }
            }
            
            // Count statistics
            let mut unit_equalities = 0;
            let mut negative_units = 0;
            for (_, clause, _) in all_clauses.iter() {
                if clause.literals.len() == 1 {
                    if !clause.literals[0].polarity {
                        negative_units += 1;
                    } else if clause.literals[0].atom.is_equality() {
                        unit_equalities += 1;
                    }
                }
            }
            
            println!("\n=== Statistics ===");
            println!("Total clauses: {}", all_clauses.len());
            println!("Unit equalities: {}", unit_equalities);
            println!("Negative unit clauses: {}", negative_units);
        }
        SaturationResult::Saturated => {
            println!("✗ SATURATED (no proof found)");
        }
        SaturationResult::Timeout => {
            println!("✗ TIMEOUT");
        }
    }
}