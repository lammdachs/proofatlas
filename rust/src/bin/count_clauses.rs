//! Count clauses generated over time

use proofatlas::{
    parse_tptp_file, AgeWeightSelector, ClauseSelector, LiteralSelectionStrategy, SaturationConfig,
    SaturationResult, SaturationState,
};
use std::env;
use std::fs::File;
use std::io::Write;
use std::time::Duration;

fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse age-weight ratio from args (optional)
    let age_weight_ratio: f64 = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.5);

    println!(
        "Using AgeWeightSelector with ratio {}",
        age_weight_ratio
    );

    // Create selector factory
    let create_selector: Box<dyn Fn() -> Box<dyn ClauseSelector>> =
        Box::new(move || Box::new(AgeWeightSelector::new(age_weight_ratio)) as Box<dyn ClauseSelector>);

    // Write right identity problem to temp file
    let tptp_content = r#"
cnf(right_identity, axiom, mult(e,X) = X).
cnf(right_inverse, axiom, mult(inv(X),X) = e).
cnf(associativity, axiom, mult(mult(X,Y),Z) = mult(X,mult(Y,Z))).
cnf(goal, negated_conjecture, mult(c,e) != c).
"#;

    let temp_filename = "right_identity_test.p";
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

    println!("Testing clause generation...\n");

    // Test different step limits
    for &steps in &[10, 20, 50, 100, 200] {
        let selector = create_selector();

        let config = SaturationConfig {
            max_clauses: 10000,
            max_iterations: 10000,
            max_clause_size: 100,
            timeout: Duration::from_secs(300),
            literal_selection: LiteralSelectionStrategy::SelectAll,
            step_limit: Some(steps),
        };

        let state = SaturationState::new(formula.clauses.clone(), config, selector);

        match state.saturate() {
            SaturationResult::ResourceLimit(proof_steps, _) => {
                println!(
                    "After {} steps: {} clauses generated",
                    steps,
                    proof_steps.len()
                );
            }
            SaturationResult::Proof(proof) => {
                println!("PROOF FOUND after {} steps!", proof.steps.len());
                println!("Empty clause at index: {}", proof.empty_clause_idx);

                // Clean up temp file
                let _ = std::fs::remove_file(temp_filename);
                return;
            }
            _ => {}
        }
    }

    // Clean up temp file
    let _ = std::fs::remove_file(temp_filename);
}
