//! Test GRP001-1 specifically with Age-Weight selection

use proofatlas::{parse_tptp_file, SaturationConfig, SaturationState, SaturationResult, LiteralSelectionStrategy};
use proofatlas::selection::AgeWeightRatioSelector;

fn main() {
    let filename = "../.data/problems/tptp/TPTP-v9.0.0/Problems/GRP/GRP001-1.p";
    let include_dirs = vec!["../.data/problems/tptp/TPTP-v9.0.0/"];
    
    // Parse the TPTP file
    let cnf_formula = match parse_tptp_file(filename, &include_dirs) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            std::process::exit(1);
        }
    };
    
    println!("Parsed {} clauses:", cnf_formula.clauses.len());
    for (i, clause) in cnf_formula.clauses.iter().enumerate() {
        println!("  [{}] {}", i, clause);
    }
    
    let config = SaturationConfig {
        max_clauses: 10000,
        max_iterations: 100000,
        max_clause_size: 20,
        timeout: std::time::Duration::from_secs(60),
        use_superposition: true,  // Need superposition for equality reasoning
        literal_selection: LiteralSelectionStrategy::SelectAll,
    };
    
    println!("\nTesting with Age-Weight 1:5 clause selection:");
    println!("Superposition: {}", config.use_superposition);
    println!("Literal selection: SelectAll");
    
    let mut state = SaturationState::new(cnf_formula.clauses.clone(), config);
    state.set_clause_selector(Box::new(AgeWeightRatioSelector::new(1, 5)));
    
    let result = state.saturate();
    
    match result {
        SaturationResult::Proof(proof) => {
            println!("\n✓ THEOREM PROVED!");
            println!("Proof length: {} steps", proof.steps.len());
            println!("Empty clause at index: {}", proof.empty_clause_idx);
            
            // Show the last few steps
            println!("\nLast 10 proof steps:");
            let start = proof.steps.len().saturating_sub(10);
            for (i, step) in proof.steps[start..].iter().enumerate() {
                println!("  [{}] {:?} from {:?} => {}", 
                    start + i,
                    step.inference.rule,
                    step.inference.premises,
                    step.inference.conclusion
                );
            }
        }
        SaturationResult::Saturated => {
            println!("\n✗ SATURATED (no proof found)");
        }
        SaturationResult::ResourceLimit => {
            println!("\n✗ RESOURCE LIMIT");
        }
        SaturationResult::Timeout => {
            println!("\n✗ TIMEOUT");
        }
    }
}