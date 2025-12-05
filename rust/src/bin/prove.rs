//! Command-line theorem prover

use std::env;
use std::time::Instant;

use proofatlas::{
    parse_tptp_file, saturate, AgeWeightSelector, ClauseSelector, LiteralSelectionStrategy,
    SaturationConfig, SaturationResult,
};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <tptp_file> [options]", args[0]);
        eprintln!("\nOptions:");
        eprintln!("  --timeout <seconds>    Set timeout (default: 60)");
        eprintln!("  --max-clauses <n>      Set max clauses (default: 10000)");
        eprintln!("  --literal-selection <n>  Literal selection strategy (default: 0):");
        eprintln!("                           0  = select all literals");
        eprintln!("                           20 = select all maximal literals");
        eprintln!("                           21 = unique maximal, else max-weight negative, else all maximal");
        eprintln!("                           22 = max-weight negative, else all maximal");
        eprintln!("  --include <dir>        Add include directory (can be used multiple times)");
        eprintln!("  --age-weight <ratio>   Age probability for age-weight selector (default: 0.5)");
        eprintln!("  --verbose              Show detailed progress");
        std::process::exit(1);
    }

    let filename = &args[1];
    let mut config = SaturationConfig::default();
    let mut verbose = false;
    let mut include_dirs: Vec<String> = Vec::new();
    let mut age_weight_ratio: f64 = 0.5;

    // Parse command line options
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--timeout" => {
                if i + 1 < args.len() {
                    if let Ok(secs) = args[i + 1].parse::<u64>() {
                        config.timeout = std::time::Duration::from_secs(secs);
                    }
                    i += 1;
                }
            }
            "--max-clauses" => {
                if i + 1 < args.len() {
                    if let Ok(n) = args[i + 1].parse::<usize>() {
                        config.max_clauses = n;
                    }
                    i += 1;
                }
            }
            "--literal-selection" => {
                if i + 1 < args.len() {
                    config.literal_selection = match args[i + 1].as_str() {
                        "0" => LiteralSelectionStrategy::Sel0,
                        "20" => LiteralSelectionStrategy::Sel20,
                        "21" => LiteralSelectionStrategy::Sel21,
                        "22" => LiteralSelectionStrategy::Sel22,
                        _ => {
                            eprintln!("Unknown literal selection strategy: {}", args[i + 1]);
                            eprintln!("Valid options: 0, 20, 21, 22");
                            std::process::exit(1);
                        }
                    };
                    i += 1;
                }
            }
            "--include" => {
                if i + 1 < args.len() {
                    include_dirs.push(args[i + 1].clone());
                    i += 1;
                }
            }
            "--verbose" => {
                verbose = true;
            }
            "--age-weight" => {
                if i + 1 < args.len() {
                    if let Ok(ratio) = args[i + 1].parse::<f64>() {
                        age_weight_ratio = ratio;
                    }
                    i += 1;
                }
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
            }
        }
        i += 1;
    }

    // Create clause selector (age-weight heuristic)
    let clause_selector: Box<dyn ClauseSelector> = Box::new(AgeWeightSelector::new(age_weight_ratio));

    // Parse TPTP with include support
    let include_dir_refs: Vec<&str> = include_dirs.iter().map(|s| s.as_str()).collect();
    let formula = match parse_tptp_file(filename, &include_dir_refs) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            std::process::exit(1);
        }
    };

    println!(
        "Parsed {} clauses from '{}'",
        formula.clauses.len(),
        filename
    );

    if verbose {
        println!("\nInput clauses:");
        for (i, clause) in formula.clauses.iter().enumerate() {
            println!("  [{}] {}", i, clause);
        }
        println!();
    }

    // Run saturation
    println!("Running saturation with:");
    println!("  Max clauses: {}", config.max_clauses);
    println!("  Timeout: {:?}", config.timeout);
    println!();

    let start_time = Instant::now();
    let result = saturate(formula, config, clause_selector);
    let elapsed = start_time.elapsed();

    // Report result
    match result {
        SaturationResult::Proof(proof) => {
            println!("✓ THEOREM PROVED in {:.3}s", elapsed.as_secs_f64());
            println!("  Proof length: {} steps", proof.steps.len());
            println!(
                "  Empty clause derived at index: {}",
                proof.empty_clause_idx
            );

            if verbose {
                println!("\nProof steps:");
                for step in proof.steps.iter() {
                    println!(
                        "  [{}] {} from {:?} => {}",
                        step.clause_idx,
                        match step.inference.rule {
                            proofatlas::InferenceRule::Input => "Input",
                            proofatlas::InferenceRule::GivenClauseSelection =>
                                "Given Clause Selected",
                            proofatlas::InferenceRule::Resolution => "Resolution",
                            proofatlas::InferenceRule::Factoring => "Factoring",
                            proofatlas::InferenceRule::Superposition => "Superposition",
                            proofatlas::InferenceRule::EqualityResolution => "Equality Resolution",
                            proofatlas::InferenceRule::EqualityFactoring => "Equality Factoring",
                            proofatlas::InferenceRule::Demodulation => "Demodulation",
                        },
                        step.inference.premises,
                        step.inference.conclusion
                    );
                }
            }
        }
        SaturationResult::Saturated(proof_steps, clauses) => {
            println!("✗ SATURATED in {:.3}s", elapsed.as_secs_f64());
            println!("  No proof found - the formula may be satisfiable");
            println!("  Final clauses: {}", clauses.len());
            
            if verbose {
                println!("\nProof steps (inference trace):");
                for step in proof_steps.iter() {
                    if !matches!(step.inference.rule, proofatlas::InferenceRule::Input) &&
                       !matches!(step.inference.rule, proofatlas::InferenceRule::GivenClauseSelection) {
                        println!(
                            "  [{}] {} from {:?} => {}",
                            step.clause_idx,
                            match step.inference.rule {
                                proofatlas::InferenceRule::Resolution => "Resolution",
                                proofatlas::InferenceRule::Factoring => "Factoring",
                                proofatlas::InferenceRule::Superposition => "Superposition",
                                proofatlas::InferenceRule::EqualityResolution => "Equality Resolution",
                                proofatlas::InferenceRule::EqualityFactoring => "Equality Factoring",
                                proofatlas::InferenceRule::Demodulation => "Demodulation",
                                _ => "Other",
                            },
                            step.inference.premises,
                            step.inference.conclusion
                        );
                    }
                }
            }
        }
        SaturationResult::ResourceLimit(proof_steps, clauses) => {
            println!("✗ RESOURCE LIMIT in {:.3}s", elapsed.as_secs_f64());
            println!("  Exceeded clause limit or iteration limit");
            println!("  Final clauses: {}", clauses.len());
            
            if verbose {
                println!("\nProof steps (inference trace):");
                for step in proof_steps.iter() {
                    if !matches!(step.inference.rule, proofatlas::InferenceRule::Input) &&
                       !matches!(step.inference.rule, proofatlas::InferenceRule::GivenClauseSelection) {
                        println!(
                            "  [{}] {} from {:?} => {}",
                            step.clause_idx,
                            match step.inference.rule {
                                proofatlas::InferenceRule::Resolution => "Resolution",
                                proofatlas::InferenceRule::Factoring => "Factoring",
                                proofatlas::InferenceRule::Superposition => "Superposition",
                                proofatlas::InferenceRule::EqualityResolution => "Equality Resolution",
                                proofatlas::InferenceRule::EqualityFactoring => "Equality Factoring",
                                proofatlas::InferenceRule::Demodulation => "Demodulation",
                                _ => "Other",
                            },
                            step.inference.premises,
                            step.inference.conclusion
                        );
                    }
                }
            }
        }
        SaturationResult::Timeout(proof_steps, clauses) => {
            println!("✗ TIMEOUT in {:.3}s", elapsed.as_secs_f64());
            println!("  Exceeded time limit");
            println!("  Final clauses: {}", clauses.len());
            
            if verbose {
                println!("\nProof steps (inference trace):");
                for step in proof_steps.iter() {
                    if !matches!(step.inference.rule, proofatlas::InferenceRule::Input) &&
                       !matches!(step.inference.rule, proofatlas::InferenceRule::GivenClauseSelection) {
                        println!(
                            "  [{}] {} from {:?} => {}",
                            step.clause_idx,
                            match step.inference.rule {
                                proofatlas::InferenceRule::Resolution => "Resolution",
                                proofatlas::InferenceRule::Factoring => "Factoring",
                                proofatlas::InferenceRule::Superposition => "Superposition",
                                proofatlas::InferenceRule::EqualityResolution => "Equality Resolution",
                                proofatlas::InferenceRule::EqualityFactoring => "Equality Factoring",
                                proofatlas::InferenceRule::Demodulation => "Demodulation",
                                _ => "Other",
                            },
                            step.inference.premises,
                            step.inference.conclusion
                        );
                    }
                }
            }
        }
    }
}
