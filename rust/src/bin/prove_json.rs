//! Command-line theorem prover with JSON export

use std::env;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use proofatlas::core::json::{ClauseJson, ConfigJson, ProofAttemptJson, StatisticsJson};
use proofatlas::{parse_tptp_file, SaturationConfig, SaturationState};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <tptp_file> [options]", args[0]);
        eprintln!("\nOptions:");
        eprintln!("  --timeout <seconds>    Set timeout (default: 60)");
        eprintln!("  --max-clauses <n>      Set max clauses (default: 10000)");
        eprintln!("  --include <dir>        Add include directory (can be used multiple times)");
        eprintln!("  --json <file>          Export proof attempt to JSON file");
        eprintln!("  --verbose              Show detailed progress");
        std::process::exit(1);
    }

    let filename = &args[1];
    let mut config = SaturationConfig::default();
    let mut verbose = false;
    let mut json_output: Option<String> = None;
    let mut include_dirs: Vec<String> = Vec::new();

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
            "--include" => {
                if i + 1 < args.len() {
                    include_dirs.push(args[i + 1].clone());
                    i += 1;
                }
            }
            "--json" => {
                if i + 1 < args.len() {
                    json_output = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--verbose" => {
                verbose = true;
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
            }
        }
        i += 1;
    }

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

    // Store initial clauses for JSON export
    let initial_clauses: Vec<ClauseJson> = formula.clauses.iter().map(|c| c.into()).collect();

    // Run saturation
    println!("Running saturation with:");
    println!("  Max clauses: {}", config.max_clauses);
    println!("  Timeout: {:?}", config.timeout);
    println!();

    let start_time = Instant::now();
    let saturation = SaturationState::new(formula.clauses, config.clone());
    let result = saturation.saturate();
    let elapsed = start_time.elapsed();

    // Report result
    match &result {
        proofatlas::SaturationResult::Proof(proof) => {
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
        proofatlas::SaturationResult::Saturated(_, clauses) => {
            println!("✗ SATURATED in {:.3}s", elapsed.as_secs_f64());
            println!("  No proof found - the formula may be satisfiable");
            println!("  Final clauses: {}", clauses.len());
        }
        proofatlas::SaturationResult::ResourceLimit(_, clauses) => {
            println!("✗ RESOURCE LIMIT in {:.3}s", elapsed.as_secs_f64());
            println!("  Exceeded clause limit or iteration limit");
            println!("  Final clauses: {}", clauses.len());
        }
        proofatlas::SaturationResult::Timeout(_, clauses) => {
            println!("✗ TIMEOUT in {:.3}s", elapsed.as_secs_f64());
            println!("  Exceeded time limit");
            println!("  Final clauses: {}", clauses.len());
        }
    }

    // Export to JSON if requested
    if let Some(json_file) = json_output {
        let proof_attempt = ProofAttemptJson {
            problem_file: filename.to_string(),
            initial_clauses,
            config: ConfigJson {
                max_clauses: config.max_clauses,
                max_iterations: config.max_iterations,
                timeout_seconds: config.timeout.as_secs_f64(),
                literal_selection: format!("{:?}", config.literal_selection),
            },
            result: result.to_json(elapsed.as_secs_f64()),
            statistics: StatisticsJson {
                clauses_generated: 0, // TODO: track this
                clauses_processed: 0, // TODO: track this
                clauses_subsumed: 0,  // TODO: track this
                time_elapsed_seconds: elapsed.as_secs_f64(),
            },
        };

        match serde_json::to_string_pretty(&proof_attempt) {
            Ok(json_str) => match File::create(&json_file) {
                Ok(mut file) => {
                    if let Err(e) = file.write_all(json_str.as_bytes()) {
                        eprintln!("Failed to write JSON file: {}", e);
                    } else {
                        println!("\nProof attempt exported to: {}", json_file);
                    }
                }
                Err(e) => eprintln!("Failed to create JSON file: {}", e),
            },
            Err(e) => eprintln!("Failed to serialize to JSON: {}", e),
        }
    }
}
