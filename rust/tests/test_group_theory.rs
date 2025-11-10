//! End-to-end tests for group theory problems

use proofatlas::{
    parse_tptp_file, LiteralSelectionStrategy, SaturationConfig, SaturationResult, SaturationState,
};
use std::io::Write;
use std::time::Duration;

/// Run a group theory problem and return the result with trace
fn run_group_problem(
    problem_name: &str,
    problem_file: &str,
    timeout_secs: u64,
) -> (SaturationResult, Vec<String>) {
    // Parse the problem
    let formula = parse_tptp_file(problem_file, &[]).expect("Failed to parse TPTP file");

    // Configure saturation with reasonable limits
    let config = SaturationConfig {
        max_clauses: 10000,
        max_iterations: 10000,
        timeout: Duration::from_secs(timeout_secs),
        use_superposition: true,
        literal_selection: LiteralSelectionStrategy::SelectAll,
        ..Default::default()
    };

    // Collect trace information
    let mut trace = Vec::new();
    trace.push(format!("=== {} ===", problem_name));
    trace.push("Initial clauses:".to_string());
    for (i, clause) in formula.clauses.iter().enumerate() {
        trace.push(format!("[{}] {}", i, clause));
    }
    trace.push("".to_string());

    // Run saturation
    let state = SaturationState::new(formula.clauses, config);
    let start_time = std::time::Instant::now();
    let result = state.saturate();
    let elapsed = start_time.elapsed();

    // Add result to trace
    match &result {
        SaturationResult::Proof(proof) => {
            trace.push(format!("✓ PROOF FOUND in {:.3}s", elapsed.as_secs_f64()));
            trace.push(format!("Generated {} clauses", proof.steps.len()));

            // Find the empty clause and trace back
            if let Some(empty_step) = proof
                .steps
                .iter()
                .find(|s| s.inference.conclusion.is_empty())
            {
                trace.push(format!("Empty clause at index: {}", empty_step.clause_idx));
            }

            // Add all proof steps
            trace.push("".to_string());
            trace.push("=== All Proof Steps ===".to_string());
            for step in &proof.steps {
                trace.push(format!(
                    "[{}] {}",
                    step.clause_idx, step.inference.conclusion
                ));
                trace.push(format!("    Rule: {:?}", step.inference.rule));
                trace.push(format!("    Parents: {:?}", step.inference.premises));
                trace.push("".to_string());
            }
        }
        SaturationResult::Saturated(steps, _clauses) => {
            trace.push(format!(
                "✗ SATURATED without proof in {:.3}s",
                elapsed.as_secs_f64()
            ));
            trace.push(format!("Generated {} clauses", steps.len()));
        }
        SaturationResult::ResourceLimit(steps, _clauses) => {
            trace.push(format!(
                "✗ RESOURCE LIMIT reached in {:.3}s",
                elapsed.as_secs_f64()
            ));
            trace.push(format!("Generated {} clauses", steps.len()));

            // Add all steps even if no proof found
            trace.push("".to_string());
            trace.push("=== All Generated Clauses (no proof found) ===".to_string());
            for step in steps {
                trace.push(format!(
                    "[{}] {}",
                    step.clause_idx, step.inference.conclusion
                ));
                trace.push(format!("    Rule: {:?}", step.inference.rule));
                trace.push(format!("    Parents: {:?}", step.inference.premises));
                trace.push("".to_string());
            }
        }
        SaturationResult::Timeout(_steps, _clauses) => {
            trace.push(format!("✗ TIMEOUT after {:.3}s", elapsed.as_secs_f64()));
        }
    }
    trace.push("".to_string());

    (result, trace)
}

/// Print trace to stdout and optionally to a file
fn print_trace(trace: &[String], file_name: Option<&str>) {
    for line in trace {
        println!("{}", line);
    }

    if let Some(file) = file_name {
        let mut f = std::fs::File::create(file).expect("Failed to create trace file");
        for line in trace {
            writeln!(f, "{}", line).expect("Failed to write trace");
        }
    }
}

#[test]
fn test_right_identity() {
    let (result, trace) =
        run_group_problem("right_identity", "tests/problems/right_identity.p", 10);
    print_trace(&trace, Some("test_traces/right_identity.trace"));

    match result {
        SaturationResult::Proof(_) => {}
        _ => panic!("Failed to prove right identity"),
    }
}

#[test]
fn test_uniqueness_of_identity() {
    let (result, trace) = run_group_problem(
        "uniqueness_of_identity",
        "tests/problems/uniqueness_of_identity.p",
        10,
    );
    print_trace(&trace, Some("test_traces/uniqueness_of_identity.trace"));

    match result {
        SaturationResult::Proof(_) => {}
        _ => panic!("Failed to prove uniqueness of identity"),
    }
}

#[test]
fn test_right_inverse() {
    let (result, trace) = run_group_problem("right_inverse", "tests/problems/right_inverse.p", 10);
    print_trace(&trace, Some("test_traces/right_inverse.trace"));

    match result {
        SaturationResult::Proof(_) => {}
        _ => panic!("Failed to prove right inverse"),
    }
}

#[test]
fn test_uniqueness_of_inverse() {
    let (result, trace) = run_group_problem(
        "uniqueness_of_inverse",
        "tests/problems/uniqueness_of_inverse.p",
        10,
    );
    print_trace(&trace, Some("test_traces/uniqueness_of_inverse.trace"));

    match result {
        SaturationResult::Proof(_) => {}
        _ => panic!("Failed to prove uniqueness of inverse"),
    }
}

#[test]
fn test_inverse_of_identity() {
    let (result, trace) = run_group_problem(
        "inverse_of_identity",
        "tests/problems/inverse_of_identity.p",
        10,
    );
    print_trace(&trace, Some("test_traces/inverse_of_identity.trace"));

    match result {
        SaturationResult::Proof(_) => {}
        _ => panic!("Failed to prove inverse of identity is identity"),
    }
}

#[test]
fn test_inverse_of_inverse() {
    let (result, trace) = run_group_problem(
        "inverse_of_inverse",
        "tests/problems/inverse_of_inverse.p",
        10,
    );
    print_trace(&trace, Some("test_traces/inverse_of_inverse.trace"));

    match result {
        SaturationResult::Proof(_) => {}
        _ => panic!("Failed to prove inverse of inverse"),
    }
}
