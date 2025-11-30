//! End-to-end tests for ONNX-based clause selection in proof search
//!
//! These tests verify that the ONNX model can be used for clause selection
//! during theorem proving and produces correct proofs.

#![cfg(feature = "onnx")]

use proofatlas::{
    parse_tptp_file, LiteralSelectionStrategy, OnnxClauseSelector, SaturationConfig,
    SaturationResult, SaturationState,
};
use std::io::Write;
use std::time::Duration;

/// Path to the test ONNX models (relative to rust/ directory)
const MODEL_P03: &str = "../.selectors/age_weight_p03.onnx";
const MODEL_P05: &str = "../.selectors/age_weight_p05.onnx";
const MODEL_P07: &str = "../.selectors/age_weight_p07.onnx";

/// Run a problem with the ONNX-based clause selector
fn run_with_onnx_selector(
    problem_name: &str,
    problem_file: &str,
    model_path: &str,
    timeout_secs: u64,
) -> (SaturationResult, Duration, usize) {
    // Parse the problem
    let formula = parse_tptp_file(problem_file, &[]).expect("Failed to parse TPTP file");

    // Create ONNX selector
    let onnx_selector =
        OnnxClauseSelector::new(model_path).expect("Failed to load ONNX model");

    // Configure saturation
    let config = SaturationConfig {
        max_clauses: 10000,
        max_iterations: 10000,
        timeout: Duration::from_secs(timeout_secs),
        use_superposition: true,
        literal_selection: LiteralSelectionStrategy::SelectAll,
        ..Default::default()
    };

    // Run saturation with ONNX selector
    let mut state = SaturationState::new(formula.clauses.clone(), config);
    state.set_clause_selector(Box::new(onnx_selector));

    let start_time = std::time::Instant::now();
    let result = state.saturate();
    let elapsed = start_time.elapsed();

    let clause_count = match &result {
        SaturationResult::Proof(proof) => proof.steps.len(),
        SaturationResult::Saturated(steps, _) => steps.len(),
        SaturationResult::ResourceLimit(steps, _) => steps.len(),
        SaturationResult::Timeout(steps, _) => steps.len(),
    };

    println!(
        "{} with ONNX selector ({}): {:?} in {:.3}s, {} clauses",
        problem_name,
        model_path,
        match &result {
            SaturationResult::Proof(_) => "PROOF",
            SaturationResult::Saturated(_, _) => "SATURATED",
            SaturationResult::ResourceLimit(_, _) => "RESOURCE_LIMIT",
            SaturationResult::Timeout(_, _) => "TIMEOUT",
        },
        elapsed.as_secs_f64(),
        clause_count
    );

    (result, elapsed, clause_count)
}

/// Run a problem with the default clause selector for comparison
fn run_with_default_selector(
    problem_name: &str,
    problem_file: &str,
    timeout_secs: u64,
) -> (SaturationResult, Duration, usize) {
    let formula = parse_tptp_file(problem_file, &[]).expect("Failed to parse TPTP file");

    let config = SaturationConfig {
        max_clauses: 10000,
        max_iterations: 10000,
        timeout: Duration::from_secs(timeout_secs),
        use_superposition: true,
        literal_selection: LiteralSelectionStrategy::SelectAll,
        ..Default::default()
    };

    let state = SaturationState::new(formula.clauses.clone(), config);

    let start_time = std::time::Instant::now();
    let result = state.saturate();
    let elapsed = start_time.elapsed();

    let clause_count = match &result {
        SaturationResult::Proof(proof) => proof.steps.len(),
        SaturationResult::Saturated(steps, _) => steps.len(),
        SaturationResult::ResourceLimit(steps, _) => steps.len(),
        SaturationResult::Timeout(steps, _) => steps.len(),
    };

    println!(
        "{} with default selector: {:?} in {:.3}s, {} clauses",
        problem_name,
        match &result {
            SaturationResult::Proof(_) => "PROOF",
            SaturationResult::Saturated(_, _) => "SATURATED",
            SaturationResult::ResourceLimit(_, _) => "RESOURCE_LIMIT",
            SaturationResult::Timeout(_, _) => "TIMEOUT",
        },
        elapsed.as_secs_f64(),
        clause_count
    );

    (result, elapsed, clause_count)
}

/// Compare ONNX selector with default selector
fn compare_selectors(problem_name: &str, problem_file: &str, timeout_secs: u64) {
    println!("\n=== {} ===", problem_name);

    let (default_result, default_time, default_clauses) =
        run_with_default_selector(problem_name, problem_file, timeout_secs);

    let (onnx_result_p03, onnx_time_p03, onnx_clauses_p03) =
        run_with_onnx_selector(problem_name, problem_file, MODEL_P03, timeout_secs);

    let (onnx_result_p05, onnx_time_p05, onnx_clauses_p05) =
        run_with_onnx_selector(problem_name, problem_file, MODEL_P05, timeout_secs);

    let (onnx_result_p07, onnx_time_p07, onnx_clauses_p07) =
        run_with_onnx_selector(problem_name, problem_file, MODEL_P07, timeout_secs);

    // Write comparison to trace file
    let trace_file = format!("test_traces/onnx_{}.trace", problem_name);
    if let Ok(mut f) = std::fs::File::create(&trace_file) {
        writeln!(f, "=== {} ===", problem_name).ok();
        writeln!(f, "").ok();
        writeln!(
            f,
            "Default:    {:?} in {:.3}s, {} clauses",
            result_type(&default_result),
            default_time.as_secs_f64(),
            default_clauses
        )
        .ok();
        writeln!(
            f,
            "ONNX p=0.3: {:?} in {:.3}s, {} clauses",
            result_type(&onnx_result_p03),
            onnx_time_p03.as_secs_f64(),
            onnx_clauses_p03
        )
        .ok();
        writeln!(
            f,
            "ONNX p=0.5: {:?} in {:.3}s, {} clauses",
            result_type(&onnx_result_p05),
            onnx_time_p05.as_secs_f64(),
            onnx_clauses_p05
        )
        .ok();
        writeln!(
            f,
            "ONNX p=0.7: {:?} in {:.3}s, {} clauses",
            result_type(&onnx_result_p07),
            onnx_time_p07.as_secs_f64(),
            onnx_clauses_p07
        )
        .ok();
    }
}

fn result_type(result: &SaturationResult) -> &'static str {
    match result {
        SaturationResult::Proof(_) => "PROOF",
        SaturationResult::Saturated(_, _) => "SATURATED",
        SaturationResult::ResourceLimit(_, _) => "RESOURCE_LIMIT",
        SaturationResult::Timeout(_, _) => "TIMEOUT",
    }
}

// ============================================================================
// Basic tests - verify ONNX selector can find proofs
// ============================================================================

#[test]
fn test_onnx_right_identity() {
    let (result, _, _) = run_with_onnx_selector(
        "right_identity",
        "tests/problems/right_identity.p",
        MODEL_P05,
        10,
    );
    assert!(
        matches!(result, SaturationResult::Proof(_)),
        "ONNX selector should find proof for right_identity"
    );
}

#[test]
fn test_onnx_uniqueness_of_identity() {
    let (result, _, _) = run_with_onnx_selector(
        "uniqueness_of_identity",
        "tests/problems/uniqueness_of_identity.p",
        MODEL_P05,
        10,
    );
    assert!(
        matches!(result, SaturationResult::Proof(_)),
        "ONNX selector should find proof for uniqueness_of_identity"
    );
}

#[test]
fn test_onnx_right_inverse() {
    let (result, _, _) = run_with_onnx_selector(
        "right_inverse",
        "tests/problems/right_inverse.p",
        MODEL_P05,
        10,
    );
    assert!(
        matches!(result, SaturationResult::Proof(_)),
        "ONNX selector should find proof for right_inverse"
    );
}

#[test]
fn test_onnx_uniqueness_of_inverse() {
    let (result, _, _) = run_with_onnx_selector(
        "uniqueness_of_inverse",
        "tests/problems/uniqueness_of_inverse.p",
        MODEL_P05,
        10,
    );
    assert!(
        matches!(result, SaturationResult::Proof(_)),
        "ONNX selector should find proof for uniqueness_of_inverse"
    );
}

#[test]
fn test_onnx_inverse_of_identity() {
    let (result, _, _) = run_with_onnx_selector(
        "inverse_of_identity",
        "tests/problems/inverse_of_identity.p",
        MODEL_P05,
        10,
    );
    assert!(
        matches!(result, SaturationResult::Proof(_)),
        "ONNX selector should find proof for inverse_of_identity"
    );
}

#[test]
fn test_onnx_inverse_of_inverse() {
    let (result, _, _) = run_with_onnx_selector(
        "inverse_of_inverse",
        "tests/problems/inverse_of_inverse.p",
        MODEL_P05,
        10,
    );
    assert!(
        matches!(result, SaturationResult::Proof(_)),
        "ONNX selector should find proof for inverse_of_inverse"
    );
}

// ============================================================================
// Tests with different model configurations (p values)
// ============================================================================

#[test]
fn test_onnx_p03_right_identity() {
    let (result, _, _) = run_with_onnx_selector(
        "right_identity",
        "tests/problems/right_identity.p",
        MODEL_P03,
        10,
    );
    assert!(
        matches!(result, SaturationResult::Proof(_)),
        "ONNX p=0.3 selector should find proof for right_identity"
    );
}

#[test]
fn test_onnx_p07_right_identity() {
    let (result, _, _) = run_with_onnx_selector(
        "right_identity",
        "tests/problems/right_identity.p",
        MODEL_P07,
        10,
    );
    assert!(
        matches!(result, SaturationResult::Proof(_)),
        "ONNX p=0.7 selector should find proof for right_identity"
    );
}

// ============================================================================
// Comparison tests - run both selectors and compare
// ============================================================================

#[test]
fn test_compare_selectors_right_identity() {
    compare_selectors("right_identity", "tests/problems/right_identity.p", 10);
}

#[test]
fn test_compare_selectors_uniqueness_of_inverse() {
    compare_selectors(
        "uniqueness_of_inverse",
        "tests/problems/uniqueness_of_inverse.p",
        10,
    );
}

// ============================================================================
// Model loading tests
// ============================================================================

#[test]
fn test_onnx_model_loading() {
    // Test that all models can be loaded
    let selector_p03 = OnnxClauseSelector::new(MODEL_P03);
    assert!(selector_p03.is_ok(), "Should load p=0.3 model");

    let selector_p05 = OnnxClauseSelector::new(MODEL_P05);
    assert!(selector_p05.is_ok(), "Should load p=0.5 model");

    let selector_p07 = OnnxClauseSelector::new(MODEL_P07);
    assert!(selector_p07.is_ok(), "Should load p=0.7 model");
}

#[test]
fn test_onnx_model_not_found() {
    let selector = OnnxClauseSelector::new("nonexistent.onnx");
    assert!(selector.is_err(), "Should fail for non-existent model");
}

// ============================================================================
// Simple propositional tests
// ============================================================================

#[test]
fn test_onnx_simple_resolution() {
    // Create a simple problem: P(a), ~P(X) | Q(X), ~Q(a) -> empty
    use proofatlas::{Atom, Clause, Constant, Literal, PredicateSymbol, Term, Variable};

    let p = PredicateSymbol {
        name: "P".to_string(),
        arity: 1,
    };
    let q = PredicateSymbol {
        name: "Q".to_string(),
        arity: 1,
    };
    let a = Term::Constant(Constant {
        name: "a".to_string(),
    });
    let x = Term::Variable(Variable {
        name: "X".to_string(),
    });

    let clauses = vec![
        Clause::new(vec![Literal::positive(Atom {
            predicate: p.clone(),
            args: vec![a.clone()],
        })]),
        Clause::new(vec![
            Literal::negative(Atom {
                predicate: p.clone(),
                args: vec![x.clone()],
            }),
            Literal::positive(Atom {
                predicate: q.clone(),
                args: vec![x.clone()],
            }),
        ]),
        Clause::new(vec![Literal::negative(Atom {
            predicate: q.clone(),
            args: vec![a.clone()],
        })]),
    ];

    // Create ONNX selector
    let onnx_selector = OnnxClauseSelector::new(MODEL_P05).expect("Failed to load model");

    let config = SaturationConfig {
        max_clauses: 1000,
        max_iterations: 1000,
        timeout: Duration::from_secs(5),
        use_superposition: true,
        literal_selection: LiteralSelectionStrategy::SelectAll,
        ..Default::default()
    };

    let mut state = SaturationState::new(clauses, config);
    state.set_clause_selector(Box::new(onnx_selector));

    let result = state.saturate();

    assert!(
        matches!(result, SaturationResult::Proof(_)),
        "ONNX selector should find proof for simple resolution problem"
    );
}

// ============================================================================
// Integration test with clause scoring
// ============================================================================

#[test]
fn test_clause_scorer_integration() {
    use proofatlas::{Atom, Clause, ClauseScorer, Literal, PredicateSymbol, Term, Variable};

    // Create test clauses
    let clauses: Vec<Clause> = (0..5)
        .map(|i| {
            let p = PredicateSymbol {
                name: format!("P{}", i),
                arity: 1,
            };
            let mut clause = Clause::new(vec![Literal::positive(Atom {
                predicate: p,
                args: vec![Term::Variable(Variable {
                    name: "X".to_string(),
                })],
            })]);
            clause.age = i * 10;
            clause
        })
        .collect();

    // Load model and score clauses
    let mut scorer = ClauseScorer::new();
    scorer.load_model(MODEL_P05).expect("Failed to load model");

    let clause_refs: Vec<&Clause> = clauses.iter().collect();
    let scores = scorer
        .score_clauses(&clause_refs)
        .expect("Failed to score clauses");

    // Verify we got scores for all clauses
    assert_eq!(scores.len(), 5);

    // All scores should be valid (not NaN or Inf)
    for score in &scores {
        assert!(score.is_finite(), "Score should be finite");
    }

    println!("Clause scores:");
    for (i, (clause, score)) in clauses.iter().zip(scores.iter()).enumerate() {
        println!("  {} (age={}): {:.4}", i, clause.age, score);
    }
}
