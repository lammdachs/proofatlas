//! End-to-end tests for ONNX-based clause selection in proof search
//!
//! These tests verify that the ONNX model can be used for clause selection
//! during theorem proving and produces correct proofs.

use proofatlas::{
    parse_tptp_file, ClauseSelector, LiteralSelectionStrategy, OnnxClauseSelector,
    SaturationConfig, SaturationResult, SaturationState,
};
use std::io::Write;
use std::time::Duration;

/// Path to the test ONNX models (relative to rust/ directory)
const MODEL_AGE_WEIGHT: &str = "../.selectors/age_weight.onnx";
const MODEL_GCN: &str = "../.selectors/gcn.onnx";
const MODEL_MLP: &str = "../.selectors/mlp.onnx";

fn create_selector(model_path: &str) -> Box<dyn ClauseSelector> {
    Box::new(OnnxClauseSelector::new(model_path).expect("Failed to load ONNX model"))
}

/// Run a problem with the ONNX-based clause selector
fn run_with_onnx_selector(
    problem_name: &str,
    problem_file: &str,
    model_path: &str,
    timeout_secs: u64,
) -> (SaturationResult, Duration, usize) {
    // Parse the problem
    let formula = parse_tptp_file(problem_file, &[]).expect("Failed to parse TPTP file");

    // Configure saturation
    let config = SaturationConfig {
        max_clauses: 10000,
        max_iterations: 10000,
        timeout: Duration::from_secs(timeout_secs),
        literal_selection: LiteralSelectionStrategy::SelectAll,
        ..Default::default()
    };

    // Run saturation with ONNX selector
    let state = SaturationState::new(formula.clauses.clone(), config, create_selector(model_path));

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

/// Compare different ONNX model configurations
fn compare_models(problem_name: &str, problem_file: &str, timeout_secs: u64) {
    println!("\n=== {} ===", problem_name);

    let (result_age_weight, time_age_weight, clauses_age_weight) =
        run_with_onnx_selector(problem_name, problem_file, MODEL_AGE_WEIGHT, timeout_secs);

    let (result_gcn, time_gcn, clauses_gcn) =
        run_with_onnx_selector(problem_name, problem_file, MODEL_GCN, timeout_secs);

    let (result_mlp, time_mlp, clauses_mlp) =
        run_with_onnx_selector(problem_name, problem_file, MODEL_MLP, timeout_secs);

    // Write comparison to trace file
    let trace_file = format!("test_traces/onnx_{}.trace", problem_name);
    if let Ok(mut f) = std::fs::File::create(&trace_file) {
        writeln!(f, "=== {} ===", problem_name).ok();
        writeln!(f, "").ok();
        writeln!(
            f,
            "age_weight: {:?} in {:.3}s, {} clauses",
            result_type(&result_age_weight),
            time_age_weight.as_secs_f64(),
            clauses_age_weight
        )
        .ok();
        writeln!(
            f,
            "gcn: {:?} in {:.3}s, {} clauses",
            result_type(&result_gcn),
            time_gcn.as_secs_f64(),
            clauses_gcn
        )
        .ok();
        writeln!(
            f,
            "mlp: {:?} in {:.3}s, {} clauses",
            result_type(&result_mlp),
            time_mlp.as_secs_f64(),
            clauses_mlp
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
        MODEL_AGE_WEIGHT,
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
        MODEL_AGE_WEIGHT,
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
        MODEL_AGE_WEIGHT,
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
        MODEL_AGE_WEIGHT,
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
        MODEL_AGE_WEIGHT,
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
        MODEL_AGE_WEIGHT,
        10,
    );
    assert!(
        matches!(result, SaturationResult::Proof(_)),
        "ONNX selector should find proof for inverse_of_inverse"
    );
}

// ============================================================================
// Tests with different model types
// ============================================================================

#[test]
fn test_onnx_gcn_right_identity() {
    let (result, _, _) = run_with_onnx_selector(
        "right_identity",
        "tests/problems/right_identity.p",
        MODEL_GCN,
        10,
    );
    assert!(
        matches!(result, SaturationResult::Proof(_)),
        "GCN selector should find proof for right_identity"
    );
}

#[test]
fn test_onnx_mlp_right_identity() {
    let (result, _, _) = run_with_onnx_selector(
        "right_identity",
        "tests/problems/right_identity.p",
        MODEL_MLP,
        10,
    );
    assert!(
        matches!(result, SaturationResult::Proof(_)),
        "MLP selector should find proof for right_identity"
    );
}

// ============================================================================
// Comparison tests - compare different ONNX models
// ============================================================================

#[test]
fn test_compare_models_right_identity() {
    compare_models("right_identity", "tests/problems/right_identity.p", 10);
}

#[test]
fn test_compare_models_uniqueness_of_inverse() {
    compare_models(
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
    let selector_age_weight = OnnxClauseSelector::new(MODEL_AGE_WEIGHT);
    assert!(selector_age_weight.is_ok(), "Should load age_weight model");

    let selector_gcn = OnnxClauseSelector::new(MODEL_GCN);
    assert!(selector_gcn.is_ok(), "Should load gcn model");

    let selector_mlp = OnnxClauseSelector::new(MODEL_MLP);
    assert!(selector_mlp.is_ok(), "Should load mlp model");
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

    let config = SaturationConfig {
        max_clauses: 1000,
        max_iterations: 1000,
        timeout: Duration::from_secs(5),
        literal_selection: LiteralSelectionStrategy::SelectAll,
        ..Default::default()
    };

    let state = SaturationState::new(clauses, config, create_selector(MODEL_AGE_WEIGHT));

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
    scorer.load_model(MODEL_AGE_WEIGHT).expect("Failed to load model");

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
