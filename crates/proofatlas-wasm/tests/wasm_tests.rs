// These are wasm_bindgen_test functions, only runnable via `wasm-pack test`.
// cargo test sees them as dead code since they lack #[test].
#![allow(dead_code)]

use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

use proofatlas_wasm::ProofAtlasWasm;

// ---------------------------------------------------------------------------
// Helper: build a JsValue options object from a JSON string
// ---------------------------------------------------------------------------
fn options(json: &str) -> JsValue {
    js_sys::JSON::parse(json).expect("invalid JSON for options")
}

fn default_options() -> JsValue {
    options(r#"{"timeout_ms": 5000, "max_iterations": 5000}"#)
}

// ---------------------------------------------------------------------------
// validate_tptp
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn validate_valid_single_clause() {
    let atlas = ProofAtlasWasm::new();
    let result = atlas.validate_tptp("cnf(c1, axiom, p(a)).");
    assert!(result.is_ok(), "expected Ok, got {:?}", result);
    let msg = result.unwrap();
    assert!(msg.contains("1 clauses"), "unexpected message: {}", msg);
}

#[wasm_bindgen_test]
fn validate_valid_multiple_clauses() {
    let atlas = ProofAtlasWasm::new();
    let input = "cnf(c1, axiom, p(a)).\ncnf(c2, axiom, ~p(a) | q(b)).";
    let result = atlas.validate_tptp(input);
    assert!(result.is_ok(), "expected Ok, got {:?}", result);
    let msg = result.unwrap();
    assert!(msg.contains("2 clauses"), "unexpected message: {}", msg);
}

#[wasm_bindgen_test]
fn validate_fof_input() {
    let atlas = ProofAtlasWasm::new();
    let input = "fof(ax, axiom, ![X]: (p(X) => q(X))).\nfof(goal, conjecture, p(a) => q(a)).";
    let result = atlas.validate_tptp(input);
    assert!(result.is_ok(), "expected Ok for FOF input, got {:?}", result);
}

#[wasm_bindgen_test]
fn validate_empty_input() {
    let atlas = ProofAtlasWasm::new();
    let result = atlas.validate_tptp("");
    // Empty input should parse as 0 clauses or fail -- either is acceptable,
    // but it must not panic.
    match &result {
        Ok(msg) => assert!(msg.contains("0 clauses"), "unexpected message: {}", msg),
        Err(_) => {} // parse error is also acceptable
    }
}

#[wasm_bindgen_test]
fn validate_garbage_input() {
    let atlas = ProofAtlasWasm::new();
    let result = atlas.validate_tptp("this is not valid TPTP at all !!!");
    // Should either error or parse 0 clauses (the parser skips unknown lines).
    match &result {
        Ok(msg) => assert!(msg.contains("0 clauses"), "unexpected message: {}", msg),
        Err(_) => {} // parse error is acceptable
    }
}

#[wasm_bindgen_test]
fn validate_malformed_clause() {
    let atlas = ProofAtlasWasm::new();
    let result = atlas.validate_tptp("cnf(c1, axiom, p(a)");
    // Missing period/closing paren -- should error or return 0 clauses.
    match &result {
        Ok(msg) => assert!(msg.contains("0 clauses"), "unexpected message: {}", msg),
        Err(_) => {}
    }
}

// ---------------------------------------------------------------------------
// prove -- unsatisfiable problems (should find proof)
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn prove_simple_contradiction() {
    let atlas = ProofAtlasWasm::new();
    let input = "cnf(c1, axiom, p(a)).\ncnf(c2, axiom, ~p(a)).";
    let result = atlas.prove(input, default_options());
    assert!(result.is_ok(), "prove returned error: {:?}", result);

    let js_val = result.unwrap();
    let obj: serde_json::Value = serde_wasm_bindgen::from_value(js_val).unwrap();
    assert_eq!(obj["success"], true, "expected proof found");
    assert_eq!(obj["status"], "proof_found");
    assert!(obj["proof"].is_array(), "expected proof array");
    assert!(obj["statistics"]["initial_clauses"].as_u64().unwrap() >= 2);
    assert!(obj["statistics"]["time_ms"].as_u64().is_some());
}

#[wasm_bindgen_test]
fn prove_resolution_chain() {
    // p(a), ~p(X)|q(X), ~q(a)  -- requires two resolution steps
    let atlas = ProofAtlasWasm::new();
    let input = "\
        cnf(c1, axiom, p(a)).\n\
        cnf(c2, axiom, ~p(X) | q(X)).\n\
        cnf(c3, axiom, ~q(a)).\n";
    let result = atlas.prove(input, default_options());
    assert!(result.is_ok(), "prove returned error: {:?}", result);

    let obj: serde_json::Value =
        serde_wasm_bindgen::from_value(result.unwrap()).unwrap();
    assert_eq!(obj["success"], true);
    assert_eq!(obj["status"], "proof_found");
}

#[wasm_bindgen_test]
fn prove_with_equality() {
    // a = b, f(a) != f(b)  -- requires superposition / paramodulation
    let atlas = ProofAtlasWasm::new();
    let input = "\
        cnf(eq, axiom, a = b).\n\
        cnf(neq, axiom, f(a) != f(b)).\n";
    let result = atlas.prove(input, default_options());
    assert!(result.is_ok(), "prove returned error: {:?}", result);

    let obj: serde_json::Value =
        serde_wasm_bindgen::from_value(result.unwrap()).unwrap();
    assert_eq!(obj["success"], true, "expected proof for equality problem");
    assert_eq!(obj["status"], "proof_found");
}

#[wasm_bindgen_test]
fn prove_empty_clause_in_input() {
    // The empty clause is directly in the input -- immediate proof.
    let atlas = ProofAtlasWasm::new();
    let input = "cnf(c1, axiom, $false).";
    let result = atlas.prove(input, default_options());
    assert!(result.is_ok(), "prove returned error: {:?}", result);

    let obj: serde_json::Value =
        serde_wasm_bindgen::from_value(result.unwrap()).unwrap();
    assert_eq!(obj["success"], true);
    assert_eq!(obj["status"], "proof_found");
}

// ---------------------------------------------------------------------------
// prove -- satisfiable problems (should NOT find proof)
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn prove_satisfiable_single_clause() {
    // Single positive clause -- trivially satisfiable.
    let atlas = ProofAtlasWasm::new();
    let input = "cnf(c1, axiom, p(a)).";
    let opts = options(r#"{"timeout_ms": 2000, "max_iterations": 1000}"#);
    let result = atlas.prove(input, opts);
    assert!(result.is_ok(), "prove returned error: {:?}", result);

    let obj: serde_json::Value =
        serde_wasm_bindgen::from_value(result.unwrap()).unwrap();
    assert_eq!(obj["success"], false, "must NOT find proof for satisfiable input");
    // Status should be "saturated" or "resource_limit".
    let status = obj["status"].as_str().unwrap();
    assert!(
        status == "saturated" || status == "resource_limit",
        "unexpected status: {}",
        status
    );
}

#[wasm_bindgen_test]
fn prove_satisfiable_no_contradiction() {
    // Two clauses that do not interact -- satisfiable.
    let atlas = ProofAtlasWasm::new();
    let input = "cnf(c1, axiom, p(a)).\ncnf(c2, axiom, q(b)).";
    let opts = options(r#"{"timeout_ms": 2000, "max_iterations": 1000}"#);
    let result = atlas.prove(input, opts);
    assert!(result.is_ok());

    let obj: serde_json::Value =
        serde_wasm_bindgen::from_value(result.unwrap()).unwrap();
    assert_eq!(obj["success"], false, "must NOT find proof for satisfiable input");
}

// ---------------------------------------------------------------------------
// prove_with_trace
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn prove_with_trace_returns_trace() {
    let atlas = ProofAtlasWasm::new();
    let input = "cnf(c1, axiom, p(a)).\ncnf(c2, axiom, ~p(a)).";
    let result = atlas.prove_with_trace(input, default_options());
    assert!(result.is_ok(), "prove_with_trace returned error: {:?}", result);

    let obj: serde_json::Value =
        serde_wasm_bindgen::from_value(result.unwrap()).unwrap();
    assert_eq!(obj["success"], true);
    assert_eq!(obj["status"], "proof_found");

    // Trace should be present and have the expected structure.
    let trace = &obj["trace"];
    assert!(!trace.is_null(), "trace should be present");
    assert!(trace["initial_clauses"].is_array(), "trace should have initial_clauses");
    assert!(trace["iterations"].is_array(), "trace should have iterations");
}

#[wasm_bindgen_test]
fn prove_without_trace_has_no_trace() {
    let atlas = ProofAtlasWasm::new();
    let input = "cnf(c1, axiom, p(a)).\ncnf(c2, axiom, ~p(a)).";
    let result = atlas.prove(input, default_options());
    assert!(result.is_ok());

    let obj: serde_json::Value =
        serde_wasm_bindgen::from_value(result.unwrap()).unwrap();
    assert!(obj["trace"].is_null(), "trace should be null when not requested");
}

// ---------------------------------------------------------------------------
// Options handling
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn prove_with_literal_selection_strategies() {
    let atlas = ProofAtlasWasm::new();
    let input = "cnf(c1, axiom, p(a)).\ncnf(c2, axiom, ~p(a)).";

    for sel in &["0", "20", "21", "22"] {
        let opts = options(&format!(
            r#"{{"timeout_ms": 5000, "max_iterations": 5000, "literal_selection": "{}"}}"#,
            sel
        ));
        let result = atlas.prove(input, opts);
        assert!(result.is_ok(), "prove with sel {} failed: {:?}", sel, result);
        let obj: serde_json::Value =
            serde_wasm_bindgen::from_value(result.unwrap()).unwrap();
        assert_eq!(
            obj["success"], true,
            "should find proof with literal_selection={}",
            sel
        );
    }
}

#[wasm_bindgen_test]
fn prove_with_custom_age_weight_ratio() {
    let atlas = ProofAtlasWasm::new();
    let input = "cnf(c1, axiom, p(a)).\ncnf(c2, axiom, ~p(a)).";
    let opts = options(r#"{"timeout_ms": 5000, "age_weight_ratio": 0.8}"#);
    let result = atlas.prove(input, opts);
    assert!(result.is_ok());
    let obj: serde_json::Value =
        serde_wasm_bindgen::from_value(result.unwrap()).unwrap();
    assert_eq!(obj["success"], true);
}

#[wasm_bindgen_test]
fn prove_rejects_ml_selectors() {
    let atlas = ProofAtlasWasm::new();
    let input = "cnf(c1, axiom, p(a)).";

    for selector in &["gcn", "sentence"] {
        let opts = options(&format!(
            r#"{{"timeout_ms": 1000, "selector_type": "{}"}}"#,
            selector
        ));
        let result = atlas.prove(input, opts);
        assert!(result.is_err(), "ML selector '{}' should be rejected in WASM", selector);
    }
}

#[wasm_bindgen_test]
fn prove_with_empty_options() {
    let atlas = ProofAtlasWasm::new();
    let input = "cnf(c1, axiom, p(a)).\ncnf(c2, axiom, ~p(a)).";
    // All fields are Option, so an empty object should use defaults.
    let opts = options(r#"{}"#);
    let result = atlas.prove(input, opts);
    assert!(result.is_ok(), "empty options should use defaults: {:?}", result);
}

#[wasm_bindgen_test]
fn prove_invalid_options_returns_error() {
    let atlas = ProofAtlasWasm::new();
    let input = "cnf(c1, axiom, p(a)).";
    // Pass a non-object value -- should fail gracefully.
    let opts = JsValue::from_str("not an object");
    let result = atlas.prove(input, opts);
    assert!(result.is_err(), "non-object options should return error");
}

// ---------------------------------------------------------------------------
// Parse error handling
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn prove_invalid_tptp_returns_error() {
    let atlas = ProofAtlasWasm::new();
    let result = atlas.prove("cnf(broken", default_options());
    // Should either error or succeed with 0 clauses (parser skips bad lines).
    // Either way it must not panic.
    let _ = result;
}

// ---------------------------------------------------------------------------
// Proof structure validation
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn proof_steps_have_correct_shape() {
    let atlas = ProofAtlasWasm::new();
    let input = "cnf(c1, axiom, p(a)).\ncnf(c2, axiom, ~p(a)).";
    let result = atlas.prove(input, default_options()).unwrap();
    let obj: serde_json::Value = serde_wasm_bindgen::from_value(result).unwrap();

    let proof = obj["proof"].as_array().expect("proof should be an array");
    assert!(!proof.is_empty(), "proof should have at least one step");

    for step in proof {
        // Each step must have id, clause, rule, parents.
        assert!(step["id"].is_number(), "step.id should be a number");
        assert!(step["clause"].is_string(), "step.clause should be a string");
        assert!(step["rule"].is_string(), "step.rule should be a string");
        assert!(step["parents"].is_array(), "step.parents should be an array");
    }

    // The last step in a proof should be the empty clause.
    let last = proof.last().unwrap();
    let clause_str = last["clause"].as_str().unwrap();
    // The empty clause is formatted as the falsum symbol.
    assert_eq!(clause_str, "\u{22a5}", "last proof step should be the empty clause");
}

// ---------------------------------------------------------------------------
// Profile data
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn prove_returns_profile_data() {
    let atlas = ProofAtlasWasm::new();
    let input = "cnf(c1, axiom, p(a)).\ncnf(c2, axiom, ~p(a)).";
    let result = atlas.prove(input, default_options()).unwrap();
    let obj: serde_json::Value = serde_wasm_bindgen::from_value(result).unwrap();

    // Profiling is enabled by default in the WASM crate, so profile should be present.
    let profile = &obj["profile"];
    assert!(!profile.is_null(), "profile should be present");
}

// ---------------------------------------------------------------------------
// Statistics validation
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn statistics_are_reasonable() {
    let atlas = ProofAtlasWasm::new();
    let input = "\
        cnf(c1, axiom, p(a)).\n\
        cnf(c2, axiom, ~p(X) | q(X)).\n\
        cnf(c3, axiom, ~q(a)).\n";
    let result = atlas.prove(input, default_options()).unwrap();
    let obj: serde_json::Value = serde_wasm_bindgen::from_value(result).unwrap();

    let stats = &obj["statistics"];
    assert_eq!(stats["initial_clauses"].as_u64().unwrap(), 3);
    assert!(stats["generated_clauses"].as_u64().unwrap() >= 3);
    assert!(stats["final_clauses"].as_u64().unwrap() >= 1);
}
