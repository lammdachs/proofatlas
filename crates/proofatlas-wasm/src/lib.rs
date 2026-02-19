use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use serde::{Serialize, Deserialize};
use proofatlas::{
    parse_tptp, ProverConfig, ProofResult, Prover, LiteralSelectionStrategy,
    AgeWeightSink, ProverSink,
    build_trace, steps_to_wire, status_message,
    ProveResult as CoreProveResult, ProveStatistics,
};
use std::time::Duration;

/// Get current time in milliseconds via `performance.now()`.
/// Works in both Window and Web Worker contexts by using `js_sys::global()`.
fn performance_now() -> f64 {
    js_sys::Reflect::get(&js_sys::global(), &"performance".into())
        .ok()
        .and_then(|perf| {
            let perf: web_sys::Performance = perf.dyn_into().ok()?;
            Some(perf.now())
        })
        .unwrap_or(0.0)
}

#[wasm_bindgen]
pub struct ProofAtlasWasm;

#[derive(Serialize, Deserialize)]
pub struct ProverOptions {
    pub timeout_ms: Option<u32>,
    pub literal_selection: Option<String>, // "0", "20", "21", or "22" (Vampire-compatible numbering)
    pub selector_type: Option<String>,     // "age_weight", "gcn", or "mlp" (default: "age_weight")
    pub age_weight_ratio: Option<f64>,     // Age probability for age_weight selector (default: 0.5)
    pub max_iterations: Option<usize>,     // Max saturation iterations (default: 10000)
}


#[wasm_bindgen]
impl ProofAtlasWasm {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        ProofAtlasWasm
    }

    #[wasm_bindgen]
    pub fn prove_with_trace(&self, tptp_input: &str, options_js: JsValue) -> Result<JsValue, JsError> {
        // This method will return detailed proof trace with saturation steps
        self.prove_internal(tptp_input, options_js, true)
    }

    #[wasm_bindgen]
    pub fn prove(&self, tptp_input: &str, options_js: JsValue) -> Result<JsValue, JsError> {
        self.prove_internal(tptp_input, options_js, false)
    }

    fn prove_internal(&self, tptp_input: &str, options_js: JsValue, include_trace: bool) -> Result<JsValue, JsError> {
        // Parse options
        let options: ProverOptions = serde_wasm_bindgen::from_value(options_js)
            .map_err(|e| JsError::new(&format!("Invalid options: {}", e)))?;

        web_sys::console::log_1(&format!("Options parsed: timeout_ms={:?}",
            options.timeout_ms).into());

        // Parse TPTP input
        let cnf = parse_tptp(tptp_input, &[], None, None)
            .map_err(|e| JsError::new(&format!("Parse error: {}", e)))?;

        web_sys::console::log_1(&format!("Parsed {} clauses", cnf.formula.clauses.len()).into());

        let initial_clauses = cnf.formula.clauses.len();
        let start_time = performance_now();

        // Create saturation config with configurable literal selection
        let literal_selection = match options.literal_selection.as_deref() {
            Some("20") => LiteralSelectionStrategy::Sel20,
            Some("21") => LiteralSelectionStrategy::Sel21,
            Some("22") => LiteralSelectionStrategy::Sel22,
            _ => LiteralSelectionStrategy::Sel0, // Default to Sel0 (all)
        };

        let config = ProverConfig {
            max_clauses: 0,
            max_iterations: options.max_iterations.unwrap_or(0),
            max_clause_size: 100,
            timeout: Duration::from_millis(options.timeout_ms.unwrap_or(60000) as u64),
            literal_selection,
            memory_limit: None,
            enable_profiling: true,
        };

        web_sys::console::log_1(&"Config created, creating clause selector...".into());

        // Create clause selection sink based on options
        // Note: ML selectors (gcn, sentence) require libtorch which is not available in WASM
        let sink: Box<dyn ProverSink> = match options.selector_type.as_deref() {
            Some("gcn") | Some("sentence") => {
                return Err(JsError::new("ML selectors (gcn, sentence) are not supported in WASM. Use age_weight instead."));
            }
            _ => {
                // Default to age_weight selector (no model needed)
                let ratio = options.age_weight_ratio.unwrap_or(0.5);
                web_sys::console::log_1(&format!("Using AgeWeight selector with ratio {}", ratio).into());
                Box::new(AgeWeightSink::new(ratio))
            }
        };

        // Run saturation
        let mut prover = Prover::new(cnf.formula.clauses, config, sink, cnf.interner);
        let result = prover.prove();

        web_sys::console::log_1(&"Saturation completed".into());

        let end_time = performance_now();

        let time_ms = (end_time - start_time) as u32;

        let interner = prover.interner();
        let events = prover.event_log();
        let profile = prover.profile();

        let (status, msg) = status_message(&result);
        let (success, proof, message, final_clauses) = match result {
            ProofResult::Proof { empty_clause_idx } => {
                let steps = prover.extract_proof(empty_clause_idx);
                let wire = steps_to_wire(&steps, interner);
                let message = format!("Proof found with {} steps", wire.len());
                let final_count = steps.len();
                (true, Some(wire), message, final_count)
            }
            _ => (false, None, msg.to_string(), prover.clauses().len()),
        };

        let prover_result = CoreProveResult {
            success,
            status: status.to_string(),
            message,
            proof,
            all_clauses: None,
            statistics: ProveStatistics {
                initial_clauses,
                generated_clauses: prover.clauses().len(),
                final_clauses,
                time_ms,
            },
            trace: if include_trace {
                Some(build_trace(events, |c| c.display(interner).to_string()))
            } else {
                None
            },
            profile: profile.and_then(|p| serde_json::to_value(p).ok()),
        };

        let serializer = serde_wasm_bindgen::Serializer::json_compatible();
        use serde::Serialize;
        prover_result.serialize(&serializer)
            .map_err(|e| JsError::new(&format!("Serialization error: {}", e)))
    }

    #[wasm_bindgen]
    pub fn validate_tptp(&self, input: &str) -> Result<String, JsError> {
        // Just parse and return success/error for validation
        match parse_tptp(input, &[], None, None) {
            Ok(cnf) => Ok(format!("Valid TPTP input with {} clauses", cnf.formula.clauses.len())),
            Err(e) => Err(JsError::new(&format!("Parse error: {}", e))),
        }
    }
}


// Required for wasm-bindgen
#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
}
