use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use proofatlas::{
    parse_tptp, ProverConfig, ProofResult, ProofAtlas, LiteralSelectionStrategy,
    Clause, Literal, AgeWeightSelector, ClauseSelector, EventLog, StateChange,
};
use std::time::Duration;

#[wasm_bindgen]
pub struct ProofAtlasWasm;

#[derive(Serialize, Deserialize)]
pub struct ProverOptions {
    pub timeout_ms: Option<u32>,
    pub literal_selection: Option<String>, // "0", "20", "21", or "22" (Vampire-compatible numbering)
    pub selector_type: Option<String>,     // "age_weight", "gcn", or "mlp" (default: "age_weight")
    pub selector_weights: Option<Vec<u8>>, // Safetensors weights for ML selectors (optional)
    pub age_weight_ratio: Option<f64>,     // Age probability for age_weight selector (default: 0.167)
    pub max_iterations: Option<usize>,     // Max saturation iterations (default: 10000)
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ProofStep {
    pub id: usize,
    pub clause: String,
    pub rule: String,
    pub parents: Vec<usize>,
}

#[derive(Serialize, Deserialize)]
pub struct ProverResult {
    pub success: bool,
    pub status: String, // "proof_found", "saturated", "timeout", "error"
    pub message: String,
    pub proof: Option<Vec<ProofStep>>,
    pub all_clauses: Option<Vec<ProofStep>>, // All generated clauses
    pub statistics: ProverStatistics,
    pub trace: Option<serde_json::Value>, // Structured SaturationTrace as JSON
    pub profile: Option<serde_json::Value>, // Profiling data
}

#[derive(Serialize, Deserialize)]
pub struct ProverStatistics {
    pub initial_clauses: usize,
    pub generated_clauses: usize,
    pub final_clauses: usize,
    pub time_ms: u32,
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
        let cnf = parse_tptp(tptp_input, &[], None)
            .map_err(|e| JsError::new(&format!("Parse error: {}", e)))?;

        web_sys::console::log_1(&format!("Parsed {} clauses", cnf.formula.clauses.len()).into());

        let initial_clauses = cnf.formula.clauses.len();
        let start_time = web_sys::window()
            .unwrap()
            .performance()
            .unwrap()
            .now();

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
            max_clause_memory_mb: None,
            enable_profiling: true,
        };

        web_sys::console::log_1(&"Config created, creating clause selector...".into());

        // Create clause selector based on options
        // Note: ML selectors (gcn, sentence) require libtorch which is not available in WASM
        let clause_selector: Box<dyn ClauseSelector> = match options.selector_type.as_deref() {
            Some("gcn") | Some("sentence") => {
                return Err(JsError::new("ML selectors (gcn, sentence) are not supported in WASM. Use age_weight instead."));
            }
            _ => {
                // Default to age_weight selector (no model needed)
                let ratio = options.age_weight_ratio.unwrap_or(0.167);
                web_sys::console::log_1(&format!("Using AgeWeight selector with ratio {}", ratio).into());
                Box::new(AgeWeightSelector::new(ratio))
            }
        };

        // Run saturation
        let prover = ProofAtlas::new(cnf.formula.clauses, config, clause_selector, cnf.interner);
        let (result, profile, sat_trace, _interner) = prover.prove();

        web_sys::console::log_1(&"Saturation completed".into());

        let end_time = web_sys::window()
            .unwrap()
            .performance()
            .unwrap()
            .now();

        let time_ms = (end_time - start_time) as u32;

        // Helper to convert proof steps to WASM ProofStep
        let convert_steps = |steps: &[proofatlas::ProofStep]| -> Vec<ProofStep> {
            steps.iter().map(|step| {
                ProofStep {
                    id: step.clause_idx,
                    clause: format_clause(&step.conclusion),
                    rule: step.derivation.rule_name.clone(),
                    parents: step.derivation.clause_indices(),
                }
            }).collect()
        };

        // Helper to build result for non-proof cases
        let build_incomplete_result = |status: &str, message: &str, steps: &[proofatlas::ProofStep], final_clauses_count: usize| {
            let all_steps = convert_steps(steps);
            ProverResult {
                success: false,
                status: status.to_string(),
                message: message.to_string(),
                proof: None,
                all_clauses: Some(all_steps),
                statistics: ProverStatistics {
                    initial_clauses,
                    generated_clauses: steps.len(),
                    final_clauses: final_clauses_count,
                    time_ms,
                },
                trace: if include_trace { Some(events_to_js_value(&sat_trace)) } else { None },
                profile: profile.as_ref().and_then(|p| serde_json::to_value(p).ok()),
            }
        };

        // Build result
        let prover_result = match result {
            ProofResult::Proof(proof) => {
                // Proof found
                let all_steps = convert_steps(&proof.steps);

                // Extract the proof path - trace back from empty clause
                let mut proof_indices = std::collections::HashSet::new();
                let mut to_visit = vec![proof.empty_clause_idx];

                // Build index mapping: step.id -> step
                let id_to_step: std::collections::HashMap<usize, &ProofStep> =
                    all_steps.iter().map(|s| (s.id, s)).collect();

                while let Some(current_id) = to_visit.pop() {
                    if proof_indices.insert(current_id) {
                        if let Some(step) = id_to_step.get(&current_id) {
                            for &parent_id in &step.parents {
                                if !proof_indices.contains(&parent_id) {
                                    to_visit.push(parent_id);
                                }
                            }
                        }
                    }
                }

                // Build proof path
                let mut proof_path: Vec<ProofStep> = all_steps.iter()
                    .filter(|step| proof_indices.contains(&step.id))
                    .cloned()
                    .collect();
                proof_path.sort_by_key(|step| step.id);

                ProverResult {
                    success: true,
                    status: "proof_found".to_string(),
                    message: format!("Proof found with {} steps", proof_path.len()),
                    proof: Some(proof_path),
                    all_clauses: Some(all_steps),
                    statistics: ProverStatistics {
                        initial_clauses,
                        generated_clauses: proof.steps.len(),
                        final_clauses: proof.steps.len(),
                        time_ms,
                    },
                    trace: if include_trace { Some(events_to_js_value(&sat_trace)) } else { None },
                    profile: profile.as_ref().and_then(|p| serde_json::to_value(p).ok()),
                }
            }
            ProofResult::Saturated(steps, clauses) => build_incomplete_result(
                "saturated",
                "Saturated without finding a proof - the formula may be satisfiable",
                &steps,
                clauses.len(),
            ),
            ProofResult::Timeout(steps, clauses) => build_incomplete_result(
                "timeout",
                "Timeout reached before finding a proof",
                &steps,
                clauses.len(),
            ),
            ProofResult::ResourceLimit(steps, clauses) => build_incomplete_result(
                "resource_limit",
                "Resource limit reached",
                &steps,
                clauses.len(),
            ),
        };

        // Convert to JS value (use json_compatible to serialize Maps as plain objects)
        let serializer = serde_wasm_bindgen::Serializer::json_compatible();
        use serde::Serialize;
        prover_result.serialize(&serializer)
            .map_err(|e| JsError::new(&format!("Serialization error: {}", e)))
    }

    #[wasm_bindgen]
    pub fn validate_tptp(&self, input: &str) -> Result<String, JsError> {
        // Just parse and return success/error for validation
        match parse_tptp(input, &[], None) {
            Ok(cnf) => Ok(format!("Valid TPTP input with {} clauses", cnf.formula.clauses.len())),
            Err(e) => Err(JsError::new(&format!("Parse error: {}", e))),
        }
    }
}

/// Convert a EventLog into the flat event format expected by the JS ProofInspector.
///
/// This replays the event log to reconstruct the iteration structure that the JS UI expects.
///
/// Output format:
/// ```json
/// {
///   "initial_clauses": [{"id": 0, "clause": "..."}, ...],
///   "iterations": [{
///     "simplification": [{"clause_idx": 0, "clause": "...", "rule": "Transfer", "premises": []}],
///     "selection": {"clause_idx": 0, "clause": "...", "rule": "GivenClauseSelection"} | null,
///     "generation": [{"clause_idx": 6, "clause": "...", "rule": "Resolution", "premises": [0, 1]}]
///   }]
/// }
/// ```
fn events_to_js_value(events: &EventLog) -> serde_json::Value {
    use serde_json::json;
    use std::collections::HashMap;

    // First pass: collect all clauses and their derivations
    let mut clauses: HashMap<usize, String> = HashMap::new();
    let mut derivations: HashMap<usize, (String, Vec<usize>)> = HashMap::new();
    let mut initial_clause_count = 0;

    for event in events {
        if let StateChange::Add { clause, derivation } = event {
            if let Some(idx) = clause.id {
                clauses.insert(idx, format_clause(clause));
                derivations.insert(idx, (derivation.rule_name.clone(), derivation.clause_indices()));
                if derivation.rule_name == "Input" {
                    initial_clause_count = initial_clause_count.max(idx + 1);
                }
            }
        }
    }

    // Build initial_clauses array
    let initial_clauses: Vec<serde_json::Value> = (0..initial_clause_count)
        .map(|i| json!({
            "id": i,
            "clause": clauses.get(&i).cloned().unwrap_or_default(),
        }))
        .collect();

    // Second pass: build iterations by replaying events
    // An iteration ends when we see AddP (given clause selected)
    let mut iterations: Vec<serde_json::Value> = Vec::new();
    let mut current_simplification = Vec::new();
    let mut current_generation = Vec::new();
    let mut current_selection: Option<serde_json::Value> = None;

    for event in events {
        match event {
            StateChange::Add { clause, derivation } => {
                if let Some(idx) = clause.id {
                    let clause_str = format_clause(clause);
                    let rule = &derivation.rule_name;
                    let premises = derivation.clause_indices();

                    // Skip initial input clauses for iteration events
                    if rule == "Input" {
                        continue;
                    }

                    // Generating inferences produce new clauses
                    let is_generating = matches!(
                        rule.as_str(),
                        "Resolution" | "Factoring" | "Superposition" | "EqualityResolution" | "EqualityFactoring"
                    );

                    if is_generating {
                        current_generation.push(json!({
                            "clause_idx": idx,
                            "clause": clause_str,
                            "rule": rule,
                            "premises": premises,
                        }));
                    } else if rule == "Demodulation" {
                        // Demodulation is a simplification
                        current_simplification.push(json!({
                            "clause_idx": idx,
                            "clause": clause_str,
                            "rule": "Demodulation",
                            "premises": premises,
                        }));
                    }
                }
            }
            StateChange::Delete { clause_idx, rule_name } => {
                let clause_str = clauses.get(clause_idx).cloned().unwrap_or_default();
                let rule = match rule_name.as_str() {
                    "Tautology" => "TautologyDeletion",
                    "Subsumption" => "ForwardSubsumptionDeletion",
                    "Demodulation" => "ForwardDemodulation",
                    _ => "BackwardSubsumptionDeletion",
                };
                current_simplification.push(json!({
                    "clause_idx": *clause_idx,
                    "clause": clause_str,
                    "rule": rule,
                    "premises": [],
                }));
            }
            StateChange::Transfer { clause_idx } => {
                // Clause transferred from N to U
                let clause_str = clauses.get(clause_idx).cloned().unwrap_or_default();
                current_simplification.push(json!({
                    "clause_idx": *clause_idx,
                    "clause": clause_str,
                    "rule": "Transfer",
                    "premises": [],
                }));
            }
            StateChange::Activate { clause_idx } => {
                // Given clause selection and end of iteration
                let clause_str = clauses.get(clause_idx).cloned().unwrap_or_default();
                current_selection = Some(json!({
                    "clause_idx": *clause_idx,
                    "clause": clause_str,
                    "rule": "GivenClauseSelection",
                }));
                // Flush current state
                if current_selection.is_some() || !current_simplification.is_empty() || !current_generation.is_empty() {
                    iterations.push(json!({
                        "simplification": std::mem::take(&mut current_simplification),
                        "selection": current_selection.take(),
                        "generation": std::mem::take(&mut current_generation),
                    }));
                }
            }
        }
    }

    // Flush any remaining events (e.g., final iteration before proof found)
    if !current_simplification.is_empty() || !current_generation.is_empty() || current_selection.is_some() {
        iterations.push(json!({
            "simplification": current_simplification,
            "selection": current_selection,
            "generation": current_generation,
        }));
    }

    json!({
        "initial_clauses": initial_clauses,
        "iterations": iterations,
    })
}

fn format_clause(clause: &Clause) -> String {
    if clause.literals.is_empty() {
        "⊥".to_string()
    } else {
        clause.literals.iter()
            .map(|lit| format_literal(lit))
            .collect::<Vec<_>>()
            .join(" ∨ ")
    }
}

fn format_literal(lit: &Literal) -> String {
    lit.to_string()
}

// Required for wasm-bindgen
#[wasm_bindgen(start)]
pub fn main() {
    // Set panic hook for better error messages
    console_error_panic_hook::set_once();

    // Also set up better panic handling
    std::panic::set_hook(Box::new(|info| {
        console_error_panic_hook::hook(info);
    }));
}
