use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use proofatlas::{
    parse_tptp, SaturationConfig, SaturationResult, SaturationState, LiteralSelectionStrategy,
    Clause, Literal, AgeWeightSelector, ClauseSelector, SaturationTrace,
    ForwardSimplification, BackwardSimplification, SimplificationOutcome, GeneratingInference,
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

        web_sys::console::log_1(&format!("Parsed {} clauses", cnf.clauses.len()).into());

        let initial_clauses = cnf.clauses.len();
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

        let config = SaturationConfig {
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
        let state = SaturationState::new(cnf.clauses, config, clause_selector);
        let (result, profile, sat_trace) = state.saturate();

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
                    rule: step.derivation.rule_name().to_string(),
                    parents: step.derivation.premises(),
                }
            }).collect()
        };

        // Build result
        let prover_result = match result {
            SaturationResult::Proof(proof) => {
                // Proof found
                let all_steps = convert_steps(&proof.steps);

                // All clauses: all proof steps (all are real derivations now)
                let all_clauses = all_steps.clone();

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
                    all_clauses: Some(all_clauses),
                    statistics: ProverStatistics {
                        initial_clauses,
                        generated_clauses: proof.steps.len(),
                        final_clauses: proof.steps.len(),
                        time_ms,
                    },
                    trace: if include_trace {
                        Some(trace_to_js_value(&sat_trace))
                    } else {
                        None
                    },
                    profile: profile.as_ref().and_then(|p| serde_json::to_value(p).ok()),
                }
            }
            SaturationResult::Saturated(steps, clauses) => {
                let all_steps = convert_steps(&steps);
                ProverResult {
                    success: false,
                    status: "saturated".to_string(),
                    message: "Saturated without finding a proof - the formula may be satisfiable".to_string(),
                    proof: None,
                    all_clauses: Some(all_steps),
                    statistics: ProverStatistics {
                        initial_clauses,
                        generated_clauses: steps.len(),
                        final_clauses: clauses.len(),
                        time_ms,
                    },
                    trace: if include_trace { Some(trace_to_js_value(&sat_trace)) } else { None },
                    profile: profile.as_ref().and_then(|p| serde_json::to_value(p).ok()),
                }
            }
            SaturationResult::Timeout(steps, clauses) => {
                let all_steps = convert_steps(&steps);
                ProverResult {
                    success: false,
                    status: "timeout".to_string(),
                    message: "Timeout reached before finding a proof".to_string(),
                    proof: None,
                    all_clauses: Some(all_steps),
                    statistics: ProverStatistics {
                        initial_clauses,
                        generated_clauses: steps.len(),
                        final_clauses: clauses.len(),
                        time_ms,
                    },
                    trace: if include_trace { Some(trace_to_js_value(&sat_trace)) } else { None },
                    profile: profile.as_ref().and_then(|p| serde_json::to_value(p).ok()),
                }
            }
            SaturationResult::ResourceLimit(steps, clauses) => {
                let all_steps = convert_steps(&steps);
                ProverResult {
                    success: false,
                    status: "resource_limit".to_string(),
                    message: "Resource limit reached".to_string(),
                    proof: None,
                    all_clauses: Some(all_steps),
                    statistics: ProverStatistics {
                        initial_clauses,
                        generated_clauses: steps.len(),
                        final_clauses: clauses.len(),
                        time_ms,
                    },
                    trace: if include_trace { Some(trace_to_js_value(&sat_trace)) } else { None },
                    profile: profile.as_ref().and_then(|p| serde_json::to_value(p).ok()),
                }
            }
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
            Ok(cnf) => Ok(format!("Valid TPTP input with {} clauses", cnf.clauses.len())),
            Err(e) => Err(JsError::new(&format!("Parse error: {}", e))),
        }
    }
}

/// Convert a SaturationTrace into the flat event format expected by the JS ProofInspector.
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
fn trace_to_js_value(trace: &SaturationTrace) -> serde_json::Value {
    use serde_json::json;

    // Build initial_clauses array
    let initial_clauses: Vec<serde_json::Value> = (0..trace.initial_clause_count)
        .map(|i| json!({
            "id": i,
            "clause": trace.clauses.get(i).cloned().unwrap_or_default(),
        }))
        .collect();

    // Build iterations
    let iterations: Vec<serde_json::Value> = trace.iterations.iter().map(|step| {
        let mut simplification_events = Vec::new();

        for cs in &step.simplifications {
            let clause_str = trace.clauses.get(cs.clause_idx).cloned().unwrap_or_default();

            match &cs.outcome {
                None => {
                    // Empty clause found — record as special event
                    simplification_events.push(json!({
                        "clause_idx": cs.clause_idx,
                        "clause": clause_str,
                        "rule": "EmptyClause",
                        "premises": [],
                    }));
                }
                Some(SimplificationOutcome::Forward(fwd)) => {
                    match fwd {
                        ForwardSimplification::Tautology => {
                            simplification_events.push(json!({
                                "clause_idx": cs.clause_idx,
                                "clause": clause_str,
                                "rule": "TautologyDeletion",
                                "premises": [],
                            }));
                        }
                        ForwardSimplification::Subsumption { subsumer } => {
                            simplification_events.push(json!({
                                "clause_idx": cs.clause_idx,
                                "clause": clause_str,
                                "rule": "ForwardSubsumptionDeletion",
                                "premises": [subsumer],
                            }));
                        }
                        ForwardSimplification::Demodulation { demodulator, result } => {
                            // The original clause is deleted; the result is a new clause in N
                            let result_str = trace.clauses.get(*result).cloned().unwrap_or_default();
                            simplification_events.push(json!({
                                "clause_idx": *result,
                                "clause": result_str,
                                "rule": "Demodulation",
                                "premises": [demodulator, cs.clause_idx],
                            }));
                        }
                    }
                }
                Some(SimplificationOutcome::Backward { effects }) => {
                    // Clause survived — emit backward effects first, then Transfer
                    for effect in effects {
                        match effect {
                            BackwardSimplification::Subsumption { deleted_clause } => {
                                let del_str = trace.clauses.get(*deleted_clause).cloned().unwrap_or_default();
                                simplification_events.push(json!({
                                    "clause_idx": *deleted_clause,
                                    "clause": del_str,
                                    "rule": "BackwardSubsumptionDeletion",
                                    "premises": [cs.clause_idx],
                                }));
                            }
                            BackwardSimplification::Demodulation { old_clause, result } => {
                                let result_str = trace.clauses.get(*result).cloned().unwrap_or_default();
                                simplification_events.push(json!({
                                    "clause_idx": *result,
                                    "clause": result_str,
                                    "rule": "Demodulation",
                                    "premises": [cs.clause_idx, old_clause],
                                }));
                            }
                        }
                    }
                    // Transfer event
                    simplification_events.push(json!({
                        "clause_idx": cs.clause_idx,
                        "clause": clause_str,
                        "rule": "Transfer",
                        "premises": [],
                    }));
                }
            }
        }

        // Selection
        let selection = step.given_clause.map(|idx| {
            let clause_str = trace.clauses.get(idx).cloned().unwrap_or_default();
            json!({
                "clause_idx": idx,
                "clause": clause_str,
                "rule": "GivenClauseSelection",
            })
        });

        // Generation
        let generation_events: Vec<serde_json::Value> = step.generating_inferences.iter().map(|gi| {
            let (clause_idx, rule, premises) = match gi {
                GeneratingInference::Resolution { clause_idx, parents } => {
                    (*clause_idx, "Resolution", vec![parents.0, parents.1])
                }
                GeneratingInference::Factoring { clause_idx, parent } => {
                    (*clause_idx, "Factoring", vec![*parent])
                }
                GeneratingInference::Superposition { clause_idx, parents } => {
                    (*clause_idx, "Superposition", vec![parents.0, parents.1])
                }
                GeneratingInference::EqualityResolution { clause_idx, parent } => {
                    (*clause_idx, "EqualityResolution", vec![*parent])
                }
                GeneratingInference::EqualityFactoring { clause_idx, parent } => {
                    (*clause_idx, "EqualityFactoring", vec![*parent])
                }
            };
            let clause_str = trace.clauses.get(clause_idx).cloned().unwrap_or_default();
            json!({
                "clause_idx": clause_idx,
                "clause": clause_str,
                "rule": rule,
                "premises": premises,
            })
        }).collect();

        json!({
            "simplification": simplification_events,
            "selection": selection,
            "generation": generation_events,
        })
    }).collect();

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
    if lit.polarity {
        lit.atom.to_string()
    } else {
        format!("~{}", lit.atom)
    }
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
