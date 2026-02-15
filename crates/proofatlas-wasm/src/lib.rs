use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use serde::{Serialize, Deserialize};
use proofatlas::{
    parse_tptp, ProverConfig, ProofResult, Prover, LiteralSelectionStrategy,
    Clause, Literal, Interner, AgeWeightSink, ProverSink, StateChange,
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
        let sat_trace = prover.event_log();
        let profile = prover.profile();

        // Helper to convert proof steps to WASM ProofStep
        let convert_steps = |steps: &[proofatlas::ProofStep], interner: &proofatlas::Interner| -> Vec<ProofStep> {
            steps.iter().map(|step| {
                ProofStep {
                    id: step.clause_idx,
                    clause: format_clause(&step.conclusion, interner),
                    rule: step.rule_name.clone(),
                    parents: proofatlas::clause_indices(&step.premises),
                }
            }).collect()
        };

        // Build result
        let prover_result = match result {
            ProofResult::Proof { empty_clause_idx } => {
                // Proof found — extract proof steps by backward traversal
                let proof_steps = prover.extract_proof(empty_clause_idx);
                let proof_path = convert_steps(&proof_steps, interner);

                ProverResult {
                    success: true,
                    status: "proof_found".to_string(),
                    message: format!("Proof found with {} steps", proof_path.len()),
                    proof: Some(proof_path),
                    statistics: ProverStatistics {
                        initial_clauses,
                        generated_clauses: prover.clauses().len(),
                        final_clauses: proof_steps.len(),
                        time_ms,
                    },
                    trace: if include_trace { Some(events_to_js_value(sat_trace, interner)) } else { None },
                    profile: profile.and_then(|p| serde_json::to_value(p).ok()),
                }
            }
            ProofResult::Saturated | ProofResult::ResourceLimit => {
                let (status, message) = match result {
                    ProofResult::Saturated => ("saturated", "Saturated without finding a proof - the formula may be satisfiable"),
                    _ => ("resource_limit", "Resource limit reached"),
                };
                let total_clauses = prover.clauses().len();
                ProverResult {
                    success: false,
                    status: status.to_string(),
                    message: message.to_string(),
                    proof: None,
                    statistics: ProverStatistics {
                        initial_clauses,
                        generated_clauses: total_clauses,
                        final_clauses: total_clauses,
                        time_ms,
                    },
                    trace: if include_trace { Some(events_to_js_value(sat_trace, interner)) } else { None },
                    profile: profile.and_then(|p| serde_json::to_value(p).ok()),
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
        match parse_tptp(input, &[], None, None) {
            Ok(cnf) => Ok(format!("Valid TPTP input with {} clauses", cnf.formula.clauses.len())),
            Err(e) => Err(JsError::new(&format!("Parse error: {}", e))),
        }
    }
}

/// Convert an event log slice into the flat event format expected by the JS ProofInspector.
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
fn events_to_js_value(events: &[StateChange], interner: &Interner) -> serde_json::Value {
    use serde_json::json;
    use std::collections::HashMap;

    // First pass: collect all clauses and their derivations
    let mut clauses: HashMap<usize, String> = HashMap::new();
    let mut derivations: HashMap<usize, (String, Vec<usize>)> = HashMap::new();
    let mut initial_clause_count = 0;

    for event in events {
        match event {
            StateChange::Add(clause, rule_name, premises) => {
                if let Some(idx) = clause.id {
                    clauses.insert(idx, format_clause(clause, interner));
                    derivations.insert(idx, (rule_name.clone(), proofatlas::clause_indices(premises)));
                    if rule_name == "Input" {
                        initial_clause_count = initial_clause_count.max(idx + 1);
                    }
                }
            }
            StateChange::Simplify(_, Some(clause), rule_name, premises) => {
                if let Some(idx) = clause.id {
                    clauses.insert(idx, format_clause(clause, interner));
                    derivations.insert(idx, (rule_name.clone(), proofatlas::clause_indices(premises)));
                }
            }
            _ => {}
        }
    }

    // Build initial_clauses array
    let initial_clauses: Vec<serde_json::Value> = (0..initial_clause_count)
        .map(|i| json!({
            "id": i,
            "clause": clauses.get(&i).cloned().unwrap_or_default(),
        }))
        .collect();

    // Second pass: build iterations using a phase-aware state machine.
    //
    // The prover emits events per iteration in this order:
    //   1. Simplification (Simplify on N, backward Simplify on U/P)
    //   2. Transfer (N → U)
    //   3. Activate (U → P) — given clause selection
    //   4. Generation (Add "Resolution", "Superposition", etc.)
    //
    // We track two phases: SIMPLIFICATION (before Activate) and GENERATION (after).
    // A non-generating event during GENERATION means the next iteration has started.
    let mut iterations: Vec<serde_json::Value> = Vec::new();
    let mut current_simplification = Vec::new();
    let mut current_generation = Vec::new();
    let mut current_selection: Option<serde_json::Value> = None;
    let mut in_generation_phase = false;

    // Helper: flush the current iteration
    let flush = |iterations: &mut Vec<serde_json::Value>,
                 simplification: &mut Vec<serde_json::Value>,
                 selection: &mut Option<serde_json::Value>,
                 generation: &mut Vec<serde_json::Value>| {
        if selection.is_some() || !simplification.is_empty() || !generation.is_empty() {
            iterations.push(json!({
                "simplification": std::mem::take(simplification),
                "selection": selection.take(),
                "generation": std::mem::take(generation),
            }));
        }
    };

    for event in events {
        match event {
            StateChange::Add(clause, rule_name, premises) => {
                if let Some(idx) = clause.id {
                    let clause_str = format_clause(clause, interner);
                    let premise_indices = proofatlas::clause_indices(premises);

                    if rule_name == "Input" {
                        // Show input clause additions in the simplification phase
                        // (before any given clause selection)
                        current_simplification.push(json!({
                            "clause_idx": idx,
                            "clause": clause_str,
                            "rule": "Input",
                            "premises": premise_indices,
                        }));
                    } else {
                        // All other Add events are generating inferences
                        current_generation.push(json!({
                            "clause_idx": idx,
                            "clause": clause_str,
                            "rule": rule_name,
                            "premises": premise_indices,
                        }));
                    }
                }
            }
            StateChange::Simplify(clause_idx, replacement, rule_name, premises) => {
                // If we're in generation phase, a Simplify means next iteration started
                if in_generation_phase {
                    flush(&mut iterations, &mut current_simplification, &mut current_selection, &mut current_generation);
                    in_generation_phase = false;
                }
                let clause_str = clauses.get(clause_idx).cloned().unwrap_or_default();
                let premise_indices = proofatlas::clause_indices(premises);

                if let Some(repl) = replacement {
                    // Replacement (demodulation): emit deletion then add
                    let repl_idx = repl.id.unwrap_or(0);
                    let repl_str = format_clause(repl, interner);
                    // Store replacement clause text for later lookups
                    clauses.insert(repl_idx, repl_str.clone());

                    current_simplification.push(json!({
                        "clause_idx": repl_idx,
                        "clause": repl_str,
                        "rule": rule_name,
                        "premises": premise_indices,
                    }));
                    current_simplification.push(json!({
                        "clause_idx": *clause_idx,
                        "clause": clause_str,
                        "rule": format!("{}Deletion", rule_name),
                        "premises": premise_indices,
                        "replacement_idx": repl_idx,
                    }));
                } else {
                    // Pure deletion (tautology, subsumption)
                    let rule = match rule_name.as_str() {
                        "Tautology" => "TautologyDeletion",
                        "Subsumption" => "SubsumptionDeletion",
                        _ => "SubsumptionDeletion",
                    };
                    current_simplification.push(json!({
                        "clause_idx": *clause_idx,
                        "clause": clause_str,
                        "rule": rule,
                        "premises": [],
                    }));
                }
            }
            StateChange::Transfer(clause_idx) => {
                // Transfer is simplification phase — if in generation, flush first
                if in_generation_phase {
                    flush(&mut iterations, &mut current_simplification, &mut current_selection, &mut current_generation);
                    in_generation_phase = false;
                }
                let clause_str = clauses.get(clause_idx).cloned().unwrap_or_default();
                current_simplification.push(json!({
                    "clause_idx": *clause_idx,
                    "clause": clause_str,
                    "rule": "Transfer",
                    "premises": [],
                }));
            }
            StateChange::Activate(clause_idx) => {
                // Activate marks selection — if we're already in generation phase,
                // flush the previous iteration first
                if in_generation_phase {
                    flush(&mut iterations, &mut current_simplification, &mut current_selection, &mut current_generation);
                }
                let clause_str = clauses.get(clause_idx).cloned().unwrap_or_default();
                current_selection = Some(json!({
                    "clause_idx": *clause_idx,
                    "clause": clause_str,
                    "rule": "GivenClauseSelection",
                    "premises": [],
                }));
                in_generation_phase = true;
            }
        }
    }

    // Flush any remaining events
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

fn format_clause(clause: &Clause, interner: &Interner) -> String {
    if clause.literals.is_empty() {
        "⊥".to_string()
    } else {
        clause.literals.iter()
            .map(|lit| format_literal(lit, interner))
            .collect::<Vec<_>>()
            .join(" ∨ ")
    }
}

fn format_literal(lit: &Literal, interner: &Interner) -> String {
    lit.display(interner).to_string()
}

// Required for wasm-bindgen
#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
}
