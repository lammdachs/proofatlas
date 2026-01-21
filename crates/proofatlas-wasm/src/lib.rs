use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use proofatlas::{parse_tptp, SaturationConfig, SaturationResult, SaturationState, LiteralSelectionStrategy, Clause, Literal, AgeWeightSelector, ClauseSelector};
use std::time::Duration;

#[wasm_bindgen]
pub struct ProofAtlasWasm;

#[derive(Serialize, Deserialize)]
pub struct ProverOptions {
    pub timeout_ms: u32,
    pub max_clauses: usize,
    pub literal_selection: Option<String>, // "0", "20", "21", or "22" (Vampire-compatible numbering)
    pub selector_type: Option<String>,     // "age_weight", "gcn", or "mlp" (default: "age_weight")
    pub selector_weights: Option<Vec<u8>>, // Safetensors weights for ML selectors (optional)
    pub age_weight_ratio: Option<f64>,     // Age probability for age_weight selector (default: 0.167)
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
    pub trace: Option<ProofTrace>, // Detailed saturation trace
}

#[derive(Serialize, Deserialize)]
pub struct ProofTrace {
    pub initial_clauses: Vec<ProofStep>,
    pub saturation_steps: Vec<SaturationStep>,
}

#[derive(Serialize, Deserialize)]
pub struct SaturationStep {
    pub step_type: String, // "given_selection" or "inference"
    pub clause_idx: usize,
    pub clause: String,
    pub rule: String,
    pub premises: Vec<usize>,
    pub processed_count: usize,
    pub unprocessed_count: usize,
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
        
        web_sys::console::log_1(&format!("Options parsed: timeout_ms={}, max_clauses={}",
            options.timeout_ms, options.max_clauses).into());
        
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
            max_clauses: options.max_clauses,
            max_iterations: 10000,
            max_clause_size: 100,
            timeout: Duration::from_millis(options.timeout_ms as u64),
            literal_selection,
            max_clause_memory_mb: None,
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
        let result = state.saturate();
        
        web_sys::console::log_1(&"Saturation completed".into());
        
        let end_time = web_sys::window()
            .unwrap()
            .performance()
            .unwrap()
            .now();
        
        let time_ms = (end_time - start_time) as u32;
        
        // Helper to convert proof steps
        let convert_steps = |steps: &[proofatlas::ProofStep]| -> Vec<ProofStep> {
            steps.iter().map(|step| ProofStep {
                id: step.clause_idx,
                clause: format_clause(&step.inference.conclusion),
                rule: format!("{:?}", step.inference.rule),
                parents: step.inference.premises.clone(),
            }).collect()
        };
        
        // Build result
        let prover_result = match result {
            SaturationResult::Proof(proof) => {
                // Proof found
                let all_steps = convert_steps(&proof.steps);
                
                // For "All Clauses" view: filter out GivenClauseSelection (keep Input and inference clauses)
                let all_clauses: Vec<ProofStep> = all_steps.iter()
                    .filter(|step| step.rule != "GivenClauseSelection")
                    .cloned()
                    .collect();
                
                // Build index mapping: step.id -> position in all_steps
                // Only include non-GivenClauseSelection steps to avoid duplicates
                let mut id_to_pos: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
                for (pos, step) in all_steps.iter().enumerate() {
                    if step.rule != "GivenClauseSelection" {
                        id_to_pos.insert(step.id, pos);
                    }
                }
                
                // Extract the proof path - trace back from empty clause
                let mut proof_indices = std::collections::HashSet::new();
                let mut to_visit = vec![proof.empty_clause_idx];
                
                // Collect all clause IDs that contribute to the proof
                while let Some(current_id) = to_visit.pop() {
                    if proof_indices.insert(current_id) {
                        // Find this clause in all_steps by ID
                        if let Some(&pos) = id_to_pos.get(&current_id) {
                            let step = &all_steps[pos];
                            // Add parents to visit (parents are clause IDs, not positions)
                            for &parent_id in &step.parents {
                                if !proof_indices.contains(&parent_id) {
                                    to_visit.push(parent_id);
                                }
                            }
                        }
                    }
                }
                
                // Build proof path: include all steps that contribute to the proof
                // but filter out GivenClauseSelection (which is just bookkeeping)
                let mut proof_path: Vec<ProofStep> = all_steps.iter()
                    .filter(|step| {
                        proof_indices.contains(&step.id) && 
                        step.rule != "GivenClauseSelection"
                    })
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
                        Some(build_trace(&proof.steps, initial_clauses))
                    } else {
                        None
                    },
                }
            }
            SaturationResult::Saturated(steps, _) => {
                // Saturated without proof
                let all_steps = convert_steps(&steps);
                // Filter out GivenClauseSelection for all_clauses view
                let all_clauses: Vec<ProofStep> = all_steps.iter()
                    .filter(|step| step.rule != "GivenClauseSelection")
                    .cloned()
                    .collect();
                ProverResult {
                    success: false,
                    status: "saturated".to_string(),
                    message: "Saturated without finding a proof - the formula may be satisfiable".to_string(),
                    proof: None,
                    all_clauses: Some(all_clauses),
                    statistics: ProverStatistics {
                        initial_clauses,
                        generated_clauses: steps.len(),
                        final_clauses: steps.len(),
                        time_ms,
                    },
                    trace: if include_trace {
                        Some(build_trace(&steps, initial_clauses))
                    } else {
                        None
                    },
                }
            }
            SaturationResult::Timeout(steps, _) => {
                // Timeout
                let all_steps = convert_steps(&steps);
                // Filter out GivenClauseSelection for all_clauses view
                let all_clauses: Vec<ProofStep> = all_steps.iter()
                    .filter(|step| step.rule != "GivenClauseSelection")
                    .cloned()
                    .collect();
                ProverResult {
                    success: false,
                    status: "timeout".to_string(),
                    message: "Timeout reached before finding a proof".to_string(),
                    proof: None,
                    all_clauses: Some(all_clauses),
                    statistics: ProverStatistics {
                        initial_clauses,
                        generated_clauses: steps.len(),
                        final_clauses: steps.len(),
                        time_ms,
                    },
                    trace: if include_trace {
                        Some(build_trace(&steps, initial_clauses))
                    } else {
                        None
                    },
                }
            }
            SaturationResult::ResourceLimit(steps, _) => {
                // Resource limit
                let all_steps = convert_steps(&steps);
                // Filter out GivenClauseSelection for all_clauses view
                let all_clauses: Vec<ProofStep> = all_steps.iter()
                    .filter(|step| step.rule != "GivenClauseSelection")
                    .cloned()
                    .collect();
                ProverResult {
                    success: false,
                    status: "resource_limit".to_string(),
                    message: "Resource limit reached".to_string(),
                    proof: None,
                    all_clauses: Some(all_clauses),
                    statistics: ProverStatistics {
                        initial_clauses,
                        generated_clauses: steps.len(),
                        final_clauses: steps.len(),
                        time_ms,
                    },
                    trace: if include_trace {
                        Some(build_trace(&steps, initial_clauses))
                    } else {
                        None
                    },
                }
            }
        };
        
        // Convert to JS value
        serde_wasm_bindgen::to_value(&prover_result)
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

fn build_trace(steps: &[proofatlas::ProofStep], initial_count: usize) -> ProofTrace {
    let mut trace = ProofTrace {
        initial_clauses: Vec::new(),
        saturation_steps: Vec::new(),
    };
    
    let mut processed_count = 0;
    let mut unprocessed_count = initial_count;
    
    for step in steps.iter() {
        if step.inference.rule == proofatlas::InferenceRule::Input {
            // Initial clause
            trace.initial_clauses.push(ProofStep {
                id: step.clause_idx,
                clause: format_clause(&step.inference.conclusion),
                rule: "Input".to_string(),
                parents: vec![],
            });
        } else {
            // Saturation step
            let step_type = if step.inference.rule == proofatlas::InferenceRule::GivenClauseSelection {
                "given_selection"
            } else {
                "inference"
            };
            
            // Update counts based on step type
            if step.inference.rule == proofatlas::InferenceRule::GivenClauseSelection {
                processed_count += 1;
                unprocessed_count = unprocessed_count.saturating_sub(1);
            } else if step.inference.rule != proofatlas::InferenceRule::Input {
                unprocessed_count += 1;
            }
            
            trace.saturation_steps.push(SaturationStep {
                step_type: step_type.to_string(),
                clause_idx: step.clause_idx,
                clause: format_clause(&step.inference.conclusion),
                rule: format!("{:?}", step.inference.rule),
                premises: step.inference.premises.clone(),
                processed_count,
                unprocessed_count,
            });
        }
    }
    
    trace
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