use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use proofatlas::{parse_tptp, saturate, SaturationConfig, SaturationResult, LiteralSelectionStrategy, Clause, Literal};
use std::time::Duration;

#[wasm_bindgen]
pub struct ProofAtlasWasm;

#[derive(Serialize, Deserialize)]
pub struct ProverOptions {
    pub timeout_ms: u32,
    pub max_clauses: usize,
    pub use_superposition: bool,
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
    pub fn prove(&self, tptp_input: &str, options_js: JsValue) -> Result<JsValue, JsError> {
        // Parse options
        let options: ProverOptions = serde_wasm_bindgen::from_value(options_js)
            .map_err(|e| JsError::new(&format!("Invalid options: {}", e)))?;
        
        web_sys::console::log_1(&format!("Options parsed: timeout_ms={}, max_clauses={}, use_superposition={}", 
            options.timeout_ms, options.max_clauses, options.use_superposition).into());
        
        // Parse TPTP input
        let cnf = parse_tptp(tptp_input)
            .map_err(|e| JsError::new(&format!("Parse error: {}", e)))?;
        
        web_sys::console::log_1(&format!("Parsed {} clauses", cnf.clauses.len()).into());
        
        let initial_clauses = cnf.clauses.len();
        let start_time = web_sys::window()
            .unwrap()
            .performance()
            .unwrap()
            .now();
        
        // Create saturation config
        let config = SaturationConfig {
            max_clauses: options.max_clauses,
            max_iterations: 10000,
            max_clause_size: 100,
            timeout: Duration::from_millis(options.timeout_ms as u64),
            use_superposition: options.use_superposition,
            literal_selection: LiteralSelectionStrategy::SelectMaxWeight,
            step_limit: None,
        };
        
        web_sys::console::log_1(&"Config created, calling saturate...".into());
        
        // Run saturation
        let result = saturate(cnf, config);
        
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
                
                // Extract the proof path - collect all ancestors
                let mut proof_indices = std::collections::HashSet::new();
                let mut to_visit = vec![proof.empty_clause_idx];
                
                // Collect all clauses that contribute to the proof
                while let Some(current) = to_visit.pop() {
                    if current < all_steps.len() && proof_indices.insert(current) {
                        // Add all parents to visit
                        for &parent in &all_steps[current].parents {
                            if parent < all_steps.len() && !proof_indices.contains(&parent) {
                                to_visit.push(parent);
                            }
                        }
                    }
                }
                
                // Sort proof steps by index to show them in order
                let mut proof_path: Vec<ProofStep> = proof_indices.into_iter()
                    .map(|idx| all_steps[idx].clone())
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
                }
            }
            SaturationResult::Saturated(steps) => {
                // Saturated without proof
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
                        final_clauses: steps.len(),
                        time_ms,
                    },
                }
            }
            SaturationResult::Timeout(steps) => {
                // Timeout
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
                        final_clauses: steps.len(),
                        time_ms,
                    },
                }
            }
            SaturationResult::ResourceLimit(steps) => {
                // Resource limit
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
                        final_clauses: steps.len(),
                        time_ms,
                    },
                }
            }
        };
        
        // Convert to JS value
        serde_wasm_bindgen::to_value(&prover_result)
            .map_err(|e| JsError::new(&format!("Serialization error: {}", e)))
    }
    
    #[wasm_bindgen]
    pub fn parse_tptp(&self, input: &str) -> Result<String, JsError> {
        // Just parse and return success/error for validation
        match parse_tptp(input) {
            Ok(cnf) => Ok(format!("Valid TPTP input with {} clauses", cnf.clauses.len())),
            Err(e) => Err(JsError::new(&format!("Parse error: {}", e))),
        }
    }
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