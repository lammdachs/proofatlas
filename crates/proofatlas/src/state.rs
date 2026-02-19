//! Core state types for saturation-based theorem proving.
//!
//! This module consolidates all state-related types: clause sets, event log,
//! proof representation, inference traits, and derivation tracking.

use crate::logic::{Clause, Position};
use crate::logic::clause_manager::ClauseManager;
use crate::index::IndexRegistry;
use indexmap::IndexSet;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

// =============================================================================
// Verification
// =============================================================================

/// Errors that can occur during proof verification.
#[derive(Debug, Clone)]
pub enum VerificationError {
    /// A premise clause index is out of range
    InvalidPremise { step_idx: usize, premise_idx: usize },
    /// The conclusion clause doesn't follow from the premises for the stated rule
    InvalidConclusion { step_idx: usize, rule: String, reason: String },
    /// The stated rule name is not recognized
    UnknownRule { step_idx: usize, rule: String },
    /// An input clause was not found in the original problem
    InputNotFound { step_idx: usize, clause_idx: usize },
}

impl fmt::Display for VerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VerificationError::InvalidPremise { step_idx, premise_idx } => {
                write!(f, "Step {}: premise clause {} is out of range", step_idx, premise_idx)
            }
            VerificationError::InvalidConclusion { step_idx, rule, reason } => {
                write!(f, "Step {} ({}): {}", step_idx, rule, reason)
            }
            VerificationError::UnknownRule { step_idx, rule } => {
                write!(f, "Step {}: unknown rule '{}'", step_idx, rule)
            }
            VerificationError::InputNotFound { step_idx, clause_idx } => {
                write!(f, "Step {}: input clause {} not found in problem", step_idx, clause_idx)
            }
        }
    }
}

// =============================================================================
// Proof
// =============================================================================

/// A single step in a proof derivation. Every step produces a clause.
#[derive(Debug, Clone)]
pub struct ProofStep {
    pub clause_idx: usize,
    pub rule_name: String,
    pub premises: Vec<Position>,
    pub conclusion: Arc<Clause>,
}

// =============================================================================
// StateChange & EventLog
// =============================================================================

/// Atomic operations on the proof state.
///
/// These operations represent all possible modifications to the clause sets:
/// - N (new): Fresh clauses awaiting simplification
/// - U (unprocessed): Simplified clauses awaiting selection
/// - P (processed): Selected clauses used for inferences
#[derive(Debug, Clone, Serialize)]
pub enum StateChange {
    /// New clause added to N (from inference or input): (clause, rule_name, premises)
    Add(Arc<Clause>, String, Vec<Position>),
    /// Clause simplified: removed and optionally replaced.
    /// (clause_idx, replacement, rule_name, premises)
    Simplify(usize, Option<Arc<Clause>>, String, Vec<Position>),
    /// Clause transferred from N to U (survived forward simplification)
    Transfer(usize),
    /// Clause selected and transferred from U to P
    Activate(usize),
}

/// Extract clause indices from a list of positions (ignoring subterm paths).
pub fn clause_indices(premises: &[Position]) -> Vec<usize> {
    premises.iter().map(|p| p.clause).collect()
}

/// Type alias for the event log (replaces semantic SaturationTrace)
pub type EventLog = Vec<StateChange>;

// =============================================================================
// Wire types (shared by WASM and Python server)
// =============================================================================

/// A proof/clause step in the frontend wire format.
#[derive(Serialize, Clone)]
pub struct WireStep {
    pub id: usize,
    pub clause: String,
    pub rule: String,
    pub parents: Vec<usize>,
}

/// Statistics for a prove result.
#[derive(Serialize, Clone)]
pub struct ProveStatistics {
    pub initial_clauses: usize,
    pub generated_clauses: usize,
    pub final_clauses: usize,
    pub time_ms: u32,
}

/// Complete result of a prove call, ready for JSON serialization.
#[derive(Serialize, Clone)]
pub struct ProveResult {
    pub success: bool,
    pub status: String,
    pub message: String,
    pub proof: Option<Vec<WireStep>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub all_clauses: Option<Vec<WireStep>>,
    pub statistics: ProveStatistics,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace: Option<Trace>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile: Option<serde_json::Value>,
}

/// Convert internal ProofSteps to wire format.
pub fn steps_to_wire(steps: &[ProofStep], interner: &crate::logic::Interner) -> Vec<WireStep> {
    steps.iter().map(|step| WireStep {
        id: step.clause_idx,
        clause: step.conclusion.display(interner).to_string(),
        rule: step.rule_name.clone(),
        parents: clause_indices(&step.premises),
    }).collect()
}

/// Collect all clause steps (Add + Simplify replacements) from the event log as wire format.
pub fn all_steps_wire(events: &[StateChange], interner: &crate::logic::Interner) -> Vec<WireStep> {
    let mut steps = Vec::new();
    for event in events {
        match event {
            StateChange::Add(clause, rule_name, premises) => {
                if let Some(idx) = clause.id {
                    steps.push(WireStep {
                        id: idx,
                        clause: clause.display(interner).to_string(),
                        rule: rule_name.clone(),
                        parents: clause_indices(premises),
                    });
                }
            }
            StateChange::Simplify(_, Some(clause), rule_name, premises) => {
                if let Some(idx) = clause.id {
                    steps.push(WireStep {
                        id: idx,
                        clause: clause.display(interner).to_string(),
                        rule: rule_name.clone(),
                        parents: clause_indices(premises),
                    });
                }
            }
            _ => {}
        }
    }
    steps
}

/// Status message for a non-proof result.
pub fn status_message(result: &ProofResult) -> (&'static str, &'static str) {
    match result {
        ProofResult::Proof { .. } => ("proof_found", ""),
        ProofResult::Saturated => ("saturated", "Saturated without finding a proof - the formula may be satisfiable"),
        ProofResult::ResourceLimit => ("resource_limit", "Resource limit reached"),
    }
}

// =============================================================================
// Trace conversion (shared by WASM and Python server)
// =============================================================================

/// A single event in the frontend trace format.
#[derive(Serialize, Clone)]
pub struct TraceEvent {
    pub clause_idx: usize,
    pub clause: String,
    pub rule: String,
    pub premises: Vec<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replacement_idx: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replacement_clause: Option<String>,
}

/// One iteration of the saturation loop.
#[derive(Serialize, Clone)]
pub struct TraceIteration {
    pub simplification: Vec<TraceEvent>,
    pub selection: Option<TraceEvent>,
    pub generation: Vec<TraceEvent>,
}

/// Initial clause entry.
#[derive(Serialize, Clone)]
pub struct TraceInitialClause {
    pub id: usize,
    pub clause: String,
}

/// Complete trace for the frontend ProofInspector.
#[derive(Serialize, Clone)]
pub struct Trace {
    pub initial_clauses: Vec<TraceInitialClause>,
    pub iterations: Vec<TraceIteration>,
}

/// Convert a raw event log into the structured trace format for the frontend.
///
/// The `format_clause` callback formats a `Clause` into a display string
/// (typically via `clause.display(interner).to_string()`).
pub fn build_trace(events: &[StateChange], format_clause: impl Fn(&Clause) -> String) -> Trace {
    use std::collections::HashMap as StdHashMap;

    // First pass: collect all clause strings and count initial clauses
    let mut clauses: StdHashMap<usize, String> = StdHashMap::new();
    let mut initial_clause_count = 0;

    for event in events {
        match event {
            StateChange::Add(clause, rule_name, _) => {
                if let Some(idx) = clause.id {
                    clauses.insert(idx, format_clause(clause));
                    if rule_name == "Input" {
                        initial_clause_count = initial_clause_count.max(idx + 1);
                    }
                }
            }
            StateChange::Simplify(_, Some(clause), _, _) => {
                if let Some(idx) = clause.id {
                    clauses.insert(idx, format_clause(clause));
                }
            }
            _ => {}
        }
    }

    let initial_clauses: Vec<TraceInitialClause> = (0..initial_clause_count)
        .map(|i| TraceInitialClause {
            id: i,
            clause: clauses.get(&i).cloned().unwrap_or_default(),
        })
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
    let mut iterations: Vec<TraceIteration> = Vec::new();
    let mut cur_simp: Vec<TraceEvent> = Vec::new();
    let mut cur_gen: Vec<TraceEvent> = Vec::new();
    let mut cur_sel: Option<TraceEvent> = None;
    let mut in_generation = false;

    let flush = |simp: &mut Vec<TraceEvent>,
                     sel: &mut Option<TraceEvent>,
                     gen: &mut Vec<TraceEvent>,
                     iters: &mut Vec<TraceIteration>| {
        if sel.is_some() || !simp.is_empty() || !gen.is_empty() {
            iters.push(TraceIteration {
                simplification: std::mem::take(simp),
                selection: sel.take(),
                generation: std::mem::take(gen),
            });
        }
    };

    for event in events {
        match event {
            StateChange::Add(clause, rule_name, premises) => {
                if let Some(idx) = clause.id {
                    let clause_str = clauses.get(&idx).cloned().unwrap_or_default();
                    let premise_indices = clause_indices(premises);

                    if rule_name == "Input" {
                        cur_simp.push(TraceEvent {
                            clause_idx: idx, clause: clause_str,
                            rule: "Input".into(), premises: premise_indices,
                            replacement_idx: None, replacement_clause: None,
                        });
                    } else {
                        cur_gen.push(TraceEvent {
                            clause_idx: idx, clause: clause_str,
                            rule: rule_name.clone(), premises: premise_indices,
                            replacement_idx: None, replacement_clause: None,
                        });
                    }
                }
            }
            StateChange::Simplify(clause_idx, replacement, rule_name, premises) => {
                if in_generation {
                    flush(&mut cur_simp, &mut cur_sel, &mut cur_gen, &mut iterations);
                    in_generation = false;
                }
                let clause_str = clauses.get(clause_idx).cloned().unwrap_or_default();
                let premise_indices = clause_indices(premises);

                if let Some(repl) = replacement {
                    let repl_idx = repl.id.unwrap_or(0);
                    let repl_str = clauses.get(&repl_idx).cloned().unwrap_or_default();
                    cur_simp.push(TraceEvent {
                        clause_idx: *clause_idx, clause: clause_str,
                        rule: rule_name.clone(), premises: premise_indices,
                        replacement_idx: Some(repl_idx), replacement_clause: Some(repl_str),
                    });
                } else {
                    let rule = match rule_name.as_str() {
                        "Tautology" => "TautologyDeletion",
                        "Subsumption" => "SubsumptionDeletion",
                        _ => "SubsumptionDeletion",
                    };
                    cur_simp.push(TraceEvent {
                        clause_idx: *clause_idx, clause: clause_str,
                        rule: rule.into(), premises: premise_indices,
                        replacement_idx: None, replacement_clause: None,
                    });
                }
            }
            StateChange::Transfer(clause_idx) => {
                if in_generation {
                    flush(&mut cur_simp, &mut cur_sel, &mut cur_gen, &mut iterations);
                    in_generation = false;
                }
                let clause_str = clauses.get(clause_idx).cloned().unwrap_or_default();
                cur_simp.push(TraceEvent {
                    clause_idx: *clause_idx, clause: clause_str,
                    rule: "Transfer".into(), premises: vec![],
                    replacement_idx: None, replacement_clause: None,
                });
            }
            StateChange::Activate(clause_idx) => {
                if in_generation {
                    flush(&mut cur_simp, &mut cur_sel, &mut cur_gen, &mut iterations);
                }
                let clause_str = clauses.get(clause_idx).cloned().unwrap_or_default();
                cur_sel = Some(TraceEvent {
                    clause_idx: *clause_idx, clause: clause_str,
                    rule: "GivenClauseSelection".into(), premises: vec![],
                    replacement_idx: None, replacement_clause: None,
                });
                in_generation = true;
            }
        }
    }

    // Flush remaining
    flush(&mut cur_simp, &mut cur_sel, &mut cur_gen, &mut iterations);

    Trace { initial_clauses, iterations }
}

// =============================================================================
// Inference traits
// =============================================================================

/// Trait for simplification rules (tautology, subsumption, demodulation).
///
/// Rules receive the full saturation state, clause manager, and index registry
/// at call time. They do not maintain internal state tracking clause lifecycle.
pub trait SimplifyingInference: Send + Sync {
    /// Get the name of this rule
    fn name(&self) -> &str;

    /// Forward simplification: try to simplify/delete a clause in N using U∪P.
    fn simplify_forward(
        &self,
        clause_idx: usize,
        state: &SaturationState,
        cm: &ClauseManager,
        indices: &IndexRegistry,
    ) -> Option<StateChange>;

    /// Backward simplification: simplify clauses in U∪P using this clause.
    fn simplify_backward(
        &self,
        _clause_idx: usize,
        _state: &SaturationState,
        _cm: &ClauseManager,
        _indices: &IndexRegistry,
    ) -> Vec<StateChange> {
        vec![]
    }

    /// Verify that a simplification step is correct.
    ///
    /// Given the simplified clause index, its optional replacement, and the premises,
    /// check that the conclusion follows from the premises according to this rule.
    fn verify(
        &self,
        _clause_idx: usize,
        _replacement: Option<&Clause>,
        _premises: &[Position],
        _state: &SaturationState,
        _cm: &ClauseManager,
    ) -> Result<(), VerificationError> {
        // Default: skip verification (for rules where verification is not yet implemented)
        Ok(())
    }
}

/// Trait for generating inference rules (resolution, superposition, factoring, etc.).
///
/// Rules receive the full saturation state, clause manager, and index registry
/// at call time. They do not maintain internal state tracking clause lifecycle.
pub trait GeneratingInference: Send + Sync {
    /// Get the name of this rule
    fn name(&self) -> &str;

    /// Generate inferences with the given clause.
    fn generate(
        &self,
        given_idx: usize,
        state: &SaturationState,
        cm: &mut ClauseManager,
        indices: &IndexRegistry,
    ) -> Vec<StateChange>;

    /// Verify that a generating inference step is correct.
    ///
    /// Given the conclusion clause and its premises, check that the conclusion
    /// follows from the premises according to this rule.
    fn verify(
        &self,
        _conclusion: &Clause,
        _premises: &[Position],
        _state: &SaturationState,
        _cm: &ClauseManager,
    ) -> Result<(), VerificationError> {
        // Default: skip verification (for rules where verification is not yet implemented)
        Ok(())
    }
}

// =============================================================================
// ProofResult
// =============================================================================

/// Result of saturation.
///
/// After `prove()`, all data (clauses, event log, proof steps) remains accessible
/// on the `ProofAtlas` instance. This enum only carries the outcome status.
#[derive(Debug, Clone)]
pub enum ProofResult {
    /// Empty clause derived - proof found
    Proof { empty_clause_idx: usize },
    /// Saturated without finding empty clause
    Saturated,
    /// Resource limit reached
    ResourceLimit,
}

// =============================================================================
// SaturationState
// =============================================================================

/// Lean data container for the three-set given-clause algorithm.
///
/// Holds clause storage, the N/U/P sets, and the event log.
/// The saturation algorithm lives in `ProofAtlas`.
pub struct SaturationState {
    /// Storage for all clauses, indexed by clause ID
    pub clauses: Vec<Arc<Clause>>,
    /// Set P: Processed/active clauses (used for generating inferences)
    pub processed: IndexSet<usize>,
    /// Set U: Unprocessed/passive clauses (awaiting selection)
    pub unprocessed: IndexSet<usize>,
    /// Set N: New clauses (awaiting forward simplification)
    pub new: Vec<usize>,
    /// Raw event log capturing all state changes
    pub event_log: Vec<StateChange>,
    /// Current iteration (used for clause age)
    pub current_iteration: usize,
    /// Number of initial input clauses
    pub initial_clause_count: usize,
}

impl SaturationState {
    /// Extract a proof by backward traversal from the given clause index.
    pub fn extract_proof(&self, clause_idx: usize) -> Vec<ProofStep> {
        let mut derivation_map: HashMap<usize, (String, Vec<Position>)> = HashMap::new();
        for event in &self.event_log {
            match event {
                StateChange::Add(clause, rule_name, premises) => {
                    if let Some(idx) = clause.id {
                        derivation_map.insert(idx, (rule_name.clone(), premises.clone()));
                    }
                }
                StateChange::Simplify(_, Some(clause), rule_name, premises) => {
                    if let Some(idx) = clause.id {
                        derivation_map.insert(idx, (rule_name.clone(), premises.clone()));
                    }
                }
                _ => {}
            }
        }

        let mut proof_clause_indices = Vec::new();
        let mut visited = HashSet::new();
        let mut to_visit = vec![clause_idx];

        while let Some(idx) = to_visit.pop() {
            if !visited.insert(idx) {
                continue;
            }
            proof_clause_indices.push(idx);
            if let Some((_, premises)) = derivation_map.get(&idx) {
                to_visit.extend(clause_indices(premises));
            }
        }

        proof_clause_indices.sort();

        proof_clause_indices
            .iter()
            .map(|&idx| {
                let (rule_name, premises) = derivation_map
                    .get(&idx)
                    .cloned()
                    .unwrap_or_else(|| ("Input".into(), vec![]));
                ProofStep {
                    clause_idx: idx,
                    rule_name,
                    premises,
                    conclusion: Arc::clone(&self.clauses[idx]),
                }
            })
            .collect()
    }

}
