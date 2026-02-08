//! Core state types for saturation-based theorem proving.
//!
//! This module consolidates all state-related types: clause sets, event log,
//! proof representation, inference traits, and derivation tracking.

use crate::logic::{Clause, Interner, Position};
use crate::logic::clause_manager::ClauseManager;
use crate::index::IndexRegistry;
use crate::json::{ProofJson, ProofResultJson, ClauseJson};
use indexmap::IndexSet;
use serde::Serialize;
use std::collections::{HashMap, HashSet};

// =============================================================================
// Proof
// =============================================================================

/// A single step in a proof derivation. Every step produces a clause.
#[derive(Debug, Clone)]
pub struct ProofStep {
    pub clause_idx: usize,
    pub rule_name: String,
    pub premises: Vec<Position>,
    pub conclusion: Clause,
}

/// A proof is a sequence of inference steps
#[derive(Debug, Clone)]
pub struct Proof {
    pub steps: Vec<ProofStep>,
    pub empty_clause_idx: usize,
    /// All clauses generated during saturation (for ML training data extraction)
    pub all_clauses: Vec<Clause>,
}

impl Proof {
    /// Convert to JSON representation
    pub fn to_json(&self, _interner: &Interner) -> ProofJson {
        self.into()
    }
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
    Add(Clause, String, Vec<Position>),
    /// Clause deleted from its current set: (clause_idx, rule_name, justification)
    Delete(usize, String, Vec<Position>),
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
    ) -> Vec<StateChange>;

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
}

// =============================================================================
// ProofResult
// =============================================================================

/// Result of saturation
#[derive(Debug, Clone)]
pub enum ProofResult {
    /// Empty clause derived - proof found
    Proof(Proof),
    /// Saturated without finding empty clause
    Saturated(Vec<ProofStep>, Vec<Clause>),
    /// Resource limit reached
    ResourceLimit(Vec<ProofStep>, Vec<Clause>),
}

impl ProofResult {
    /// Convert to JSON representation
    pub fn to_json(&self, time_seconds: f64, interner: &Interner) -> ProofResultJson {
        match self {
            ProofResult::Proof(proof) => ProofResultJson::Proof {
                proof: proof.to_json(interner),
                time_seconds,
            },
            ProofResult::Saturated(steps, clauses) => ProofResultJson::Saturated {
                final_clauses: clauses.iter().map(|c| ClauseJson::from_clause(c, interner)).collect(),
                proof_steps: steps.iter().map(|s| s.into()).collect(),
                time_seconds,
            },
            ProofResult::ResourceLimit(steps, clauses) => {
                ProofResultJson::ResourceLimit {
                    reason: "Resource limit exceeded".to_string(),
                    final_clauses: clauses.iter().map(|c| ClauseJson::from_clause(c, interner)).collect(),
                    proof_steps: steps.iter().map(|s| s.into()).collect(),
                    time_seconds,
                }
            }
        }
    }
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
    pub clauses: Vec<Clause>,
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
    /// Extract a proof by backward traversal from the empty clause.
    pub fn extract_proof(&self, empty_clause_idx: usize) -> Proof {
        let mut derivation_map: HashMap<usize, (String, Vec<Position>)> = HashMap::new();
        for event in &self.event_log {
            if let StateChange::Add(clause, rule_name, premises) = event {
                if let Some(idx) = clause.id {
                    derivation_map.insert(idx, (rule_name.clone(), premises.clone()));
                }
            }
        }

        let mut proof_clause_indices = Vec::new();
        let mut visited = HashSet::new();
        let mut to_visit = vec![empty_clause_idx];

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

        let steps = proof_clause_indices
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
                    conclusion: self.clauses[idx].clone(),
                }
            })
            .collect();

        Proof {
            steps,
            empty_clause_idx,
            all_clauses: self.clauses.clone(),
        }
    }

    /// Build proof steps from event log.
    pub fn build_proof_steps(&self) -> Vec<ProofStep> {
        self.event_log
            .iter()
            .filter_map(|event| {
                if let StateChange::Add(clause, rule_name, premises) = event {
                    clause.id.map(|idx| ProofStep {
                        clause_idx: idx,
                        rule_name: rule_name.clone(),
                        premises: premises.clone(),
                        conclusion: clause.clone(),
                    })
                } else {
                    None
                }
            })
            .collect()
    }
}
