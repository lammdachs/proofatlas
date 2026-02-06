//! Core state types for saturation-based theorem proving.
//!
//! This module consolidates all state-related types: clause sets, event log,
//! proof representation, inference traits, and derivation tracking.

use crate::logic::{Clause, Interner, Position};
use crate::json::{ProofJson, ProofResultJson, ClauseJson};
use crate::logic::literal_selection::LiteralSelector;
use indexmap::IndexSet;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// =============================================================================
// Derivation
// =============================================================================

/// How a clause was derived (for proofs and internal derivation tracking).
///
/// This is a dynamic struct that stores the rule name and premise positions,
/// allowing new rules to be added without modifying this type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Derivation {
    /// Name of the inference rule that produced this clause
    pub rule_name: String,
    /// Positions of the premise clauses used in the inference
    pub premises: Vec<Position>,
}

impl Derivation {
    /// Create an Input derivation (no premises)
    pub fn input() -> Self {
        Derivation {
            rule_name: "Input".into(),
            premises: vec![],
        }
    }

    /// Get the clause indices of all premises (ignoring subterm paths).
    pub fn clause_indices(&self) -> Vec<usize> {
        self.premises.iter().map(|p| p.clause).collect()
    }
}

// =============================================================================
// InferenceResult
// =============================================================================

/// Result of an inference rule application
#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub derivation: Derivation,
    pub conclusion: Clause,
}

// =============================================================================
// Proof
// =============================================================================

/// A single step in a proof derivation. Every step produces a clause.
#[derive(Debug, Clone)]
pub struct ProofStep {
    pub clause_idx: usize,
    pub derivation: Derivation,
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
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum StateChange {
    /// New clause added to N (from inference or input)
    Add { clause: Clause, derivation: Derivation },
    /// Clause deleted from its current set (simplification)
    Delete { clause_idx: usize, rule_name: String },
    /// Clause transferred from N to U (survived forward simplification)
    Transfer { clause_idx: usize },
    /// Clause selected and transferred from U to P
    Activate { clause_idx: usize },
}

/// Type alias for the event log (replaces semantic SaturationTrace)
pub type EventLog = Vec<StateChange>;

// =============================================================================
// ClauseSet & ClauseNotification
// =============================================================================

/// Which clause set a notification refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClauseSet {
    /// Unprocessed set (U)
    Unprocessed,
    /// Processed set (P)
    Processed,
}

/// Notifications sent to rules when clauses are added/removed from U or P.
///
/// Rules can maintain internal indices by listening to these notifications.
#[derive(Debug, Clone)]
pub enum ClauseNotification<'a> {
    /// Clause was added to a set
    Added { clause_idx: usize, clause: &'a Clause },
    /// Clause was removed from a set
    Removed { clause_idx: usize, clause: &'a Clause },
}

// =============================================================================
// Inference traits
// =============================================================================

/// Trait for simplification rules (tautology, subsumption, demodulation).
pub trait SimplifyingInference: Send + Sync {
    /// Get the name of this rule
    fn name(&self) -> &str;

    /// Initialize the rule with input clauses.
    fn initialize(&mut self, _clauses: &[Clause]) {}

    /// Called when a clause is added to N (pending, not yet active).
    fn on_clause_pending(&mut self, _clause_idx: usize, _clause: &Clause) {}

    /// Called when a clause is activated (transferred from N to U).
    fn on_clause_activated(&mut self, _clause_idx: usize, _clause: &Clause) {}

    /// Notified when a clause is added to or removed from U or P.
    fn notify(&mut self, _set: ClauseSet, _notif: ClauseNotification) {}

    /// Forward simplification: try to simplify/delete a clause in N using U∪P.
    fn simplify_forward(
        &self,
        clause_idx: usize,
        clause: &Clause,
        clauses: &[Clause],
        interner: &Interner,
    ) -> Vec<StateChange>;

    /// Backward simplification: simplify clauses in U∪P using this clause.
    fn simplify_backward(
        &self,
        _clause_idx: usize,
        _clause: &Clause,
        _clauses: &[Clause],
        _unprocessed: &IndexSet<usize>,
        _processed: &IndexSet<usize>,
        _interner: &Interner,
    ) -> Vec<StateChange> {
        vec![]
    }
}

/// Trait for generating inference rules (resolution, superposition, factoring, etc.).
pub trait GeneratingInference: Send + Sync {
    /// Get the name of this rule
    fn name(&self) -> &str;

    /// Notified when a clause is added to or removed from P (processed)
    fn notify(&mut self, _notif: ClauseNotification) {}

    /// Generate inferences with the given clause.
    fn generate(
        &self,
        given_idx: usize,
        given: &Clause,
        clauses: &[Clause],
        processed: &IndexSet<usize>,
        selector: &dyn LiteralSelector,
        interner: &mut Interner,
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
    /// Timeout reached
    Timeout(Vec<ProofStep>, Vec<Clause>),
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
                    reason: "Clause or iteration limit exceeded".to_string(),
                    final_clauses: clauses.iter().map(|c| ClauseJson::from_clause(c, interner)).collect(),
                    proof_steps: steps.iter().map(|s| s.into()).collect(),
                    time_seconds,
                }
            }
            ProofResult::Timeout(steps, clauses) => ProofResultJson::Timeout {
                final_clauses: clauses.iter().map(|c| ClauseJson::from_clause(c, interner)).collect(),
                proof_steps: steps.iter().map(|s| s.into()).collect(),
                time_seconds,
            },
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
    pub new: VecDeque<usize>,
    /// Raw event log capturing all state changes
    pub event_log: Vec<StateChange>,
    /// Tracked clause memory usage in bytes
    pub clause_memory_bytes: usize,
    /// Current iteration (used for clause age)
    pub current_iteration: usize,
    /// Number of initial input clauses
    pub initial_clause_count: usize,
}
