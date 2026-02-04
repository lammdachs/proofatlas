//! Saturation trace types for recording proof search iterations.

use crate::inference::Derivation;
use serde::{Deserialize, Serialize};

/// One iteration of the given-clause algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaturationStep {
    pub simplifications: Vec<ClauseSimplification>,
    /// Index of the clause selected from P. `None` means saturated (P was empty).
    pub given_clause: Option<usize>,
    pub generating_inferences: Vec<GeneratingInference>,
}

/// What happened when a clause was popped from N for simplification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClauseSimplification {
    pub clause_idx: usize,
    /// `None` means the empty clause was found (proof terminates).
    /// `Some(Forward(..))` means the clause was deleted.
    /// `Some(Backward(..))` means the clause survived and was transferred to P.
    pub outcome: Option<SimplificationOutcome>,
}

/// Result of forward simplification on a clause from N.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum SimplificationOutcome {
    /// Clause eliminated by forward simplification (deleted, not transferred).
    Forward(ForwardSimplification),
    /// Clause survived forward simplification; backward effects applied, then transferred to P.
    Backward { effects: Vec<BackwardSimplification> },
}

/// A forward simplification that deletes a clause.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ForwardSimplification {
    Tautology,
    Demodulation { demodulator: usize, result: usize },
    Subsumption { subsumer: usize },
}

/// A backward simplification effect on existing clauses in P or A.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum BackwardSimplification {
    Subsumption { deleted_clause: usize },
    Demodulation { old_clause: usize, result: usize },
}

/// A generating inference that produced a new clause.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum GeneratingInference {
    Resolution { clause_idx: usize, parents: (usize, usize) },
    Factoring { clause_idx: usize, parent: usize },
    Superposition { clause_idx: usize, parents: (usize, usize) },
    EqualityResolution { clause_idx: usize, parent: usize },
    EqualityFactoring { clause_idx: usize, parent: usize },
}

impl GeneratingInference {
    /// Convert a `Derivation` (from an inference result) into a `GeneratingInference`.
    ///
    /// Panics if the derivation is `Input` or `Demodulation` (not generating inferences).
    pub fn from_derivation(clause_idx: usize, d: &Derivation) -> Self {
        match d.rule_name.as_str() {
            "Resolution" => {
                Self::Resolution { clause_idx, parents: (d.premises[0], d.premises[1]) }
            }
            "Factoring" => {
                Self::Factoring { clause_idx, parent: d.premises[0] }
            }
            "Superposition" => {
                Self::Superposition { clause_idx, parents: (d.premises[0], d.premises[1]) }
            }
            "EqualityResolution" => {
                Self::EqualityResolution { clause_idx, parent: d.premises[0] }
            }
            "EqualityFactoring" => {
                Self::EqualityFactoring { clause_idx, parent: d.premises[0] }
            }
            _ => unreachable!("Input/Demodulation are not generating inferences: {}", d.rule_name),
        }
    }
}

/// Complete trace of a saturation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaturationTrace {
    /// All clauses generated during saturation (indexed by clause_idx).
    pub clauses: Vec<String>,
    /// Number of clauses from the initial input.
    pub initial_clause_count: usize,
    /// One entry per iteration of the given-clause loop.
    pub iterations: Vec<SaturationStep>,
}

// =============================================================================
// Event Log Replay Utilities
// =============================================================================

use super::rule::{ProofStateChange, SaturationEventLog};
use crate::fol::Clause;
use std::collections::HashSet;

/// Replay event log to extract proof (backward traversal from empty clause)
pub fn extract_proof_from_events(events: &SaturationEventLog) -> Option<Vec<usize>> {
    // Find the empty clause
    let empty_clause_idx = events.iter().find_map(|e| {
        if let ProofStateChange::AddN { clause, derivation: _ } = e {
            if clause.is_empty() {
                clause.id
            } else {
                None
            }
        } else {
            None
        }
    })?;

    // Build derivation map: clause_idx -> (rule_name, premises)
    let mut derivation_map: std::collections::HashMap<usize, (String, Vec<usize>)> =
        std::collections::HashMap::new();
    for event in events {
        if let ProofStateChange::AddN { clause, derivation } = event {
            if let Some(idx) = clause.id {
                derivation_map.insert(idx, (derivation.rule_name.clone(), derivation.premises.clone()));
            }
        }
    }

    // Backward traversal from empty clause
    let mut proof_clauses = Vec::new();
    let mut visited = HashSet::new();
    let mut to_visit = vec![empty_clause_idx];

    while let Some(idx) = to_visit.pop() {
        if !visited.insert(idx) {
            continue;
        }
        proof_clauses.push(idx);
        if let Some((_, premises)) = derivation_map.get(&idx) {
            to_visit.extend(premises);
        }
    }

    // Sort topologically (by index, since parents always have lower indices)
    proof_clauses.sort();
    Some(proof_clauses)
}

/// Replay event log to reconstruct clause sets at any point
pub struct EventLogReplayer<'a> {
    events: &'a SaturationEventLog,
    position: usize,
    n: HashSet<usize>,
    u: HashSet<usize>,
    p: HashSet<usize>,
    clauses: Vec<Clause>,
}

impl<'a> EventLogReplayer<'a> {
    pub fn new(events: &'a SaturationEventLog) -> Self {
        EventLogReplayer {
            events,
            position: 0,
            n: HashSet::new(),
            u: HashSet::new(),
            p: HashSet::new(),
            clauses: Vec::new(),
        }
    }

    /// Replay all events
    pub fn replay_all(&mut self) {
        while self.position < self.events.len() {
            self.step();
        }
    }

    /// Step through one event
    pub fn step(&mut self) -> Option<&ProofStateChange> {
        if self.position >= self.events.len() {
            return None;
        }

        let event = &self.events[self.position];
        self.position += 1;

        match event {
            ProofStateChange::AddN { clause, derivation: _ } => {
                let idx = clause.id.unwrap_or(self.clauses.len());
                if idx >= self.clauses.len() {
                    self.clauses.resize(idx + 1, Clause::new(vec![]));
                }
                self.clauses[idx] = clause.clone();
                self.n.insert(idx);
            }
            ProofStateChange::RemoveN { clause_idx, rule_name: _ } => {
                self.n.remove(clause_idx);
            }
            ProofStateChange::AddU { clause_idx } => {
                self.n.remove(clause_idx);
                self.u.insert(*clause_idx);
            }
            ProofStateChange::RemoveU { clause_idx, rule_name: _ } => {
                self.u.remove(clause_idx);
            }
            ProofStateChange::AddP { clause_idx } => {
                self.u.remove(clause_idx);
                self.p.insert(*clause_idx);
            }
            ProofStateChange::RemoveP { clause_idx, rule_name: _ } => {
                self.p.remove(clause_idx);
            }
        }

        Some(event)
    }

    /// Get current N set
    pub fn n_set(&self) -> &HashSet<usize> {
        &self.n
    }

    /// Get current U set
    pub fn u_set(&self) -> &HashSet<usize> {
        &self.u
    }

    /// Get current P set
    pub fn p_set(&self) -> &HashSet<usize> {
        &self.p
    }

    /// Get all clauses
    pub fn clauses(&self) -> &[Clause] {
        &self.clauses
    }
}
