//! Event log replay utilities for proof extraction and state reconstruction.

use super::rule::{ProofStateChange, SaturationEventLog};
use crate::fol::Clause;
use std::collections::HashSet;

/// Replay event log to extract proof (backward traversal from empty clause)
pub fn extract_proof_from_events(events: &SaturationEventLog) -> Option<Vec<usize>> {
    // Find the empty clause
    let empty_clause_idx = events.iter().find_map(|e| {
        if let ProofStateChange::New { clause, derivation: _ } = e {
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
        if let ProofStateChange::New { clause, derivation } = event {
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
            ProofStateChange::New { clause, derivation: _ } => {
                let idx = clause.id.unwrap_or(self.clauses.len());
                if idx >= self.clauses.len() {
                    self.clauses.resize(idx + 1, Clause::new(vec![]));
                }
                self.clauses[idx] = clause.clone();
                self.n.insert(idx);
            }
            ProofStateChange::DeleteN { clause_idx, rule_name: _ } => {
                self.n.remove(clause_idx);
            }
            ProofStateChange::Transfer { clause_idx } => {
                // Implicit: removed from N, added to U
                self.n.remove(clause_idx);
                self.u.insert(*clause_idx);
            }
            ProofStateChange::DeleteU { clause_idx, rule_name: _ } => {
                self.u.remove(clause_idx);
            }
            ProofStateChange::Select { clause_idx } => {
                // Implicit: removed from U, added to P
                self.u.remove(clause_idx);
                self.p.insert(*clause_idx);
            }
            ProofStateChange::DeleteP { clause_idx, rule_name: _ } => {
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
