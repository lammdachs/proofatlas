//! Subsumption checker index for forward/backward subsumption filtering.
//!
//! This wraps the feature vector index, clause keys, and unit clause tracking
//! needed for efficient subsumption checking into a single `Index` implementation.

use crate::index::feature_vector::FeatureIndex;
use crate::index::{Index, IndexKind};
use crate::logic::{Clause, ClauseKey};
use crate::simplifying::subsumption::{
    are_variants, compatible_structure, subsumes, subsumes_greedy, subsumes_unit,
};
use std::any::Any;
use std::collections::{HashMap, HashSet};

/// Subsumption checker implementing a balanced redundancy elimination strategy.
///
/// Maintains internal indices (feature vectors, clause keys, unit clauses) and
/// tracks the clause lifecycle via the `Index` trait.
pub struct SubsumptionChecker {
    /// Map from clause key to clause index (for active clauses, used by find_subsumer)
    clause_key_to_idx: HashMap<ClauseKey, usize>,

    /// Unit clauses for unit subsumption
    units: Vec<(Clause, usize)>,

    /// All clauses for subsumption checking
    clauses: Vec<Clause>,

    /// Indices of active clauses (in U ∪ P)
    active: HashSet<usize>,

    /// Feature vector index for efficient subsumption filtering
    feature_index: FeatureIndex,
}

impl SubsumptionChecker {
    pub fn new() -> Self {
        SubsumptionChecker {
            clause_key_to_idx: HashMap::new(),
            units: Vec::new(),
            clauses: Vec::new(),
            active: HashSet::new(),
            feature_index: FeatureIndex::new(),
        }
    }

    /// Find the active clause that subsumes the given clause, returning its index.
    /// Returns None if no active clause subsumes it.
    pub fn find_subsumer(&self, clause: &Clause) -> Option<usize> {
        // 1. Check for exact duplicates (very fast with structural hashing)
        let clause_key = ClauseKey::from_clause(clause);
        if let Some(&idx) = self.clause_key_to_idx.get(&clause_key) {
            return Some(idx);
        }

        // 2. Check for variants (duplicates up to variable renaming)
        if let Some(idx) = self.find_variant(clause) {
            return Some(idx);
        }

        // 3. Unit subsumption (fast and complete)
        for (unit, idx) in &self.units {
            if subsumes_unit(unit, clause) {
                return Some(*idx);
            }
        }

        // 4. Use feature index to get candidate subsumers
        let candidates = self.feature_index.find_potential_subsumers(clause);

        // 5. Check candidates with full subsumption
        for idx in candidates {
            let existing = &self.clauses[idx];

            // Skip if same size (handled by variant check) or larger
            if existing.literals.len() >= clause.literals.len() {
                continue;
            }

            // Skip unit clauses (already handled above)
            if existing.literals.len() == 1 {
                continue;
            }

            // For small clauses (2-3 literals), do complete subsumption
            if existing.literals.len() <= 3 {
                if subsumes(existing, clause) {
                    return Some(idx);
                }
            } else {
                // For larger clauses, use greedy heuristic
                if compatible_structure(existing, clause) && subsumes_greedy(existing, clause) {
                    return Some(idx);
                }
            }
        }

        None
    }

    /// Check if a clause is subsumed by any processed clause (excluding itself)
    pub fn is_subsumed_by_processed(&self, exclude_idx: usize, clause: &Clause) -> bool {
        // Unit subsumption (excluding self)
        for (unit, idx) in &self.units {
            if *idx == exclude_idx {
                continue;
            }
            if subsumes_unit(unit, clause) {
                return true;
            }
        }

        // Use feature index to get potential subsumers
        let candidates = self.feature_index.find_potential_subsumers(clause);

        for idx in candidates {
            if idx == exclude_idx {
                continue;
            }

            let existing = &self.clauses[idx];

            // Check for variants (same size)
            if existing.literals.len() == clause.literals.len() {
                if are_variants(existing, clause) {
                    return true;
                }
                continue;
            }

            // Skip if existing is larger (can't subsume)
            if existing.literals.len() > clause.literals.len() {
                continue;
            }

            // Skip unit clauses (already handled above)
            if existing.literals.len() == 1 {
                continue;
            }

            // Full subsumption for small clauses
            if existing.literals.len() <= 3 {
                if subsumes(existing, clause) {
                    return true;
                }
            } else {
                if compatible_structure(existing, clause) && subsumes_greedy(existing, clause) {
                    return true;
                }
            }
        }

        false
    }

    /// Find which clauses from the given indices are subsumed by the subsumer clause.
    /// Returns the indices of subsumed clauses.
    pub fn find_subsumed_by(&self, subsumer_idx: usize, candidate_indices: &[usize]) -> Vec<usize> {
        let subsumer = &self.clauses[subsumer_idx];
        let mut subsumed = Vec::new();

        // Use feature index to filter candidates
        let feature_candidates: HashSet<usize> = self
            .feature_index
            .find_potentially_subsumed(subsumer_idx)
            .into_iter()
            .collect();

        for &idx in candidate_indices {
            if idx == subsumer_idx {
                continue;
            }

            // Skip if not a feature-compatible candidate
            if !feature_candidates.contains(&idx) {
                continue;
            }

            let candidate = &self.clauses[idx];

            // Quick check: subsumer can't be larger
            if subsumer.literals.len() > candidate.literals.len() {
                continue;
            }

            // Check if subsumer subsumes candidate
            if subsumer.literals.len() == 1 {
                if subsumes_unit(subsumer, candidate) {
                    subsumed.push(idx);
                }
            } else if subsumer.literals.len() <= 3 {
                if subsumes(subsumer, candidate) {
                    subsumed.push(idx);
                }
            } else {
                if subsumes_greedy(subsumer, candidate) {
                    subsumed.push(idx);
                }
            }
        }

        subsumed
    }

    /// Find a variant of this clause among active clauses, returning its index
    fn find_variant(&self, clause: &Clause) -> Option<usize> {
        let shape = get_clause_shape(clause);

        let candidates = self.feature_index.find_potential_subsumers(clause);

        for idx in candidates {
            let existing = &self.clauses[idx];

            if existing.literals.len() != clause.literals.len() {
                continue;
            }

            if get_clause_shape(existing) != shape {
                continue;
            }

            if are_variants(existing, clause) {
                return Some(idx);
            }
        }

        None
    }
}

impl Default for SubsumptionChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl Index for SubsumptionChecker {
    fn kind(&self) -> IndexKind {
        IndexKind::Subsumption
    }

    fn initialize(&mut self, clauses: &[Clause]) {
        self.feature_index.initialize_symbols(clauses);
    }

    fn on_add(&mut self, idx: usize, clause: &Clause) {
        // Add clause to internal store + feature index (not transferred yet)
        let feature_idx = self.feature_index.add_clause(clause);
        debug_assert_eq!(idx, feature_idx, "SubsumptionChecker index mismatch: expected {}, got {}", idx, feature_idx);
        // Ensure clauses vec is large enough
        if idx >= self.clauses.len() {
            self.clauses.resize(idx + 1, Clause::new(vec![]));
        }
        self.clauses[idx] = clause.clone();
    }

    fn on_transfer(&mut self, idx: usize, _clause: &Clause) {
        if self.active.contains(&idx) {
            return;
        }
        self.active.insert(idx);
        self.feature_index.activate(idx);

        // Use the stored (oriented) clause for keys and units
        let clause = &self.clauses[idx];

        // Add to key index
        let clause_key = ClauseKey::from_clause(clause);
        self.clause_key_to_idx.insert(clause_key, idx);

        // Add to unit index if applicable
        if clause.literals.len() == 1 {
            self.units.push((clause.clone(), idx));
        }
    }

    fn on_delete(&mut self, idx: usize, _clause: &Clause) {
        self.active.remove(&idx);
        self.feature_index.deactivate(idx);

        // Remove from clause_key_to_idx
        let clause = &self.clauses[idx];
        let clause_key = ClauseKey::from_clause(clause);
        if self.clause_key_to_idx.get(&clause_key) == Some(&idx) {
            self.clause_key_to_idx.remove(&clause_key);
        }

        // Remove from units
        self.units.retain(|(_, unit_idx)| *unit_idx != idx);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Get the "shape" of a clause (predicates and polarities)
fn get_clause_shape(clause: &Clause) -> Vec<(crate::logic::PredicateId, bool)> {
    let mut shape: Vec<_> = clause
        .literals
        .iter()
        .map(|lit| (lit.predicate.id, lit.polarity))
        .collect();
    shape.sort();
    shape
}

impl std::fmt::Debug for SubsumptionChecker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubsumptionChecker")
            .field("clauses", &self.clauses.len())
            .field("active", &self.active.len())
            .field("units", &self.units.len())
            .finish()
    }
}
