//! Subsumption checker index for forward/backward subsumption filtering.
//!
//! This wraps a literal discrimination tree, clause keys, and unit clause
//! tracking needed for efficient subsumption checking into a single `Index`
//! implementation.

use crate::index::disc_tree::{self, DiscTreeNode};
use crate::logic::{Clause, Literal, PredicateId};
use crate::simplifying::subsumption::{
    are_variants, subsumes, subsumes_greedy, subsumes_unit,
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

// =============================================================================
// Predicate-polarity signature for fast pre-filtering
// =============================================================================

/// Sorted predicate-polarity signature for a clause.
///
/// Stored as a sorted Vec for fast componentwise comparison without hashing.
/// Each entry is (predicate_id, polarity, count).
type PredSignature = Vec<(PredicateId, bool, u16)>;

/// Compute the predicate-polarity signature of a clause.
fn compute_pred_signature(clause: &Clause) -> PredSignature {
    let mut counts: HashMap<(PredicateId, bool), u16> = HashMap::new();
    for lit in &clause.literals {
        *counts.entry((lit.predicate.id, lit.polarity)).or_insert(0) += 1;
    }
    let mut sig: PredSignature = counts.into_iter().map(|((p, pol), c)| (p, pol, c)).collect();
    sig.sort_unstable();
    sig
}

/// Check if `subsumer_sig` is componentwise <= `target_sig`.
///
/// Both signatures must be sorted. Returns true if for every (pred, pol, count)
/// in the subsumer, the target has at least that count.
fn signature_leq(subsumer_sig: &PredSignature, target_sig: &PredSignature) -> bool {
    // Since both are sorted by (pred, pol), we can merge-scan
    let mut ti = 0;
    for &(sp, spol, scount) in subsumer_sig {
        // Advance target iterator to find matching (pred, pol)
        while ti < target_sig.len() && (target_sig[ti].0, target_sig[ti].1) < (sp, spol) {
            ti += 1;
        }
        if ti >= target_sig.len() || target_sig[ti].0 != sp || target_sig[ti].1 != spol {
            return false; // pred-pol not in target
        }
        if scount > target_sig[ti].2 {
            return false; // count exceeds target
        }
    }
    true
}

// =============================================================================
// LiteralDiscTree — per-literal discrimination tree for subsumption filtering
// =============================================================================

/// Discrimination tree that indexes individual literals for subsumption filtering.
///
/// Each literal is indexed under its (predicate, polarity) pair. The argument
/// terms are stored in the same flattened preorder representation as the
/// demodulation tree (shared `disc_tree` implementation).
struct LiteralDiscTree {
    /// Trie roots keyed by (predicate, polarity)
    roots: HashMap<(PredicateId, bool), DiscTreeNode>,
    /// Flat entry lists per (predicate, polarity) for backward subsumption.
    /// Each entry is a clause index; duplicates are OK (gen counter handles dedup).
    /// Periodically compacted to remove deactivated entries.
    flat_entries: HashMap<(PredicateId, bool), Vec<usize>>,
    /// Total entries across all flat_entries lists (for compaction threshold).
    flat_total: usize,
    /// Active clause indices (bitset for O(1) lookup)
    active: Vec<bool>,
    /// Number of active clauses (for compaction threshold).
    active_count: usize,
    /// Clause literal counts (for hit-count filtering)
    clause_lens: Vec<u16>,
    /// Reusable scratch buffers for hit-count filtering (avoids per-call allocation).
    hit_counts: Vec<u16>,
    last_gen: Vec<u16>,
}

impl LiteralDiscTree {
    fn new() -> Self {
        LiteralDiscTree {
            roots: HashMap::new(),
            flat_entries: HashMap::new(),
            flat_total: 0,
            active: Vec::new(),
            active_count: 0,
            clause_lens: Vec::new(),
            hit_counts: Vec::new(),
            last_gen: Vec::new(),
        }
    }

    /// Insert all literals of a clause into the tree and flat index.
    fn insert(&mut self, clause_idx: usize, clause: &Clause) {
        if clause_idx >= self.clause_lens.len() {
            self.clause_lens.resize(clause_idx + 1, 0);
        }
        self.clause_lens[clause_idx] = clause.literals.len() as u16;

        for lit in &clause.literals {
            let key = (lit.predicate.id, lit.polarity);
            let root = self.roots.entry(key).or_default();
            let mut keys = Vec::new();
            for arg in &lit.args {
                disc_tree::flatten_insert(arg, &mut keys);
            }
            disc_tree::trie_insert(root, &keys, clause_idx);

            // Also add to flat index for fast backward subsumption
            self.flat_entries.entry(key).or_default().push(clause_idx);
            self.flat_total += 1;
        }
    }

    /// Mark a clause as active.
    fn activate(&mut self, idx: usize) {
        if idx >= self.active.len() {
            self.active.resize(idx + 1, false);
        }
        if !self.active[idx] {
            self.active[idx] = true;
            self.active_count += 1;
        }
    }

    /// Mark a clause as inactive.
    fn deactivate(&mut self, idx: usize) {
        if idx < self.active.len() && self.active[idx] {
            self.active[idx] = false;
            self.active_count -= 1;
        }
    }

    /// Compact flat_entries by removing deactivated entries.
    /// Called when the flat list has grown too large relative to the active set.
    fn compact_flat_entries(&mut self) {
        let active = &self.active;
        let mut new_total = 0;
        for entries in self.flat_entries.values_mut() {
            entries.retain(|&idx| idx < active.len() && active[idx]);
            new_total += entries.len();
        }
        self.flat_total = new_total;
    }

    /// Find subsumer candidates using interleaved hit-count filtering.
    ///
    /// For C to subsume D, every literal of C must generalize some literal of D.
    /// We process D's literals one by one, counting hits per candidate. A candidate
    /// is checked immediately when its hit count reaches its clause length —
    /// preserving early exit for the common success case while filtering false
    /// positives for the failure case.
    ///
    /// Uses a generation counter for per-literal dedup (O(1) per entry) instead
    /// of sorting each result set.
    fn for_each_subsumer_candidate(
        &mut self,
        clause: &Clause,
        mut callback: impl FnMut(usize) -> bool,
    ) {
        let needed_len = self.clause_lens.len();
        if self.hit_counts.len() < needed_len {
            self.hit_counts.resize(needed_len, 0);
            self.last_gen.resize(needed_len, 0);
        }
        let hit_counts = &mut self.hit_counts;
        let last_gen = &mut self.last_gen;
        let active = &self.active;
        let clause_lens = &self.clause_lens;

        let mut touched = Vec::new();
        let mut early_exit = false;
        let mut gen: u16 = 0;

        for lit in &clause.literals {
            gen = gen.wrapping_add(1);
            let key = (lit.predicate.id, lit.polarity);
            if let Some(root) = self.roots.get(&key) {
                let mut query_keys = Vec::new();
                for arg in &lit.args {
                    disc_tree::flatten_query_vars_concrete(arg, &mut query_keys);
                }
                let mut results = Vec::new();
                disc_tree::retrieve_generalizations(root, &query_keys, 0, &mut results);

                for idx in results {
                    let is_active = idx < active.len() && active[idx];
                    if is_active && last_gen[idx] != gen {
                        last_gen[idx] = gen;
                        if hit_counts[idx] == 0 {
                            touched.push(idx);
                        }
                        hit_counts[idx] += 1;

                        // Check immediately when candidate reaches its threshold
                        if hit_counts[idx] == clause_lens[idx] {
                            if callback(idx) {
                                early_exit = true;
                                break;
                            }
                        }
                    }
                }
                if early_exit {
                    break;
                }
            }
        }

        // Reset touched entries (sparse cleanup)
        for idx in touched {
            hit_counts[idx] = 0;
        }
    }

    /// Find candidate subsumed clauses (backward subsumption).
    ///
    /// Hybrid approach: for each subsumer literal, chooses between:
    /// - Flat entry list: when all arguments are variables (Star fan-out makes
    ///   trie traversal O(trie_size) with poor cache locality)
    /// - Disc tree `retrieve_instances`: when some arguments are concrete
    ///   (trie prunes to matching branches efficiently)
    ///
    /// Hit-count intersection with generation-counter dedup then filters to
    /// candidates where every subsumer literal found a match.
    fn find_subsumed_candidates(&mut self, clause: &Clause) -> Vec<usize> {
        // Compact when flat lists grow too large relative to active set
        if self.flat_total > self.active_count * 9 + 1000 {
            self.compact_flat_entries();
        }

        let needed_len = self.clause_lens.len();
        if self.hit_counts.len() < needed_len {
            self.hit_counts.resize(needed_len, 0);
            self.last_gen.resize(needed_len, 0);
        }
        let hit_counts = &mut self.hit_counts;
        let last_gen = &mut self.last_gen;
        let active = &self.active;

        let subsumer_len = clause.literals.len() as u16;
        let mut touched = Vec::new();
        let mut gen: u16 = 0;

        for lit in &clause.literals {
            gen = gen.wrapping_add(1);
            let key = (lit.predicate.id, lit.polarity);

            // Check if literal has any concrete (non-variable) arguments
            let has_concrete_args = lit.args.iter().any(|arg| !matches!(arg, crate::logic::Term::Variable(_)));

            if has_concrete_args {
                // Use disc tree — concrete args prune branches efficiently
                if let Some(root) = self.roots.get(&key) {
                    let mut query_keys = Vec::new();
                    for arg in &lit.args {
                        disc_tree::flatten_insert(arg, &mut query_keys);
                    }
                    let mut results = Vec::new();
                    disc_tree::retrieve_instances(root, &query_keys, 0, &mut results);

                    for idx in results {
                        let is_active = idx < active.len() && active[idx];
                        if is_active && last_gen[idx] != gen {
                            last_gen[idx] = gen;
                            if hit_counts[idx] == 0 {
                                touched.push(idx);
                            }
                            hit_counts[idx] += 1;
                        }
                    }
                } else {
                    for idx in touched { hit_counts[idx] = 0; }
                    return Vec::new();
                }
            } else {
                // Use flat list — all args are variables, trie provides no filtering
                if let Some(entries) = self.flat_entries.get(&key) {
                    for &idx in entries {
                        let is_active = idx < active.len() && active[idx];
                        if is_active && last_gen[idx] != gen {
                            last_gen[idx] = gen;
                            if hit_counts[idx] == 0 {
                                touched.push(idx);
                            }
                            hit_counts[idx] += 1;
                        }
                    }
                } else {
                    for idx in touched { hit_counts[idx] = 0; }
                    return Vec::new();
                }
            }
        }

        // Collect candidates where all subsumer literals matched
        let mut candidates = Vec::new();
        for idx in &touched {
            if hit_counts[*idx] == subsumer_len {
                candidates.push(*idx);
            }
        }

        // Sparse cleanup
        for idx in touched {
            hit_counts[idx] = 0;
        }

        candidates
    }
}

// =============================================================================
// SubsumptionChecker
// =============================================================================

/// Subsumption checker implementing a balanced redundancy elimination strategy.
///
/// Maintains internal indices (literal discrimination tree, clause keys, unit
/// clauses) and tracks the clause lifecycle via the `Index` trait.
pub struct SubsumptionChecker {
    /// Map from clause literals to clause index (for exact duplicate detection)
    clause_key_to_idx: HashMap<Vec<Literal>, usize>,

    /// Unit clauses for unit subsumption
    units: Vec<(Arc<Clause>, usize)>,

    /// All clauses for subsumption checking (Arc avoids deep clone on add)
    clauses: Vec<Arc<Clause>>,

    /// Precomputed predicate-polarity signatures per clause
    pred_signatures: Vec<PredSignature>,

    /// Indices of active clauses (in U + P)
    active: HashSet<usize>,

    /// Literal discrimination tree for efficient subsumption filtering
    literal_tree: LiteralDiscTree,
}

impl SubsumptionChecker {
    pub fn new() -> Self {
        SubsumptionChecker {
            clause_key_to_idx: HashMap::new(),
            units: Vec::new(),
            clauses: Vec::new(),
            pred_signatures: Vec::new(),
            active: HashSet::new(),
            literal_tree: LiteralDiscTree::new(),
        }
    }

    /// Find the active clause that subsumes the given clause, returning its index.
    /// Returns None if no active clause subsumes it.
    pub fn find_subsumer(&mut self, clause: &Clause) -> Option<usize> {
        // 1. Check for exact duplicates (very fast with structural hashing)
        if let Some(&idx) = self.clause_key_to_idx.get(clause.literals.as_slice()) {
            return Some(idx);
        }

        // 2. Unit subsumption (fast and complete)
        for (unit, idx) in &self.units {
            if subsumes_unit(unit, clause) {
                return Some(*idx);
            }
        }

        // 3. Use literal tree with fused candidate checking
        let target_sig = compute_pred_signature(clause);
        let shape = get_clause_shape(clause);
        let clause_len = clause.literals.len();

        let mut found = None;
        let clauses = &self.clauses;
        let pred_signatures = &self.pred_signatures;
        self.literal_tree
            .for_each_subsumer_candidate(clause, |idx| {
                let existing = &clauses[idx];
                let existing_len = existing.literals.len();

                // Same size: check for variants
                if existing_len == clause_len {
                    if get_clause_shape(existing) == shape && are_variants(existing, clause) {
                        found = Some(idx);
                        return true; // stop
                    }
                    return false; // continue
                }

                // Skip if larger (can't subsume)
                if existing_len >= clause_len {
                    return false;
                }

                // Skip unit clauses (already handled above)
                if existing_len == 1 {
                    return false;
                }

                // Predicate-polarity signature filter (precomputed, no allocation)
                if !signature_leq(&pred_signatures[idx], &target_sig) {
                    return false;
                }

                // For small clauses (2-3 literals), do complete subsumption
                if existing_len <= 3 {
                    if subsumes(existing, clause) {
                        found = Some(idx);
                        return true;
                    }
                } else if subsumes_greedy(existing, clause) {
                    found = Some(idx);
                    return true;
                }

                false
            });

        found
    }

    /// Check if a clause is subsumed by any processed clause (excluding itself)
    pub fn is_subsumed_by_processed(&mut self, exclude_idx: usize, clause: &Clause) -> bool {
        // Unit subsumption (excluding self)
        for (unit, idx) in &self.units {
            if *idx == exclude_idx {
                continue;
            }
            if subsumes_unit(unit, clause) {
                return true;
            }
        }

        // Use literal tree with fused candidate checking
        let target_sig = compute_pred_signature(clause);
        let clause_len = clause.literals.len();

        let mut found = false;
        let clauses = &self.clauses;
        let pred_signatures = &self.pred_signatures;
        self.literal_tree
            .for_each_subsumer_candidate(clause, |idx| {
                if idx == exclude_idx {
                    return false;
                }

                let existing = &clauses[idx];
                let existing_len = existing.literals.len();

                // Check for variants (same size)
                if existing_len == clause_len {
                    if are_variants(existing, clause) {
                        found = true;
                        return true;
                    }
                    return false;
                }

                // Skip if existing is larger (can't subsume)
                if existing_len > clause_len {
                    return false;
                }

                // Skip unit clauses (already handled above)
                if existing_len == 1 {
                    return false;
                }

                // Predicate-polarity signature filter
                if !signature_leq(&pred_signatures[idx], &target_sig) {
                    return false;
                }

                // Full subsumption for small clauses
                if existing_len <= 3 {
                    if subsumes(existing, clause) {
                        found = true;
                        return true;
                    }
                } else if subsumes_greedy(existing, clause) {
                    found = true;
                    return true;
                }

                false
            });

        found
    }

    /// Find which clauses from the given indices are subsumed by the subsumer clause.
    /// Returns the indices of subsumed clauses.
    ///
    /// The `candidate_indices` parameter is kept for API compatibility but is not
    /// used — the literal tree's active set already tracks U∪P membership, so
    /// `find_subsumed_candidates` returns only clauses in U∪P. Iterating the
    /// (small) tree result set directly avoids O(|U∪P|) iteration per call.
    pub fn find_subsumed_by(
        &mut self,
        subsumer_idx: usize,
        _candidate_indices: &[usize],
    ) -> Vec<usize> {
        let subsumer = &self.clauses[subsumer_idx];
        let mut subsumed = Vec::new();

        let tree_candidates = self.literal_tree.find_subsumed_candidates(subsumer);

        let subsumer_sig = &self.pred_signatures[subsumer_idx];

        for idx in tree_candidates {
            if idx == subsumer_idx {
                continue;
            }

            let candidate = &self.clauses[idx];

            // Quick check: subsumer can't be larger
            if subsumer.literals.len() > candidate.literals.len() {
                continue;
            }

            // Predicate-polarity signature filter (precomputed, no allocation)
            if !signature_leq(subsumer_sig, &self.pred_signatures[idx]) {
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
}

impl Default for SubsumptionChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl SubsumptionChecker {
    pub fn on_add(&mut self, idx: usize, clause: &Arc<Clause>) {
        // Add clause to internal store + literal tree
        self.literal_tree.insert(idx, clause);
        // Ensure vecs are large enough
        if idx >= self.clauses.len() {
            self.clauses.resize_with(idx + 1, || Arc::new(Clause::new(vec![])));
            self.pred_signatures.resize(idx + 1, Vec::new());
        }
        self.clauses[idx] = Arc::clone(clause);
        self.pred_signatures[idx] = compute_pred_signature(clause);
    }

    pub fn on_transfer(&mut self, idx: usize, _clause: &Arc<Clause>) {
        if self.active.contains(&idx) {
            return;
        }
        self.active.insert(idx);
        self.literal_tree.activate(idx);

        // Use the stored (oriented) clause for keys and units
        let clause = &self.clauses[idx];

        // Add to duplicate detection index
        self.clause_key_to_idx.insert(clause.literals.to_vec(), idx);

        // Add to unit index if applicable
        if clause.literals.len() == 1 {
            self.units.push((Arc::clone(clause), idx));
        }
    }

    pub fn on_delete(&mut self, idx: usize, _clause: &Arc<Clause>) {
        self.active.remove(&idx);
        self.literal_tree.deactivate(idx);

        // Remove from duplicate detection index
        let clause = &self.clauses[idx];
        if self.clause_key_to_idx.get(clause.literals.as_slice()) == Some(&idx) {
            self.clause_key_to_idx.remove(clause.literals.as_slice());
        }

        // Remove from units
        self.units.retain(|(_, unit_idx)| *unit_idx != idx);
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
