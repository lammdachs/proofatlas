//! Subsumption checking for redundancy elimination in theorem proving
//!
//! This module implements a pragmatic approach to subsumption that balances
//! completeness with performance. Subsumption is a key redundancy elimination
//! technique where a clause C subsumes clause D if there exists a substitution σ
//! such that Cσ ⊆ D (every literal in Cσ appears in D).
//!
//! ## Implementation Strategy
//!
//! Our implementation uses a tiered approach:
//!
//! 1. **Exact Duplicate Detection** (100% complete, O(1))
//!    - Uses string representation hashing for instant detection
//!    - Catches identical clauses regardless of literal order
//!
//! 2. **Variant Detection** (100% complete for variants)
//!    - Detects clauses that are identical up to variable renaming
//!    - Example: P(X,Y) subsumes P(A,B) as a variant
//!
//! 3. **Unit Subsumption** (100% complete for unit clauses)
//!    - Special handling for single-literal clauses
//!    - Very effective in practice as many derived clauses are units
//!
//! 4. **Complete Subsumption for Small Clauses** (≤3 literals)
//!    - Full subsumption checking with proper backtracking
//!    - Feasible for small clauses where the search space is limited
//!
//! 5. **Greedy Heuristic for Large Clauses** (>3 literals)
//!    - Uses a greedy matching strategy that may miss some subsumptions
//!    - Trades completeness for performance on larger clauses
//!
//! ## Design Rationale
//!
//! This design is based on empirical observations:
//! - Most redundant clauses are exact duplicates or variants
//! - Unit clauses are common and unit subsumption is very effective
//! - Full subsumption checking becomes expensive for larger clauses
//! - A greedy heuristic catches many subsumptions with reasonable cost

use crate::logic::{Clause, ClauseKey, Literal, PredicateId, Substitution, Term, VariableId, Interner};
use crate::index::feature_vector::FeatureIndex;
use crate::state::{SimplifyingInference, StateChange};
use indexmap::IndexSet;
use std::collections::{HashMap, HashSet};

// Re-export feature vector types from their canonical location
pub use crate::index::feature_vector::{FeatureVector, SymbolTable};

// =============================================================================
// SubsumptionChecker
// =============================================================================

/// Subsumption checker implementing a balanced redundancy elimination strategy
pub struct SubsumptionChecker {
    /// All clauses indexed by their structural key for duplicate detection
    clause_keys: HashSet<ClauseKey>,

    /// Map from clause key to clause index (for active clauses, used by find_subsumer)
    clause_key_to_idx: HashMap<ClauseKey, usize>,

    /// Unit clauses for unit subsumption
    units: Vec<(Clause, usize)>,

    /// All clauses for subsumption checking
    clauses: Vec<Clause>,

    /// Indices of active clauses (in P ∪ A, not in N)
    active: HashSet<usize>,

    /// Feature vector index for efficient subsumption filtering
    feature_index: FeatureIndex,
}

impl SubsumptionChecker {
    pub fn new() -> Self {
        SubsumptionChecker {
            clause_keys: HashSet::new(),
            clause_key_to_idx: HashMap::new(),
            units: Vec::new(),
            clauses: Vec::new(),
            active: HashSet::new(),
            feature_index: FeatureIndex::new(),
        }
    }

    /// Initialize the feature index with all symbols from the input clauses.
    /// Should be called once before adding any clauses to ensure fixed-dimension feature vectors.
    pub fn initialize_symbols(&mut self, clauses: &[Clause]) {
        self.feature_index.initialize_symbols(clauses);
    }

    /// Add a clause as pending (in N) - not yet active for subsumption
    pub fn add_clause_pending(&mut self, clause: Clause) -> usize {
        let idx = self.clauses.len();
        // Add to feature index (but not activated yet)
        self.feature_index.add_clause(&clause);
        self.clauses.push(clause);
        idx
    }

    /// Activate a pending clause (transfer from N to P)
    pub fn activate_clause(&mut self, idx: usize) {
        if self.active.contains(&idx) {
            return;
        }
        self.active.insert(idx);
        self.feature_index.activate(idx);

        let clause = &self.clauses[idx];

        // Add to key index (structural hashing instead of string formatting)
        let clause_key = ClauseKey::from_clause(clause);
        self.clause_keys.insert(clause_key.clone());
        self.clause_key_to_idx.insert(clause_key, idx);

        // Add to unit index if applicable
        if clause.literals.len() == 1 {
            self.units.push((clause.clone(), idx));
        }
    }

    /// Add a clause and return its index (immediately active)
    pub fn add_clause(&mut self, clause: Clause) -> usize {
        let idx = self.clauses.len();

        // Add to feature index and activate
        self.feature_index.add_clause(&clause);
        self.feature_index.activate(idx);

        // Add to key index (structural hashing instead of string formatting)
        let clause_key = ClauseKey::from_clause(&clause);
        self.clause_keys.insert(clause_key.clone());
        self.clause_key_to_idx.insert(clause_key, idx);

        // Add to unit index if applicable
        if clause.literals.len() == 1 {
            self.units.push((clause.clone(), idx));
        }

        self.clauses.push(clause);
        self.active.insert(idx);
        idx
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
        // Note: units only contains active unit clauses
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

    /// Check if a clause is subsumed by active clauses (those in P ∪ A)
    pub fn is_subsumed(&self, clause: &Clause) -> bool {
        self.find_subsumer(clause).is_some()
    }

    /// Check if a clause is subsumed by any processed clause (excluding itself)
    /// Used in otter loop to check if the simplified given clause is redundant
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
                // Use greedy heuristic for larger clauses
                if compatible_structure(existing, clause) && subsumes_greedy(existing, clause) {
                    return true;
                }
            }
        }

        false
    }

    /// Find which clauses from the given indices are subsumed by the subsumer clause
    /// Returns the indices of subsumed clauses
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
                // Use greedy heuristic for larger clauses
                if subsumes_greedy(subsumer, candidate) {
                    subsumed.push(idx);
                }
            }
        }

        subsumed
    }

    /// Find a variant of this clause among active clauses, returning its index
    fn find_variant(&self, clause: &Clause) -> Option<usize> {
        // Get the clause's "shape" (predicate symbols and polarities)
        let shape = get_clause_shape(clause);

        // Use feature index to get potential subsumers (variants have same features)
        let candidates = self.feature_index.find_potential_subsumers(clause);

        for idx in candidates {
            let existing = &self.clauses[idx];

            // Variants must have same number of literals
            if existing.literals.len() != clause.literals.len() {
                continue;
            }

            // Quick shape check
            if get_clause_shape(existing) != shape {
                continue;
            }

            // Check if they're variants
            if are_variants(existing, clause) {
                return Some(idx);
            }
        }

        None
    }
}

// =============================================================================
// SubsumptionRule (rule adapter)
// =============================================================================

/// Forward subsumption rule.
///
/// Deletes clauses that are subsumed by existing clauses in U∪P.
/// Uses the SubsumptionChecker for efficient subsumption testing.
pub struct SubsumptionRule {
    checker: SubsumptionChecker,
}

impl SubsumptionRule {
    pub fn new(_interner: &Interner) -> Self {
        SubsumptionRule {
            checker: SubsumptionChecker::new(),
        }
    }

    /// Initialize symbols for the feature index
    pub fn initialize_symbols(&mut self, clauses: &[Clause]) {
        self.checker.initialize_symbols(clauses);
    }

    /// Add a clause to the subsumption checker (pending, not yet active)
    pub fn add_clause_pending(&mut self, clause: Clause) -> usize {
        self.checker.add_clause_pending(clause)
    }

    /// Activate a clause (transfer from N to U or P)
    pub fn activate_clause(&mut self, idx: usize) {
        self.checker.activate_clause(idx);
    }

    /// Find clauses subsumed by a given clause
    pub fn find_subsumed_by(&self, subsumer_idx: usize, candidate_indices: &[usize]) -> Vec<usize> {
        self.checker.find_subsumed_by(subsumer_idx, candidate_indices)
    }
}

impl Default for SubsumptionRule {
    fn default() -> Self {
        SubsumptionRule {
            checker: SubsumptionChecker::new(),
        }
    }
}

impl SimplifyingInference for SubsumptionRule {
    fn name(&self) -> &str {
        "Subsumption"
    }

    fn initialize(&mut self, clauses: &[Clause]) {
        self.checker.initialize_symbols(clauses);
    }

    fn on_clause_pending(&mut self, _clause_idx: usize, clause: &Clause) {
        self.checker.add_clause_pending(clause.clone());
    }

    fn on_clause_activated(&mut self, clause_idx: usize, _clause: &Clause) {
        self.checker.activate_clause(clause_idx);
    }

    fn simplify_forward(
        &self,
        clause_idx: usize,
        clause: &Clause,
        _clauses: &[Clause],
        _interner: &Interner,
    ) -> Vec<StateChange> {
        if let Some(_subsumer_idx) = self.checker.find_subsumer(clause) {
            vec![StateChange::Delete { clause_idx, rule_name: self.name().into() }]
        } else {
            vec![]
        }
    }

    fn simplify_backward(
        &self,
        clause_idx: usize,
        _clause: &Clause,
        _clauses: &[Clause],
        unprocessed: &IndexSet<usize>,
        processed: &IndexSet<usize>,
        _interner: &Interner,
    ) -> Vec<StateChange> {
        let mut changes = Vec::new();

        // Collect all candidate indices
        let all_indices: Vec<usize> = unprocessed
            .iter()
            .chain(processed.iter())
            .copied()
            .collect();

        // Find clauses subsumed by this clause
        let subsumed = self.checker.find_subsumed_by(clause_idx, &all_indices);

        let rule_name: String = self.name().into();
        for idx in subsumed {
            if processed.contains(&idx) {
                changes.push(StateChange::Delete { clause_idx: idx, rule_name: rule_name.clone() });
            } else if unprocessed.contains(&idx) {
                changes.push(StateChange::Delete { clause_idx: idx, rule_name: rule_name.clone() });
            }
        }

        changes
    }
}

// =============================================================================
// Pure Subsumption Functions
// =============================================================================

/// Get the "shape" of a clause (predicates and polarities)
fn get_clause_shape(clause: &Clause) -> Vec<(PredicateId, bool)> {
    let mut shape: Vec<_> = clause
        .literals
        .iter()
        .map(|lit| (lit.predicate.id, lit.polarity))
        .collect();
    shape.sort();
    shape
}

/// Check if two clauses are variants (identical up to variable renaming)
pub fn are_variants(clause1: &Clause, clause2: &Clause) -> bool {
    if clause1.literals.len() != clause2.literals.len() {
        return false;
    }

    // Try to find a variable mapping
    let mut var_map: HashMap<VariableId, VariableId> = HashMap::new();

    for (lit1, lit2) in clause1.literals.iter().zip(&clause2.literals) {
        if lit1.polarity != lit2.polarity {
            return false;
        }

        if !literals_match_with_mapping(lit1, lit2, &mut var_map) {
            return false;
        }
    }

    true
}

/// Check if two literals match with a variable mapping (comparing predicate and args directly)
fn literals_match_with_mapping(
    lit1: &Literal,
    lit2: &Literal,
    var_map: &mut HashMap<VariableId, VariableId>,
) -> bool {
    if lit1.predicate != lit2.predicate {
        return false;
    }

    if lit1.args.len() != lit2.args.len() {
        return false;
    }

    for (term1, term2) in lit1.args.iter().zip(&lit2.args) {
        if !terms_match_with_mapping(term1, term2, var_map) {
            return false;
        }
    }

    true
}

/// Check if terms match with a variable mapping
fn terms_match_with_mapping(
    term1: &Term,
    term2: &Term,
    var_map: &mut HashMap<VariableId, VariableId>,
) -> bool {
    match (term1, term2) {
        (Term::Variable(v1), Term::Variable(v2)) => match var_map.get(&v1.id) {
            Some(&mapped) => mapped == v2.id,
            None => {
                var_map.insert(v1.id, v2.id);
                true
            }
        },
        (Term::Constant(c1), Term::Constant(c2)) => c1 == c2,
        (Term::Function(f1, args1), Term::Function(f2, args2)) => {
            f1 == f2
                && args1.len() == args2.len()
                && args1
                    .iter()
                    .zip(args2)
                    .all(|(a1, a2)| terms_match_with_mapping(a1, a2, var_map))
        }
        _ => false,
    }
}

/// Check if a unit clause subsumes another clause using trail-based matching
pub fn subsumes_unit(unit: &Clause, clause: &Clause) -> bool {
    if unit.literals.len() != 1 {
        return false;
    }

    let unit_lit = &unit.literals[0];
    let var_count = count_variables(unit);
    let mut subst = Substitution::with_capacity(var_count);

    // Try to match the unit literal with each literal in the clause
    for lit in &clause.literals {
        if lit.polarity == unit_lit.polarity {
            let mark = subst.mark();
            if match_literals_trail(unit_lit, lit, &mut subst) {
                return true;
            }
            subst.backtrack(mark);
        }
    }

    false
}

/// Full subsumption check using trail-based backtracking
pub fn subsumes(subsumer: &Clause, subsumee: &Clause) -> bool {
    if subsumer.literals.len() > subsumee.literals.len() {
        return false;
    }

    // Use trail-based substitution for efficient backtracking
    let var_count = count_variables(subsumer);
    let mut subst = Substitution::with_capacity(var_count);

    find_subsumption_mapping_trail(
        subsumer,
        subsumee,
        0,
        &mut subst,
        &mut vec![false; subsumee.literals.len()],
    )
}

/// Greedy subsumption for larger clauses using trail-based backtracking
pub fn subsumes_greedy(subsumer: &Clause, subsumee: &Clause) -> bool {
    if subsumer.literals.len() > subsumee.literals.len() {
        return false;
    }

    let var_count = count_variables(subsumer);
    let mut subst = Substitution::with_capacity(var_count);
    let mut used = vec![false; subsumee.literals.len()];

    // Greedy matching: for each literal in subsumer, find the first compatible match
    for subsumer_lit in &subsumer.literals {
        let mut found = false;

        for (i, subsumee_lit) in subsumee.literals.iter().enumerate() {
            if used[i] || subsumee_lit.polarity != subsumer_lit.polarity {
                continue;
            }

            let mark = subst.mark();
            if match_literals_trail(subsumer_lit, subsumee_lit, &mut subst) {
                // For greedy, we commit to this match immediately
                used[i] = true;
                found = true;
                break;
            }
            subst.backtrack(mark);
        }

        if !found {
            return false;
        }
    }

    true
}

/// Check if two clauses have compatible structure for subsumption.
/// Returns true if subsumer's predicates are a subset of subsumee's predicates.
pub fn compatible_structure(clause1: &Clause, clause2: &Clause) -> bool {
    // Check predicate symbols
    let preds1: HashSet<_> = clause1.literals.iter().map(|l| &l.predicate).collect();
    let preds2: HashSet<_> = clause2.literals.iter().map(|l| &l.predicate).collect();

    // subsumer's predicates must be subset of subsumee's
    preds1.is_subset(&preds2)
}

/// Check if two terms are equal
fn terms_equal(term1: &Term, term2: &Term) -> bool {
    match (term1, term2) {
        (Term::Variable(v1), Term::Variable(v2)) => v1 == v2,
        (Term::Constant(c1), Term::Constant(c2)) => c1 == c2,
        (Term::Function(f1, args1), Term::Function(f2, args2)) => {
            f1 == f2
                && args1.len() == args2.len()
                && args1.iter().zip(args2).all(|(a1, a2)| terms_equal(a1, a2))
        }
        _ => false,
    }
}

// =============================================================================
// Trail-Based Subsumption (Phase 3 Optimization)
// =============================================================================

/// Count the number of unique variables in a clause (for capacity hints)
fn count_variables(clause: &Clause) -> usize {
    let mut vars = HashSet::new();
    for lit in &clause.literals {
        for term in &lit.args {
            term.collect_variable_ids(&mut vars);
        }
    }
    vars.len()
}

/// Try to match two terms with a trailed substitution
fn match_terms_trail(term1: &Term, term2: &Term, subst: &mut Substitution) -> bool {
    match term1 {
        Term::Variable(v) => {
            if let Some(bound_term) = subst.get(v.id) {
                terms_equal(bound_term, term2)
            } else {
                subst.bind(*v, term2.clone());
                true
            }
        }
        Term::Constant(c1) => match term2 {
            Term::Constant(c2) => c1 == c2,
            _ => false,
        },
        Term::Function(f1, args1) => match term2 {
            Term::Function(f2, args2) => {
                f1 == f2
                    && args1.len() == args2.len()
                    && args1
                        .iter()
                        .zip(args2)
                        .all(|(a1, a2)| match_terms_trail(a1, a2, subst))
            }
            _ => false,
        },
    }
}

/// Try to match two literals with a trailed substitution
fn match_literals_trail(lit1: &Literal, lit2: &Literal, subst: &mut Substitution) -> bool {
    if lit1.polarity != lit2.polarity {
        return false;
    }

    if lit1.predicate != lit2.predicate {
        return false;
    }

    if lit1.args.len() != lit2.args.len() {
        return false;
    }

    for (term1, term2) in lit1.args.iter().zip(&lit2.args) {
        if !match_terms_trail(term1, term2, subst) {
            return false;
        }
    }

    true
}

/// Recursive function to find subsumption mapping using trail-based backtracking
fn find_subsumption_mapping_trail(
    subsumer: &Clause,
    subsumee: &Clause,
    subsumer_idx: usize,
    subst: &mut Substitution,
    used: &mut Vec<bool>,
) -> bool {
    if subsumer_idx >= subsumer.literals.len() {
        return true; // All literals matched
    }

    let subsumer_lit = &subsumer.literals[subsumer_idx];

    // Try to match with each unused literal in subsumee
    for (i, subsumee_lit) in subsumee.literals.iter().enumerate() {
        if used[i] || subsumee_lit.polarity != subsumer_lit.polarity {
            continue;
        }

        let mark = subst.mark(); // O(1) instead of clone
        if match_literals_trail(subsumer_lit, subsumee_lit, subst) {
            used[i] = true;
            if find_subsumption_mapping_trail(subsumer, subsumee, subsumer_idx + 1, subst, used) {
                return true;
            }
            used[i] = false;
        }
        subst.backtrack(mark); // O(k) undo instead of discard clone
    }

    false
}
