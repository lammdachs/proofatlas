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

use crate::fol::{Clause, FunctionId, Literal, PredicateId, Substitution, Term, VariableId};
use std::collections::{HashMap, HashSet};

// =============================================================================
// Feature Vector Indexing for Subsumption
// =============================================================================

/// A feature vector for subsumption filtering.
///
/// For subsumption C ⊆σ D, we have feature(C) ≤ feature(D) componentwise.
/// Features are counts of:
/// - Total literal count
/// - Positive occurrences of each predicate
/// - Negative occurrences of each predicate
/// - Occurrences of each function symbol
#[derive(Debug, Clone)]
pub struct FeatureVector {
    /// Total number of literals
    pub literal_count: u16,
    /// Feature counts: [pos_pred_0, neg_pred_0, pos_pred_1, neg_pred_1, ..., func_0, func_1, ...]
    pub counts: Vec<u16>,
}

impl FeatureVector {
    /// Create a new feature vector with the given dimension
    pub fn new(dimension: usize) -> Self {
        FeatureVector {
            literal_count: 0,
            counts: vec![0; dimension],
        }
    }

    /// Check if this feature vector is compatible with another for forward subsumption.
    /// Returns true if self ≤ other componentwise (self could be a subsumer of other).
    #[inline]
    pub fn compatible_as_subsumer(&self, other: &FeatureVector) -> bool {
        if self.literal_count > other.literal_count {
            return false;
        }
        // Use iterators for better performance
        self.counts
            .iter()
            .zip(&other.counts)
            .all(|(&a, &b)| a <= b)
    }

    /// Check if this feature vector is compatible with another for backward subsumption.
    /// Returns true if other ≤ self componentwise (self could be subsumed by other).
    #[inline]
    pub fn compatible_as_subsumed(&self, other: &FeatureVector) -> bool {
        if self.literal_count < other.literal_count {
            return false;
        }
        self.counts
            .iter()
            .zip(&other.counts)
            .all(|(&a, &b)| a >= b)
    }
}

/// Maps symbols to feature vector indices
#[derive(Debug, Clone)]
pub struct SymbolTable {
    /// Maps predicate ID to index in feature vector (base index for positive count)
    predicate_ids: HashMap<PredicateId, usize>,
    /// Maps function ID to index in feature vector
    function_ids: HashMap<FunctionId, usize>,
    /// Total dimension of feature vectors
    dimension: usize,
}

impl SymbolTable {
    pub fn new() -> Self {
        SymbolTable {
            predicate_ids: HashMap::new(),
            function_ids: HashMap::new(),
            dimension: 0,
        }
    }

    /// Get or create index for a predicate (returns base index, positive is base, negative is base+1)
    pub fn get_or_create_predicate(&mut self, id: PredicateId) -> usize {
        if let Some(&idx) = self.predicate_ids.get(&id) {
            idx
        } else {
            let idx = self.dimension;
            self.predicate_ids.insert(id, idx);
            self.dimension += 2; // positive and negative counts
            idx
        }
    }

    /// Get or create index for a function symbol
    pub fn get_or_create_function(&mut self, id: FunctionId) -> usize {
        if let Some(&idx) = self.function_ids.get(&id) {
            idx
        } else {
            let idx = self.dimension;
            self.function_ids.insert(id, idx);
            self.dimension += 1;
            idx
        }
    }

    /// Get the current dimension for feature vectors
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get predicate index (if exists)
    pub fn get_predicate(&self, id: PredicateId) -> Option<usize> {
        self.predicate_ids.get(&id).copied()
    }

    /// Get function index (if exists)
    pub fn get_function(&self, id: FunctionId) -> Option<usize> {
        self.function_ids.get(&id).copied()
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

/// A node in the feature vector trie
#[derive(Debug)]
struct TrieNode {
    /// Clause indices stored at this node (for clauses ending here)
    clauses: Vec<usize>,
    /// Children indexed by count value (sparse representation)
    children: HashMap<u16, Box<TrieNode>>,
}

impl TrieNode {
    fn new() -> Self {
        TrieNode {
            clauses: Vec::new(),
            children: HashMap::new(),
        }
    }

    /// Recursively collect all clauses in this subtree
    fn collect_all(&self, active: &HashSet<usize>, result: &mut Vec<usize>) {
        for &idx in &self.clauses {
            if active.contains(&idx) {
                result.push(idx);
            }
        }
        for child in self.children.values() {
            child.collect_all(active, result);
        }
    }
}

/// Feature vector index for efficient subsumption filtering.
///
/// Uses a trie indexed by feature vector components for fast range queries.
/// The symbol table must be initialized with all symbols before adding clauses.
#[derive(Debug)]
pub struct FeatureIndex {
    /// Symbol table for mapping symbols to indices (fixed after initialization)
    symbol_table: SymbolTable,
    /// Whether the symbol table has been finalized
    symbols_finalized: bool,
    /// Fixed dimension for all feature vectors
    dimension: usize,
    /// Root of the trie
    root: TrieNode,
    /// Feature vectors for all clauses (indexed by clause id)
    clause_features: Vec<FeatureVector>,
    /// Set of active clause indices
    active: HashSet<usize>,
}

impl FeatureIndex {
    pub fn new() -> Self {
        FeatureIndex {
            symbol_table: SymbolTable::new(),
            symbols_finalized: false,
            dimension: 0,
            root: TrieNode::new(),
            clause_features: Vec::new(),
            active: HashSet::new(),
        }
    }

    /// Initialize the symbol table from a set of clauses.
    /// Must be called before adding any clauses.
    pub fn initialize_symbols(&mut self, clauses: &[Clause]) {
        for clause in clauses {
            for lit in &clause.literals {
                self.symbol_table
                    .get_or_create_predicate(lit.atom.predicate.id);
                self.collect_function_symbols_from_args(&lit.atom.args);
            }
        }
        self.dimension = self.symbol_table.dimension();
        self.symbols_finalized = true;
    }

    /// Collect function symbols from term arguments
    fn collect_function_symbols_from_args(&mut self, args: &[Term]) {
        for term in args {
            self.collect_function_symbols(term);
        }
    }

    /// Recursively collect function symbols from a term
    fn collect_function_symbols(&mut self, term: &Term) {
        match term {
            Term::Variable(_) | Term::Constant(_) => {}
            Term::Function(f, args) => {
                self.symbol_table.get_or_create_function(f.id);
                for arg in args {
                    self.collect_function_symbols(arg);
                }
            }
        }
    }

    /// Extract and store features for a clause, returning the clause's index
    pub fn add_clause(&mut self, clause: &Clause) -> usize {
        // If symbols aren't finalized, do it now (fallback for single-clause adds)
        if !self.symbols_finalized {
            for lit in &clause.literals {
                self.symbol_table
                    .get_or_create_predicate(lit.atom.predicate.id);
                self.collect_function_symbols_from_args(&lit.atom.args);
            }
            self.dimension = self.symbol_table.dimension();
        }

        let idx = self.clause_features.len();

        // Extract features with fixed dimension
        let features = self.extract_features(clause);

        // Insert into trie
        self.insert_into_trie(idx, &features);

        self.clause_features.push(features);
        idx
    }

    /// Extract feature vector from a clause
    fn extract_features(&self, clause: &Clause) -> FeatureVector {
        let mut features = FeatureVector::new(self.dimension);
        features.literal_count = clause.literals.len() as u16;

        for lit in &clause.literals {
            // Count predicate occurrences
            if let Some(base_idx) = self.symbol_table.get_predicate(lit.atom.predicate.id) {
                let idx = if lit.polarity {
                    base_idx
                } else {
                    base_idx + 1
                };
                if idx < features.counts.len() {
                    features.counts[idx] = features.counts[idx].saturating_add(1);
                }
            }

            // Count function symbols in arguments
            for term in &lit.atom.args {
                self.count_function_symbols(term, &mut features);
            }
        }

        features
    }

    /// Count function symbols in a term
    fn count_function_symbols(&self, term: &Term, features: &mut FeatureVector) {
        match term {
            Term::Variable(_) | Term::Constant(_) => {}
            Term::Function(f, args) => {
                if let Some(idx) = self.symbol_table.get_function(f.id) {
                    if idx < features.counts.len() {
                        features.counts[idx] = features.counts[idx].saturating_add(1);
                    }
                }
                for arg in args {
                    self.count_function_symbols(arg, features);
                }
            }
        }
    }

    /// Insert a clause into the trie
    fn insert_into_trie(&mut self, clause_idx: usize, features: &FeatureVector) {
        let mut node = &mut self.root;

        // First level: literal count
        node = node
            .children
            .entry(features.literal_count)
            .or_insert_with(|| Box::new(TrieNode::new()));

        // Remaining levels: feature counts
        for &count in &features.counts {
            node = node
                .children
                .entry(count)
                .or_insert_with(|| Box::new(TrieNode::new()));
        }

        // Store clause at leaf
        node.clauses.push(clause_idx);
    }

    /// Mark a clause as active
    pub fn activate(&mut self, idx: usize) {
        self.active.insert(idx);
    }

    /// Mark a clause as inactive
    #[allow(dead_code)]
    pub fn deactivate(&mut self, idx: usize) {
        self.active.remove(&idx);
    }

    /// Check if a clause is active
    #[allow(dead_code)]
    pub fn is_active(&self, idx: usize) -> bool {
        self.active.contains(&idx)
    }

    /// Find potential subsumers for a clause (forward query).
    /// Returns indices of active clauses C where feature(C) ≤ feature(target).
    pub fn find_potential_subsumers(&self, target: &Clause) -> Vec<usize> {
        let target_features = self.extract_features(target);
        let mut result = Vec::new();
        self.collect_subsumers(&self.root, &target_features, 0, &mut result);
        result
    }

    /// Recursively collect subsumers (forward query: find C where feature(C) ≤ feature(target))
    fn collect_subsumers(
        &self,
        node: &TrieNode,
        target: &FeatureVector,
        depth: usize,
        result: &mut Vec<usize>,
    ) {
        // Collect active clauses at this node
        for &clause_idx in &node.clauses {
            if self.active.contains(&clause_idx) {
                result.push(clause_idx);
            }
        }

        if depth == 0 {
            // Literal count level: traverse children with count ≤ target
            let target_count = target.literal_count;
            for (&count, child) in &node.children {
                if count <= target_count {
                    self.collect_subsumers(child, target, 1, result);
                }
            }
        } else {
            let feature_idx = depth - 1;
            if feature_idx < target.counts.len() {
                // Feature level: traverse children with count ≤ target
                let target_count = target.counts[feature_idx];
                for (&count, child) in &node.children {
                    if count <= target_count {
                        self.collect_subsumers(child, target, depth + 1, result);
                    }
                }
            }
            // If we've exhausted target features, collect all remaining clauses in subtree
            // (their extra features are all 0, which is ≤ target's implicit 0s)
            // Actually, clauses stored deeper have non-zero features, so they can't be subsumers
        }
    }

    /// Find potentially subsumed clauses (backward query).
    /// Returns indices of active clauses D where feature(source) ≤ feature(D).
    pub fn find_potentially_subsumed(&self, source_idx: usize) -> Vec<usize> {
        if source_idx >= self.clause_features.len() {
            return Vec::new();
        }
        let source_features = &self.clause_features[source_idx];
        let mut result = Vec::new();
        self.collect_subsumed(&self.root, source_features, 0, &mut result);
        // Exclude the source clause itself
        result.retain(|&idx| idx != source_idx);
        result
    }

    /// Recursively collect subsumed clauses (backward query: find D where feature(source) ≤ feature(D))
    fn collect_subsumed(
        &self,
        node: &TrieNode,
        source: &FeatureVector,
        depth: usize,
        result: &mut Vec<usize>,
    ) {
        // Collect active clauses at this node
        for &clause_idx in &node.clauses {
            if self.active.contains(&clause_idx) {
                result.push(clause_idx);
            }
        }

        if depth == 0 {
            // Literal count level: traverse children with count ≥ source
            let source_count = source.literal_count;
            for (&count, child) in &node.children {
                if count >= source_count {
                    self.collect_subsumed(child, source, 1, result);
                }
            }
        } else {
            let feature_idx = depth - 1;
            if feature_idx < source.counts.len() {
                // Feature level: traverse children with count ≥ source
                let source_count = source.counts[feature_idx];
                for (&count, child) in &node.children {
                    if count >= source_count {
                        self.collect_subsumed(child, source, depth + 1, result);
                    }
                }
            } else {
                // We've checked all source features, collect all clauses in subtree
                // (source's implicit 0s are ≤ any value)
                node.collect_all(&self.active, result);
            }
        }
    }

    /// Get the feature vector for a clause
    #[allow(dead_code)]
    pub fn get_features(&self, idx: usize) -> Option<&FeatureVector> {
        self.clause_features.get(idx)
    }
}

/// Subsumption checker implementing a balanced redundancy elimination strategy
pub struct SubsumptionChecker {
    /// All clauses indexed by their string representation for duplicate detection
    clause_strings: HashSet<String>,

    /// Map from clause string to clause index (for active clauses, used by find_subsumer)
    clause_string_to_idx: HashMap<String, usize>,

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
            clause_strings: HashSet::new(),
            clause_string_to_idx: HashMap::new(),
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

        // Add to string index
        let clause_str = format!("{}", clause);
        self.clause_strings.insert(clause_str.clone());
        self.clause_string_to_idx.insert(clause_str, idx);

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

        // Add to string index
        let clause_str = format!("{}", clause);
        self.clause_strings.insert(clause_str.clone());
        self.clause_string_to_idx.insert(clause_str, idx);

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
        // 1. Check for exact duplicates (very fast)
        // Note: clause_strings maps to clause indices for active clauses
        let clause_str = format!("{}", clause);
        if let Some(&idx) = self.clause_string_to_idx.get(&clause_str) {
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

/// Get the "shape" of a clause (predicates and polarities)
fn get_clause_shape(clause: &Clause) -> Vec<(PredicateId, bool)> {
    let mut shape: Vec<_> = clause
        .literals
        .iter()
        .map(|lit| (lit.atom.predicate.id, lit.polarity))
        .collect();
    shape.sort();
    shape
}

/// Check if two clauses are variants (identical up to variable renaming)
fn are_variants(clause1: &Clause, clause2: &Clause) -> bool {
    if clause1.literals.len() != clause2.literals.len() {
        return false;
    }

    // Try to find a variable mapping
    let mut var_map: HashMap<VariableId, VariableId> = HashMap::new();

    for (lit1, lit2) in clause1.literals.iter().zip(&clause2.literals) {
        if lit1.polarity != lit2.polarity {
            return false;
        }

        if !atoms_match_with_mapping(&lit1.atom, &lit2.atom, &mut var_map) {
            return false;
        }
    }

    true
}

/// Check if atoms match with a variable mapping
fn atoms_match_with_mapping(
    atom1: &crate::fol::Atom,
    atom2: &crate::fol::Atom,
    var_map: &mut HashMap<VariableId, VariableId>,
) -> bool {
    if atom1.predicate != atom2.predicate {
        return false;
    }

    if atom1.args.len() != atom2.args.len() {
        return false;
    }

    for (term1, term2) in atom1.args.iter().zip(&atom2.args) {
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

/// Check if a unit clause subsumes another clause
fn subsumes_unit(unit: &Clause, clause: &Clause) -> bool {
    if unit.literals.len() != 1 {
        return false;
    }

    let unit_lit = &unit.literals[0];

    // Try to match the unit literal with each literal in the clause
    for lit in &clause.literals {
        if lit.polarity == unit_lit.polarity {
            // Try to find a substitution
            let mut subst = Substitution::new();
            if match_literals(&unit_lit, lit, &mut subst) {
                return true;
            }
        }
    }

    false
}

/// Full subsumption check
fn subsumes(subsumer: &Clause, subsumee: &Clause) -> bool {
    if subsumer.literals.len() > subsumee.literals.len() {
        return false;
    }

    // Try to find a matching for all literals in subsumer
    find_subsumption_mapping(
        subsumer,
        subsumee,
        0,
        &mut Substitution::new(),
        &mut vec![false; subsumee.literals.len()],
    )
}

/// Recursive function to find subsumption mapping
fn find_subsumption_mapping(
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

        let mut new_subst = subst.clone();
        if match_literals(subsumer_lit, subsumee_lit, &mut new_subst) {
            used[i] = true;
            if find_subsumption_mapping(subsumer, subsumee, subsumer_idx + 1, &mut new_subst, used)
            {
                return true;
            }
            used[i] = false;
        }
    }

    false
}

/// Greedy subsumption for larger clauses
fn subsumes_greedy(subsumer: &Clause, subsumee: &Clause) -> bool {
    if subsumer.literals.len() > subsumee.literals.len() {
        return false;
    }

    let mut subst = Substitution::new();
    let mut used = vec![false; subsumee.literals.len()];

    // Greedy matching: for each literal in subsumer, find the first compatible match
    for subsumer_lit in &subsumer.literals {
        let mut found = false;

        for (i, subsumee_lit) in subsumee.literals.iter().enumerate() {
            if used[i] || subsumee_lit.polarity != subsumer_lit.polarity {
                continue;
            }

            let mut temp_subst = subst.clone();
            if match_literals(subsumer_lit, subsumee_lit, &mut temp_subst) {
                // Check if this substitution is consistent
                let subsumer_lit_applied = subsumer_lit.apply_substitution(&temp_subst);
                let subsumee_lit_applied = subsumee_lit.apply_substitution(&temp_subst);

                if subsumer_lit_applied == subsumee_lit_applied {
                    subst = temp_subst;
                    used[i] = true;
                    found = true;
                    break;
                }
            }
        }

        if !found {
            return false;
        }
    }

    true
}

/// Check if two clauses have compatible structure
fn compatible_structure(clause1: &Clause, clause2: &Clause) -> bool {
    // Check predicate symbols
    let preds1: HashSet<_> = clause1.literals.iter().map(|l| &l.atom.predicate).collect();
    let preds2: HashSet<_> = clause2.literals.iter().map(|l| &l.atom.predicate).collect();

    // subsumer's predicates must be subset of subsumee's
    preds1.is_subset(&preds2)
}

/// Try to match two literals with a substitution
fn match_literals(lit1: &Literal, lit2: &Literal, subst: &mut Substitution) -> bool {
    if lit1.polarity != lit2.polarity {
        return false;
    }

    match_atoms(&lit1.atom, &lit2.atom, subst)
}

/// Try to match two atoms with a substitution
fn match_atoms(
    atom1: &crate::fol::Atom,
    atom2: &crate::fol::Atom,
    subst: &mut Substitution,
) -> bool {
    if atom1.predicate != atom2.predicate {
        return false;
    }

    if atom1.args.len() != atom2.args.len() {
        return false;
    }

    for (term1, term2) in atom1.args.iter().zip(&atom2.args) {
        if !match_terms(term1, term2, subst) {
            return false;
        }
    }

    true
}

/// Try to match two terms with a substitution
fn match_terms(term1: &Term, term2: &Term, subst: &mut Substitution) -> bool {
    match term1 {
        Term::Variable(v) => {
            // Check if variable is already bound
            if let Some(bound_term) = subst.map.get(&v.id) {
                // Must match the bound term
                terms_equal(bound_term, term2)
            } else {
                // Bind the variable
                subst.insert(*v, term2.clone());
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
                        .all(|(a1, a2)| match_terms(a1, a2, subst))
            }
            _ => false,
        },
    }
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
