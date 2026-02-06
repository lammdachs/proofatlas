//! Feature vector indexing for subsumption filtering.
//!
//! Feature vectors provide a necessary condition for subsumption: if clause C subsumes
//! clause D, then feature(C) ≤ feature(D) componentwise. This allows efficient filtering
//! of subsumption candidates using a trie data structure.

use crate::logic::{Clause, FunctionId, PredicateId, Term};
use std::collections::{HashMap, HashSet};

// =============================================================================
// Feature Vector
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
        // Check overlapping elements
        if !self.counts
            .iter()
            .zip(&other.counts)
            .all(|(&a, &b)| a <= b)
        {
            return false;
        }
        // If self (subsumer) has more features than other (target),
        // those extra features in self must be 0 (since 0 ≤ other's implicit 0)
        if self.counts.len() > other.counts.len() {
            if self.counts[other.counts.len()..].iter().any(|&c| c > 0) {
                return false;
            }
        }
        true
    }

    /// Check if this feature vector is compatible with another for backward subsumption.
    /// Returns true if other ≤ self componentwise (self could be subsumed by other).
    #[inline]
    pub fn compatible_as_subsumed(&self, other: &FeatureVector) -> bool {
        if self.literal_count < other.literal_count {
            return false;
        }
        // Check overlapping elements
        if !self.counts
            .iter()
            .zip(&other.counts)
            .all(|(&a, &b)| a >= b)
        {
            return false;
        }
        // If other (source/subsumer) has more features than self (target),
        // those extra features must all be 0 (i.e., target has implicit 0s which are ≥ 0)
        // But if any extra feature in source is non-zero, source can't subsume target
        if other.counts.len() > self.counts.len() {
            if other.counts[self.counts.len()..].iter().any(|&c| c > 0) {
                return false;
            }
        }
        true
    }
}

// =============================================================================
// Symbol Table
// =============================================================================

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

// =============================================================================
// Trie Node (internal)
// =============================================================================

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

// =============================================================================
// Feature Index
// =============================================================================

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
                    .get_or_create_predicate(lit.predicate.id);
                self.collect_function_symbols_from_args(&lit.args);
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
                    .get_or_create_predicate(lit.predicate.id);
                self.collect_function_symbols_from_args(&lit.args);
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
            if let Some(base_idx) = self.symbol_table.get_predicate(lit.predicate.id) {
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
            for term in &lit.args {
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

impl Default for FeatureIndex {
    fn default() -> Self {
        Self::new()
    }
}
