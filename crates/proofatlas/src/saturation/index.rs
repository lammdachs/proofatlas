//! Registry-based index architecture for saturation-based theorem proving.
//!
//! This module provides a centralized index management system where:
//! - Rules declare which indices they need via `required_indices()`
//! - SaturationState creates only the needed indices in an `IndexRegistry`
//! - On clause lifecycle events, the registry routes updates to all indices
//! - At rule invocation time, an `IndexProvider` gives read-only access
//!
//! ## Index Types
//!
//! - `UnitClauses`: Single-literal clauses for unit subsumption
//! - `UnitEqualities`: Unit positive equalities for demodulation
//! - `FeatureVectors`: Feature vector trie for subsumption filtering
//! - `ClauseKeys`: Structural hashing for duplicate detection
//!
//! ## Design
//!
//! The registry pattern decouples index management from rules:
//! - Rules don't maintain internal state; they declare requirements
//! - Indices are created once and shared across rules that need them
//! - Lifecycle events are routed atomically to all indices

use crate::fol::{Clause, ClauseKey, Interner, PredicateId};
use indexmap::IndexSet;
use std::any::Any;
use std::collections::{HashMap, HashSet};

use super::subsumption::FeatureIndex;


// =============================================================================
// Index Kind Enumeration
// =============================================================================

/// Types of indices available for rules to request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexKind {
    /// Single-literal clauses for unit subsumption
    UnitClauses,
    /// Unit positive equalities (subset of UnitClauses) for demodulation
    UnitEqualities,
    /// Feature vector trie for subsumption filtering
    FeatureVectors,
    /// Structural hashing for duplicate detection
    ClauseKeys,
}

// =============================================================================
// Index Trait
// =============================================================================

/// Trait for index implementations that track clause lifecycle events.
///
/// Indices receive notifications when clauses enter/exit different states:
/// - `on_clause_pending`: Clause added to N (new, awaiting forward simplification)
/// - `on_clause_activated`: Clause transferred from N to U (survived forward simplification)
/// - `on_clause_removed`: Clause deleted from U or P (backward simplification)
pub trait Index: Send + Sync {
    /// Get the kind of this index
    fn kind(&self) -> IndexKind;

    /// Called when a clause is added to N (pending, not yet active).
    /// Some indices (like FeatureVectors) pre-compute features here.
    fn on_clause_pending(&mut self, idx: usize, clause: &Clause);

    /// Called when a clause is activated (transferred from N to U).
    /// Most indices start tracking the clause here.
    fn on_clause_activated(&mut self, idx: usize, clause: &Clause);

    /// Called when a clause is removed from U or P.
    /// Indices should stop tracking the clause.
    fn on_clause_removed(&mut self, idx: usize, clause: &Clause);

    /// Initialize the index with all input clauses (for pre-computing symbol tables, etc.)
    fn initialize(&mut self, _clauses: &[Clause]) {}

    /// Get as Any for downcasting
    fn as_any(&self) -> &dyn Any;
}

// =============================================================================
// UnitClausesIndex
// =============================================================================

/// Index tracking single-literal clauses for efficient unit subsumption.
#[derive(Debug)]
pub struct UnitClausesIndex {
    /// Set of active unit clause indices
    units: IndexSet<usize>,
}

impl UnitClausesIndex {
    pub fn new() -> Self {
        UnitClausesIndex {
            units: IndexSet::new(),
        }
    }

    /// Get all active unit clause indices
    pub fn iter(&self) -> impl Iterator<Item = &usize> {
        self.units.iter()
    }

    /// Check if an index is in the unit set
    pub fn contains(&self, idx: &usize) -> bool {
        self.units.contains(idx)
    }

    /// Get the number of unit clauses
    pub fn len(&self) -> usize {
        self.units.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.units.is_empty()
    }
}

impl Default for UnitClausesIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl Index for UnitClausesIndex {
    fn kind(&self) -> IndexKind {
        IndexKind::UnitClauses
    }

    fn on_clause_pending(&mut self, _idx: usize, _clause: &Clause) {
        // Units are only tracked when activated
    }

    fn on_clause_activated(&mut self, idx: usize, clause: &Clause) {
        if clause.literals.len() == 1 {
            self.units.insert(idx);
        }
    }

    fn on_clause_removed(&mut self, idx: usize, _clause: &Clause) {
        self.units.shift_remove(&idx);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// UnitEqualitiesIndex
// =============================================================================

/// Index tracking unit positive equalities for demodulation.
///
/// This is a subset of UnitClauses that only includes positive equalities.
#[derive(Debug)]
pub struct UnitEqualitiesIndex {
    /// Set of active unit equality clause indices
    unit_equalities: IndexSet<usize>,
    /// Predicate ID for equality (cached, None if "=" not interned)
    equality_pred_id: Option<PredicateId>,
}

impl UnitEqualitiesIndex {
    pub fn new(interner: &Interner) -> Self {
        UnitEqualitiesIndex {
            unit_equalities: IndexSet::new(),
            equality_pred_id: interner.get_predicate("="),
        }
    }

    /// Get all active unit equality clause indices
    pub fn iter(&self) -> impl Iterator<Item = &usize> {
        self.unit_equalities.iter()
    }

    /// Check if an index is in the unit equality set
    pub fn contains(&self, idx: &usize) -> bool {
        self.unit_equalities.contains(idx)
    }

    /// Get the number of unit equalities
    pub fn len(&self) -> usize {
        self.unit_equalities.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.unit_equalities.is_empty()
    }

    /// Check if a clause is a unit positive equality
    fn is_unit_equality(&self, clause: &Clause) -> bool {
        if clause.literals.len() != 1 || !clause.literals[0].polarity {
            return false;
        }
        if let Some(eq_pred_id) = self.equality_pred_id {
            clause.literals[0].predicate.id == eq_pred_id
                && clause.literals[0].predicate.arity == 2
        } else {
            false
        }
    }
}

impl Index for UnitEqualitiesIndex {
    fn kind(&self) -> IndexKind {
        IndexKind::UnitEqualities
    }

    fn on_clause_pending(&mut self, _idx: usize, _clause: &Clause) {
        // Unit equalities are only tracked when activated
    }

    fn on_clause_activated(&mut self, idx: usize, clause: &Clause) {
        if self.is_unit_equality(clause) {
            self.unit_equalities.insert(idx);
        }
    }

    fn on_clause_removed(&mut self, idx: usize, _clause: &Clause) {
        self.unit_equalities.shift_remove(&idx);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// FeatureVectorIndex
// =============================================================================

/// Index using feature vectors for efficient subsumption candidate filtering.
///
/// This wraps the existing FeatureIndex from subsumption.rs.
#[derive(Debug)]
pub struct FeatureVectorIndex {
    /// The underlying feature index
    inner: FeatureIndex,
}

impl FeatureVectorIndex {
    pub fn new() -> Self {
        FeatureVectorIndex {
            inner: FeatureIndex::new(),
        }
    }

    /// Get the underlying feature index for queries
    pub fn inner(&self) -> &FeatureIndex {
        &self.inner
    }

    /// Find potential subsumers for a clause (forward query).
    /// Returns indices of active clauses C where feature(C) ≤ feature(target).
    pub fn find_potential_subsumers(&self, target: &Clause) -> Vec<usize> {
        self.inner.find_potential_subsumers(target)
    }

    /// Find potentially subsumed clauses (backward query).
    /// Returns indices of active clauses D where feature(source) ≤ feature(D).
    pub fn find_potentially_subsumed(&self, source_idx: usize) -> Vec<usize> {
        self.inner.find_potentially_subsumed(source_idx)
    }

    /// Check if source could potentially subsume target based on feature vectors.
    /// Returns true if feature(source) ≤ feature(target) componentwise.
    /// This is a direct O(d) comparison without trie traversal.
    #[inline]
    pub fn could_subsume(&self, source_idx: usize, target_idx: usize) -> bool {
        if let (Some(source_features), Some(target_features)) = (
            self.inner.get_features(source_idx),
            self.inner.get_features(target_idx),
        ) {
            target_features.compatible_as_subsumed(source_features)
        } else {
            false
        }
    }
}

impl Default for FeatureVectorIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl Index for FeatureVectorIndex {
    fn kind(&self) -> IndexKind {
        IndexKind::FeatureVectors
    }

    fn initialize(&mut self, clauses: &[Clause]) {
        self.inner.initialize_symbols(clauses);
    }

    fn on_clause_pending(&mut self, idx: usize, clause: &Clause) {
        // Add clause to feature index (but not activated yet)
        let feature_idx = self.inner.add_clause(clause);
        // Verify indices stay in sync
        debug_assert_eq!(idx, feature_idx, "FeatureIndex index mismatch: expected {}, got {}", idx, feature_idx);
    }

    fn on_clause_activated(&mut self, idx: usize, _clause: &Clause) {
        self.inner.activate(idx);
    }

    fn on_clause_removed(&mut self, idx: usize, _clause: &Clause) {
        self.inner.deactivate(idx);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// ClauseKeysIndex
// =============================================================================

/// Index using structural hashing for duplicate detection.
///
/// Uses ClauseKey for efficient exact duplicate detection.
#[derive(Debug)]
pub struct ClauseKeysIndex {
    /// Set of all clause keys (for duplicate checking)
    keys: HashSet<ClauseKey>,
    /// Map from clause key to clause index (for active clauses)
    key_to_idx: HashMap<ClauseKey, usize>,
}

impl ClauseKeysIndex {
    pub fn new() -> Self {
        ClauseKeysIndex {
            keys: HashSet::new(),
            key_to_idx: HashMap::new(),
        }
    }

    /// Check if a clause with this key exists
    pub fn contains(&self, key: &ClauseKey) -> bool {
        self.keys.contains(key)
    }

    /// Find the index of a clause with this exact key, if any
    pub fn find_exact(&self, clause: &Clause) -> Option<usize> {
        let key = ClauseKey::from_clause(clause);
        self.key_to_idx.get(&key).copied()
    }

    /// Get the index for a given key
    pub fn get(&self, key: &ClauseKey) -> Option<usize> {
        self.key_to_idx.get(key).copied()
    }
}

impl Default for ClauseKeysIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl Index for ClauseKeysIndex {
    fn kind(&self) -> IndexKind {
        IndexKind::ClauseKeys
    }

    fn on_clause_pending(&mut self, _idx: usize, _clause: &Clause) {
        // Keys are only tracked when activated
    }

    fn on_clause_activated(&mut self, idx: usize, clause: &Clause) {
        let key = ClauseKey::from_clause(clause);
        self.keys.insert(key.clone());
        self.key_to_idx.insert(key, idx);
    }

    fn on_clause_removed(&mut self, _idx: usize, clause: &Clause) {
        let key = ClauseKey::from_clause(clause);
        self.keys.remove(&key);
        self.key_to_idx.remove(&key);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// IndexRegistry
// =============================================================================

/// Central registry that owns all indices and routes lifecycle events.
///
/// The registry is created with only the indices required by the active rules.
/// It provides type-safe accessors for each index type.
pub struct IndexRegistry {
    /// Map from index kind to the boxed index
    indices: HashMap<IndexKind, Box<dyn Index>>,
}

impl IndexRegistry {
    /// Create a new registry with only the required indices.
    ///
    /// # Arguments
    /// * `required` - Set of index kinds that rules have declared they need
    /// * `interner` - Symbol interner for creating indices that need it
    pub fn new(required: &HashSet<IndexKind>, interner: &Interner) -> Self {
        let mut indices: HashMap<IndexKind, Box<dyn Index>> = HashMap::new();

        for &kind in required {
            let index: Box<dyn Index> = match kind {
                IndexKind::UnitClauses => Box::new(UnitClausesIndex::new()),
                IndexKind::UnitEqualities => Box::new(UnitEqualitiesIndex::new(interner)),
                IndexKind::FeatureVectors => Box::new(FeatureVectorIndex::new()),
                IndexKind::ClauseKeys => Box::new(ClauseKeysIndex::new()),
            };
            indices.insert(kind, index);
        }

        IndexRegistry { indices }
    }

    /// Initialize all indices with the input clauses.
    pub fn initialize(&mut self, clauses: &[Clause]) {
        for index in self.indices.values_mut() {
            index.initialize(clauses);
        }
    }

    /// Route a clause pending event to all indices.
    pub fn on_clause_pending(&mut self, idx: usize, clause: &Clause) {
        for index in self.indices.values_mut() {
            index.on_clause_pending(idx, clause);
        }
    }

    /// Route a clause activated event to all indices.
    pub fn on_clause_activated(&mut self, idx: usize, clause: &Clause) {
        for index in self.indices.values_mut() {
            index.on_clause_activated(idx, clause);
        }
    }

    /// Route a clause removed event to all indices.
    pub fn on_clause_removed(&mut self, idx: usize, clause: &Clause) {
        for index in self.indices.values_mut() {
            index.on_clause_removed(idx, clause);
        }
    }

    // Type-safe accessors

    /// Get the UnitClausesIndex if it was created
    pub fn unit_clauses(&self) -> Option<&UnitClausesIndex> {
        self.indices
            .get(&IndexKind::UnitClauses)
            .and_then(|idx| idx.as_any().downcast_ref())
    }

    /// Get the UnitEqualitiesIndex if it was created
    pub fn unit_equalities(&self) -> Option<&UnitEqualitiesIndex> {
        self.indices
            .get(&IndexKind::UnitEqualities)
            .and_then(|idx| idx.as_any().downcast_ref())
    }

    /// Get the FeatureVectorIndex if it was created
    pub fn feature_vectors(&self) -> Option<&FeatureVectorIndex> {
        self.indices
            .get(&IndexKind::FeatureVectors)
            .and_then(|idx| idx.as_any().downcast_ref())
    }

    /// Get the ClauseKeysIndex if it was created
    pub fn clause_keys(&self) -> Option<&ClauseKeysIndex> {
        self.indices
            .get(&IndexKind::ClauseKeys)
            .and_then(|idx| idx.as_any().downcast_ref())
    }

    /// Check if an index kind is present
    pub fn has(&self, kind: IndexKind) -> bool {
        self.indices.contains_key(&kind)
    }
}

impl std::fmt::Debug for IndexRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexRegistry")
            .field("indices", &self.indices.keys().collect::<Vec<_>>())
            .finish()
    }
}

// =============================================================================
// IndexProvider
// =============================================================================

/// Read-only view of indices, restricted to what a rule declared it needs.
///
/// This is passed to rules at call time and only allows access to indices
/// that the rule declared via `required_indices()`.
pub struct IndexProvider<'a> {
    registry: &'a IndexRegistry,
    allowed: &'a HashSet<IndexKind>,
}

impl<'a> IndexProvider<'a> {
    /// Create a new provider with the given allowed indices.
    pub fn new(registry: &'a IndexRegistry, allowed: &'a HashSet<IndexKind>) -> Self {
        IndexProvider { registry, allowed }
    }

    /// Get the UnitClausesIndex if the rule declared it as a requirement
    pub fn unit_clauses(&self) -> Option<&UnitClausesIndex> {
        if self.allowed.contains(&IndexKind::UnitClauses) {
            self.registry.unit_clauses()
        } else {
            None
        }
    }

    /// Get the UnitEqualitiesIndex if the rule declared it as a requirement
    pub fn unit_equalities(&self) -> Option<&UnitEqualitiesIndex> {
        if self.allowed.contains(&IndexKind::UnitEqualities) {
            self.registry.unit_equalities()
        } else {
            None
        }
    }

    /// Get the FeatureVectorIndex if the rule declared it as a requirement
    pub fn feature_vectors(&self) -> Option<&FeatureVectorIndex> {
        if self.allowed.contains(&IndexKind::FeatureVectors) {
            self.registry.feature_vectors()
        } else {
            None
        }
    }

    /// Get the ClauseKeysIndex if the rule declared it as a requirement
    pub fn clause_keys(&self) -> Option<&ClauseKeysIndex> {
        if self.allowed.contains(&IndexKind::ClauseKeys) {
            self.registry.clause_keys()
        } else {
            None
        }
    }
}

impl std::fmt::Debug for IndexProvider<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexProvider")
            .field("allowed", self.allowed)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fol::{Literal, PredicateSymbol, Term, Variable};

    fn create_test_interner() -> Interner {
        Interner::new()
    }

    fn create_unit_clause(interner: &mut Interner) -> Clause {
        let pred_id = interner.intern_predicate("P");
        let var_id = interner.intern_variable("X");
        Clause::new(vec![Literal::positive(
            PredicateSymbol::new(pred_id, 1),
            vec![Term::Variable(Variable::new(var_id))],
        )])
    }

    fn create_unit_equality(interner: &mut Interner) -> Clause {
        let eq_id = interner.intern_predicate("=");
        let a_id = interner.intern_constant("a");
        let b_id = interner.intern_constant("b");
        Clause::new(vec![Literal::positive(
            PredicateSymbol::new(eq_id, 2),
            vec![
                Term::Constant(crate::fol::Constant::new(a_id)),
                Term::Constant(crate::fol::Constant::new(b_id)),
            ],
        )])
    }

    #[test]
    fn test_unit_clauses_index() {
        let mut index = UnitClausesIndex::new();
        let mut interner = create_test_interner();
        let clause = create_unit_clause(&mut interner);

        assert!(index.is_empty());

        index.on_clause_activated(0, &clause);
        assert_eq!(index.len(), 1);
        assert!(index.contains(&0));

        index.on_clause_removed(0, &clause);
        assert!(index.is_empty());
    }

    #[test]
    fn test_unit_equalities_index() {
        let mut interner = create_test_interner();
        let mut index = UnitEqualitiesIndex::new(&interner);
        let eq_clause = create_unit_equality(&mut interner);
        let non_eq = create_unit_clause(&mut interner);

        // Recreate index with updated interner
        index = UnitEqualitiesIndex::new(&interner);

        index.on_clause_activated(0, &eq_clause);
        index.on_clause_activated(1, &non_eq);

        assert_eq!(index.len(), 1);
        assert!(index.contains(&0));
        assert!(!index.contains(&1));
    }

    #[test]
    fn test_index_registry_creation() {
        let interner = create_test_interner();
        let mut required = HashSet::new();
        required.insert(IndexKind::UnitClauses);
        required.insert(IndexKind::UnitEqualities);

        let registry = IndexRegistry::new(&required, &interner);

        assert!(registry.has(IndexKind::UnitClauses));
        assert!(registry.has(IndexKind::UnitEqualities));
        assert!(!registry.has(IndexKind::FeatureVectors));
        assert!(!registry.has(IndexKind::ClauseKeys));
    }

    #[test]
    fn test_index_provider_restricts_access() {
        let interner = create_test_interner();
        let mut required = HashSet::new();
        required.insert(IndexKind::UnitClauses);
        required.insert(IndexKind::UnitEqualities);
        required.insert(IndexKind::FeatureVectors);

        let registry = IndexRegistry::new(&required, &interner);

        // Provider that only allows UnitClauses
        let mut allowed = HashSet::new();
        allowed.insert(IndexKind::UnitClauses);
        let provider = IndexProvider::new(&registry, &allowed);

        assert!(provider.unit_clauses().is_some());
        assert!(provider.unit_equalities().is_none()); // Not allowed
        assert!(provider.feature_vectors().is_none()); // Not allowed
    }
}
