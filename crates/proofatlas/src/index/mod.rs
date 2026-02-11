//! Registry-based index architecture for saturation-based theorem proving.
//!
//! This module provides a centralized index management system where:
//! - Rules declare which indices they need via `required_indices()`
//! - SaturationState creates only the needed indices in an `IndexRegistry`
//! - On clause lifecycle events, the registry routes updates to all indices
//!
//! ## Index Types
//!
//! - `UnitEqualities`: Unit positive equalities for demodulation
//! - `Subsumption`: Feature-vector-based subsumption checker
//! - `SelectedLiterals`: Selected literal index for generating inference candidate filtering
//!
//! ## Design
//!
//! The registry pattern decouples index management from rules:
//! - Rules don't maintain internal state; they declare requirements
//! - Indices are created once and shared across rules that need them
//! - Lifecycle events are routed atomically to all indices

pub mod feature_vector;
pub mod selected_literals;
pub mod subsumption;

use crate::logic::{Clause, Interner, PredicateId};
use indexmap::IndexSet;
use std::any::Any;
use std::collections::HashMap;

pub use feature_vector::FeatureIndex;
pub use selected_literals::SelectedLiteralIndex;
pub use subsumption::SubsumptionChecker;


// =============================================================================
// Index Kind Enumeration
// =============================================================================

/// Types of indices available for rules to request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexKind {
    /// Unit positive equalities for demodulation
    UnitEqualities,
    /// Subsumption checker for forward/backward subsumption
    Subsumption,
    /// Selected literal index for generating inference candidate filtering
    SelectedLiterals,
}

// =============================================================================
// Index Trait
// =============================================================================

/// Trait for index implementations that track clause lifecycle events.
///
/// Method names mirror the `StateChange` enum variants:
/// - `on_add`: Clause added to N (new, awaiting forward simplification)
/// - `on_transfer`: Clause transferred from N to U (survived forward simplification)
/// - `on_delete`: Clause deleted from U or P (backward simplification)
/// - `on_activate`: Clause activated from U to P (given clause selected)
pub trait Index: Send + Sync {
    /// Get the kind of this index
    fn kind(&self) -> IndexKind;

    /// Called on `Add`: clause added to N (pending, not yet active).
    fn on_add(&mut self, idx: usize, clause: &Clause);

    /// Called on `Transfer`: clause transferred from N to U.
    /// Most indices start tracking the clause here.
    fn on_transfer(&mut self, idx: usize, clause: &Clause);

    /// Called on `Delete`: clause removed from U or P.
    /// Indices should stop tracking the clause.
    fn on_delete(&mut self, idx: usize, clause: &Clause);

    /// Called on `Activate`: clause moved from U to P (processed).
    /// Used by SelectedLiteralIndex to index clauses by their selected literals.
    fn on_activate(&mut self, _idx: usize, _clause: &Clause) {}

    /// Initialize the index with all input clauses (for pre-computing symbol tables, etc.)
    fn initialize(&mut self, _clauses: &[Clause]) {}

    /// Get as Any for downcasting
    fn as_any(&self) -> &dyn Any;
}

// =============================================================================
// UnitEqualitiesIndex
// =============================================================================

/// Index tracking unit positive equalities for demodulation.
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

    fn on_add(&mut self, _idx: usize, _clause: &Clause) {
        // Unit equalities are only tracked when transferred
    }

    fn on_transfer(&mut self, idx: usize, clause: &Clause) {
        if self.is_unit_equality(clause) {
            self.unit_equalities.insert(idx);
        }
    }

    fn on_delete(&mut self, idx: usize, _clause: &Clause) {
        self.unit_equalities.shift_remove(&idx);
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
    pub fn new(required: &std::collections::HashSet<IndexKind>, interner: &Interner) -> Self {
        let mut indices: HashMap<IndexKind, Box<dyn Index>> = HashMap::new();

        for &kind in required {
            let index: Box<dyn Index> = match kind {
                IndexKind::UnitEqualities => Box::new(UnitEqualitiesIndex::new(interner)),
                IndexKind::Subsumption => Box::new(SubsumptionChecker::new()),
                IndexKind::SelectedLiterals => continue, // Created externally via add_index()
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

    /// Route an Add event to all indices.
    pub fn on_add(&mut self, idx: usize, clause: &Clause) {
        for index in self.indices.values_mut() {
            index.on_add(idx, clause);
        }
    }

    /// Route a Transfer event (N → U) to all indices.
    pub fn on_transfer(&mut self, idx: usize, clause: &Clause) {
        for index in self.indices.values_mut() {
            index.on_transfer(idx, clause);
        }
    }

    /// Route a Delete event to all indices.
    pub fn on_delete(&mut self, idx: usize, clause: &Clause) {
        for index in self.indices.values_mut() {
            index.on_delete(idx, clause);
        }
    }

    /// Route an Activate event (U → P) to all indices.
    pub fn on_activate(&mut self, idx: usize, clause: &Clause) {
        for index in self.indices.values_mut() {
            index.on_activate(idx, clause);
        }
    }

    /// Add an externally-created index to the registry.
    pub fn add_index(&mut self, index: Box<dyn Index>) {
        self.indices.insert(index.kind(), index);
    }

    // Type-safe accessors

    /// Get the UnitEqualitiesIndex if it was created
    pub fn unit_equalities(&self) -> Option<&UnitEqualitiesIndex> {
        self.indices
            .get(&IndexKind::UnitEqualities)
            .and_then(|idx| idx.as_any().downcast_ref())
    }

    /// Get the SubsumptionChecker if it was created
    pub fn subsumption_checker(&self) -> Option<&SubsumptionChecker> {
        self.indices
            .get(&IndexKind::Subsumption)
            .and_then(|idx| idx.as_any().downcast_ref())
    }

    /// Get the SelectedLiteralIndex if it was created
    pub fn selected_literals(&self) -> Option<&SelectedLiteralIndex> {
        self.indices
            .get(&IndexKind::SelectedLiterals)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::{Constant, Literal, PredicateSymbol, Term, Variable};

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
                Term::Constant(Constant::new(a_id)),
                Term::Constant(Constant::new(b_id)),
            ],
        )])
    }

    #[test]
    fn test_unit_equalities_index() {
        let mut interner = create_test_interner();
        let eq_clause = create_unit_equality(&mut interner);
        let non_eq = create_unit_clause(&mut interner);

        // Create index after interning all symbols
        let mut index = UnitEqualitiesIndex::new(&interner);

        index.on_transfer(0, &eq_clause);
        index.on_transfer(1, &non_eq);

        assert_eq!(index.len(), 1);
        assert!(index.contains(&0));
        assert!(!index.contains(&1));
    }
}
