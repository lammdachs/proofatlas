//! Selected literal index for generating inference candidate filtering.
//!
//! Indexes processed clauses by their selected literals' (PredicateId, polarity)
//! pairs so that generating rules (resolution, superposition) can query only
//! relevant candidates instead of iterating all processed clauses.

use super::{Index, IndexKind};
use crate::logic::literal_selection::LiteralSelector;
use crate::logic::{Clause, PredicateId};
use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Index mapping selected literal predicates to processed clause entries.
pub struct SelectedLiteralIndex {
    /// Literal selector (shared with ClauseManager via Arc)
    selector: Arc<dyn LiteralSelector>,
    /// Equality predicate ID (if "=" has been interned)
    equality_pred_id: Option<PredicateId>,
    /// (predicate_id, polarity) -> Vec<(clause_idx, literal_idx)>
    predicate_map: HashMap<(PredicateId, bool), Vec<(usize, usize)>>,
    /// Clause indices with at least one selected positive equality
    equality_clauses: HashSet<usize>,
    /// Cached selections per clause
    selections: HashMap<usize, Vec<usize>>,
}

impl SelectedLiteralIndex {
    pub fn new(selector: Arc<dyn LiteralSelector>, equality_pred_id: Option<PredicateId>) -> Self {
        SelectedLiteralIndex {
            selector,
            equality_pred_id,
            predicate_map: HashMap::new(),
            equality_clauses: HashSet::new(),
            selections: HashMap::new(),
        }
    }

    /// Get candidate (clause_idx, literal_idx) entries for a given predicate and polarity.
    pub fn candidates_by_predicate(&self, pred: PredicateId, polarity: bool) -> &[(usize, usize)] {
        self.predicate_map
            .get(&(pred, polarity))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get clause indices that have at least one selected positive equality.
    pub fn equality_clauses(&self) -> &HashSet<usize> {
        &self.equality_clauses
    }

    /// Get cached selected literal indices for a clause.
    pub fn selected_literals(&self, clause_idx: usize) -> Option<&[usize]> {
        self.selections.get(&clause_idx).map(|v| v.as_slice())
    }

    /// Remove all entries for a clause from all internal maps.
    fn purge_clause(&mut self, idx: usize) {
        if let Some(selected) = self.selections.remove(&idx) {
            // We need to remove entries from predicate_map, but we don't have
            // the clause anymore. Retain entries that don't match this idx.
            for key in self.predicate_map.keys().cloned().collect::<Vec<_>>() {
                let entries = self.predicate_map.get_mut(&key).unwrap();
                entries.retain(|(ci, _)| *ci != idx);
                if entries.is_empty() {
                    self.predicate_map.remove(&key);
                }
            }
            drop(selected);
        }
        self.equality_clauses.remove(&idx);
    }
}

impl std::fmt::Debug for SelectedLiteralIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectedLiteralIndex")
            .field("predicate_map_keys", &self.predicate_map.len())
            .field("equality_clauses", &self.equality_clauses.len())
            .field("cached_selections", &self.selections.len())
            .finish()
    }
}

impl Index for SelectedLiteralIndex {
    fn kind(&self) -> IndexKind {
        IndexKind::SelectedLiterals
    }

    fn on_clause_pending(&mut self, _idx: usize, _clause: &Clause) {
        // Only interested in processed clauses
    }

    fn on_clause_activated(&mut self, _idx: usize, _clause: &Clause) {
        // Only interested in processed clauses (U -> P transition)
    }

    fn on_clause_removed(&mut self, idx: usize, _clause: &Clause) {
        self.purge_clause(idx);
    }

    fn on_clause_processed(&mut self, idx: usize, clause: &Clause) {
        let selected: Vec<usize> = self.selector.select(clause).into_iter().collect();

        for &lit_idx in &selected {
            let lit = &clause.literals[lit_idx];
            let key = (lit.predicate.id, lit.polarity);
            self.predicate_map
                .entry(key)
                .or_default()
                .push((idx, lit_idx));

            // Check for selected positive equality
            if lit.polarity {
                if let Some(eq_pred) = self.equality_pred_id {
                    if lit.predicate.id == eq_pred && lit.predicate.arity == 2 {
                        self.equality_clauses.insert(idx);
                    }
                }
            }
        }

        self.selections.insert(idx, selected);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::{Constant, Interner, Literal, PredicateSymbol, Term};
    use crate::selection::SelectAll;

    #[test]
    fn test_selected_literal_index_basic() {
        let mut interner = Interner::new();
        let p_id = interner.intern_predicate("P");
        let q_id = interner.intern_predicate("Q");
        let a_id = interner.intern_constant("a");

        let selector: Arc<dyn LiteralSelector> = Arc::new(SelectAll);
        let mut index = SelectedLiteralIndex::new(selector, None);

        // Create clause: P(a) v ~Q(a)
        let clause = Clause::new(vec![
            Literal::positive(PredicateSymbol::new(p_id, 1), vec![Term::Constant(Constant::new(a_id))]),
            Literal::negative(PredicateSymbol::new(q_id, 1), vec![Term::Constant(Constant::new(a_id))]),
        ]);

        // Process clause
        index.on_clause_processed(0, &clause);

        // Should find candidates for (P, true) and (Q, false)
        assert_eq!(index.candidates_by_predicate(p_id, true).len(), 1);
        assert_eq!(index.candidates_by_predicate(q_id, false).len(), 1);
        // Should NOT find candidates for (P, false) or (Q, true)
        assert_eq!(index.candidates_by_predicate(p_id, false).len(), 0);
        assert_eq!(index.candidates_by_predicate(q_id, true).len(), 0);

        // Remove clause
        index.on_clause_removed(0, &clause);
        assert_eq!(index.candidates_by_predicate(p_id, true).len(), 0);
        assert_eq!(index.candidates_by_predicate(q_id, false).len(), 0);
    }

    #[test]
    fn test_equality_clause_tracking() {
        let mut interner = Interner::new();
        let eq_id = interner.intern_predicate("=");
        let a_id = interner.intern_constant("a");
        let b_id = interner.intern_constant("b");

        let selector: Arc<dyn LiteralSelector> = Arc::new(SelectAll);
        let mut index = SelectedLiteralIndex::new(selector, Some(eq_id));

        // Create clause: a = b
        let clause = Clause::new(vec![
            Literal::positive(
                PredicateSymbol::new(eq_id, 2),
                vec![Term::Constant(Constant::new(a_id)), Term::Constant(Constant::new(b_id))],
            ),
        ]);

        index.on_clause_processed(0, &clause);
        assert!(index.equality_clauses().contains(&0));

        index.on_clause_removed(0, &clause);
        assert!(!index.equality_clauses().contains(&0));
    }

    #[test]
    fn test_negative_equality_not_tracked() {
        let mut interner = Interner::new();
        let eq_id = interner.intern_predicate("=");
        let a_id = interner.intern_constant("a");
        let b_id = interner.intern_constant("b");

        let selector: Arc<dyn LiteralSelector> = Arc::new(SelectAll);
        let mut index = SelectedLiteralIndex::new(selector, Some(eq_id));

        // Create clause: a != b (negative equality)
        let clause = Clause::new(vec![
            Literal::negative(
                PredicateSymbol::new(eq_id, 2),
                vec![Term::Constant(Constant::new(a_id)), Term::Constant(Constant::new(b_id))],
            ),
        ]);

        index.on_clause_processed(0, &clause);
        assert!(!index.equality_clauses().contains(&0));
    }
}
