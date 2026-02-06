//! Tautology deletion rule.
//!
//! Deletes clauses that are tautologies (contain complementary literals or
//! reflexive equalities like t=t).

use crate::logic::{Clause, Interner, PredicateId};
use crate::logic::clause_manager::ClauseManager;
use crate::index::IndexRegistry;
use crate::state::{SaturationState, SimplifyingInference, StateChange};

/// Tautology deletion rule.
pub struct TautologyRule {
    /// Predicate ID for equality (cached for performance, None if "=" not interned)
    equality_pred_id: Option<PredicateId>,
}

impl TautologyRule {
    pub fn new(interner: &Interner) -> Self {
        TautologyRule {
            equality_pred_id: interner.get_predicate("="),
        }
    }
}

impl Default for TautologyRule {
    fn default() -> Self {
        TautologyRule {
            equality_pred_id: None,
        }
    }
}

impl SimplifyingInference for TautologyRule {
    fn name(&self) -> &str {
        "Tautology"
    }

    fn simplify_forward(
        &self,
        clause_idx: usize,
        state: &SaturationState,
        _cm: &ClauseManager,
        _indices: &IndexRegistry,
    ) -> Vec<StateChange> {
        let clause = &state.clauses[clause_idx];
        if self.is_tautology(clause) {
            vec![StateChange::Delete { clause_idx, rule_name: self.name().into() }]
        } else {
            vec![]
        }
    }
}

impl TautologyRule {
    /// Check if a clause is a tautology
    fn is_tautology(&self, clause: &Clause) -> bool {
        // Check for complementary literals
        for i in 0..clause.literals.len() {
            for j in (i + 1)..clause.literals.len() {
                let lit_i = &clause.literals[i];
                let lit_j = &clause.literals[j];
                if lit_i.polarity != lit_j.polarity && lit_i.predicate == lit_j.predicate && lit_i.args == lit_j.args {
                    return true;
                }
            }
        }

        // Check for reflexive equalities (t = t)
        if let Some(eq_pred_id) = self.equality_pred_id {
            for lit in &clause.literals {
                if lit.polarity
                    && lit.predicate.id == eq_pred_id
                    && lit.predicate.arity == 2
                {
                    if let [ref t1, ref t2] = lit.args.as_slice() {
                        if t1 == t2 {
                            return true;
                        }
                    }
                }
            }
        }

        false
    }
}
