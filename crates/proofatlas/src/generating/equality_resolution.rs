//! Equality resolution inference rule

use super::common::collect_literals_except;
use crate::logic::{Clause, Interner, Position};
use crate::state::{SaturationState, StateChange, GeneratingInference};
use crate::logic::clause_manager::ClauseManager;
use crate::logic::ordering::orient_equalities::orient_clause_equalities;
use crate::index::IndexRegistry;
use crate::selection::LiteralSelector;
use crate::logic::unify;
use std::sync::Arc;

/// Apply equality resolution rule using literal selection
/// From ~s = t, if we can unify s and t, derive the remaining clause
pub fn equality_resolution(
    clause: &Clause,
    idx: usize,
    selector: &dyn LiteralSelector,
    interner: &Interner,
) -> Vec<StateChange> {
    let mut results = Vec::new();

    // Get selected literals
    let selected = selector.select(clause);

    // If no literals are selected, no equality resolution is possible
    if selected.is_empty() {
        return results;
    }

    // Only check SELECTED negative equality literals
    for &i in &selected {
        let lit = &clause.literals[i];

        // Look for negative equality literals
        if !lit.polarity && lit.is_equality(interner) {
            if let [ref s, ref t] = lit.args.as_slice() {
                // Try to unify s and t
                if let Ok(mgu) = unify(s, t) {
                    // The negative equality disappears, leaving the remaining literals
                    let new_literals = collect_literals_except(clause, &[i], &mgu);
                    let mut new_clause = Clause::new(new_literals);
                    orient_clause_equalities(&mut new_clause, interner);

                    results.push(StateChange::Add(
                        Arc::new(new_clause),
                        "EqualityResolution".into(),
                        vec![Position::clause(idx)],
                    ));
                }
            }
        }
    }

    results
}

/// Equality resolution inference rule.
///
/// Resolves negative equalities of the form s!=s.
pub struct EqualityResolutionRule;

impl EqualityResolutionRule {
    pub fn new() -> Self {
        EqualityResolutionRule
    }
}

impl Default for EqualityResolutionRule {
    fn default() -> Self {
        Self::new()
    }
}

impl GeneratingInference for EqualityResolutionRule {
    fn name(&self) -> &str {
        "EqualityResolution"
    }

    fn generate(
        &self,
        given_idx: usize,
        state: &SaturationState,
        cm: &mut ClauseManager,
        _indices: &IndexRegistry,
    ) -> Vec<StateChange> {
        let given = &state.clauses[given_idx];
        let selector = cm.literal_selector.as_ref();
        let interner = &cm.interner;
        equality_resolution(given, given_idx, selector, interner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::{Constant, Literal, PredicateSymbol, Term};
    use crate::selection::SelectAll;

    struct TestContext {
        interner: Interner,
    }

    impl TestContext {
        fn new() -> Self {
            TestContext {
                interner: Interner::new(),
            }
        }

        fn const_(&mut self, name: &str) -> Term {
            let id = self.interner.intern_constant(name);
            Term::Constant(Constant::new(id))
        }

        fn pred(&mut self, name: &str, arity: u8) -> PredicateSymbol {
            let id = self.interner.intern_predicate(name);
            PredicateSymbol::new(id, arity)
        }
    }

    #[test]
    fn test_equality_resolution_with_select_all() {
        let mut ctx = TestContext::new();

        // Test ~a = a should resolve to empty clause
        let eq_pred = ctx.pred("=", 2);
        let a = ctx.const_("a");

        let clause = Clause::new(vec![Literal::negative(eq_pred, vec![a.clone(), a.clone()])]);

        let selector = SelectAll;
        let results = equality_resolution(&clause, 0, &selector, &ctx.interner);
        assert_eq!(results.len(), 1);
        if let StateChange::Add(clause, rule, premises) = &results[0] {
            assert!(clause.is_empty());
            assert_eq!(rule, "EqualityResolution");
            assert_eq!(premises, &vec![Position::clause(0)]);
        } else {
            panic!("Expected StateChange::Add");
        }
    }
}
