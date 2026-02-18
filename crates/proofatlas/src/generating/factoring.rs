//! Factoring inference rule

use super::common::{collect_literals_except, remove_duplicate_literals, unify_atoms};
use crate::logic::{Clause, Position};
use crate::state::{SaturationState, StateChange, GeneratingInference, VerificationError};
use crate::logic::clause_manager::ClauseManager;
use crate::index::IndexRegistry;
use crate::selection::LiteralSelector;
use std::sync::Arc;

/// Apply factoring to a clause using literal selection
pub fn factoring(
    clause: &Clause,
    idx: usize,
    selector: &dyn LiteralSelector,
) -> Vec<StateChange> {
    let mut results = Vec::new();

    // Get selected literals
    let selected = selector.select(clause);

    // If no literals are selected, no factoring is possible
    if selected.is_empty() {
        return results;
    }

    // Only try to factor SELECTED literals
    for &i in &selected {
        let lit1 = &clause.literals[i];

        // Try to factor with other literals (both selected and non-selected)
        // According to the calculus, we factor the selected literal with any other literal
        for j in 0..clause.literals.len() {
            if i != j {
                let lit2 = &clause.literals[j];

                // Must have same polarity and predicate
                if lit1.polarity == lit2.polarity && lit1.predicate == lit2.predicate {
                    if let Ok(mgu) = unify_atoms(lit1.predicate, &lit1.args, lit2.predicate, &lit2.args) {
                        // Collect all literals except the factored one (j)
                        let new_literals = remove_duplicate_literals(
                            collect_literals_except(clause, &[j], &mgu)
                        );

                        let new_clause = Clause::new(new_literals);

                        // Tautology check delegated to TautologyRule during forward simplification
                        results.push(StateChange::Add(
                            Arc::new(new_clause),
                            "Factoring".into(),
                            vec![Position::clause(idx)],
                        ));
                    }
                }
            }
        }
    }

    results
}

/// Factoring inference rule.
///
/// Generates factors of the given clause.
pub struct FactoringRule;

impl FactoringRule {
    pub fn new() -> Self {
        FactoringRule
    }
}

impl Default for FactoringRule {
    fn default() -> Self {
        Self::new()
    }
}

impl GeneratingInference for FactoringRule {
    fn name(&self) -> &str {
        "Factoring"
    }

    fn verify(
        &self,
        conclusion: &Clause,
        premises: &[Position],
        state: &SaturationState,
        _cm: &ClauseManager,
    ) -> Result<(), VerificationError> {
        use crate::state::VerificationError;

        if premises.len() != 1 {
            return Err(VerificationError::InvalidConclusion {
                step_idx: 0,
                rule: "Factoring".into(),
                reason: format!("expected 1 premise, got {}", premises.len()),
            });
        }

        let premise = &state.clauses[premises[0].clause];

        // Factoring: two literals in the premise unify, producing a clause with one fewer literal.
        // The conclusion must be strictly smaller than the premise.
        if conclusion.literals.len() >= premise.literals.len() {
            return Err(VerificationError::InvalidConclusion {
                step_idx: 0,
                rule: "Factoring".into(),
                reason: "conclusion must have fewer literals than premise".into(),
            });
        }

        // Verify that some pair of literals in the premise can be unified to produce the conclusion.
        for i in 0..premise.literals.len() {
            let lit1 = &premise.literals[i];
            for j in (i + 1)..premise.literals.len() {
                let lit2 = &premise.literals[j];
                if lit1.polarity == lit2.polarity && lit1.predicate == lit2.predicate {
                    if let Ok(mgu) = super::common::unify_atoms(
                        lit1.predicate, &lit1.args, lit2.predicate, &lit2.args,
                    ) {
                        let new_lits = super::common::remove_duplicate_literals(
                            super::common::collect_literals_except(premise, &[j], &mgu),
                        );
                        if conclusion.literals.len() == new_lits.len()
                            && conclusion.literals.iter().all(|cl| new_lits.contains(cl))
                        {
                            return Ok(());
                        }
                        // Also try removing i instead of j
                        let new_lits2 = super::common::remove_duplicate_literals(
                            super::common::collect_literals_except(premise, &[i], &mgu),
                        );
                        if conclusion.literals.len() == new_lits2.len()
                            && conclusion.literals.iter().all(|cl| new_lits2.contains(cl))
                        {
                            return Ok(());
                        }
                    }
                }
            }
        }

        Err(VerificationError::InvalidConclusion {
            step_idx: 0,
            rule: "Factoring".into(),
            reason: "no literal pair produces this factor".into(),
        })
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
        let mut changes = factoring(given, given_idx, selector);
        // Orient after selector borrow ends (NLL)
        for change in &mut changes {
            if let StateChange::Add(ref mut arc, _, _) = change {
                cm.orient_equalities(Arc::get_mut(arc).expect("refcount must be 1"));
            }
        }
        changes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::{Interner, Literal, PredicateSymbol, Term, Variable};
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

        fn var(&mut self, name: &str) -> Term {
            let id = self.interner.intern_variable(name);
            Term::Variable(Variable::new(id))
        }

        fn pred(&mut self, name: &str, arity: u8) -> PredicateSymbol {
            let id = self.interner.intern_predicate(name);
            PredicateSymbol::new(id, arity)
        }
    }

    #[test]
    fn test_factoring_with_select_all() {
        let mut ctx = TestContext::new();

        // P(X) v P(Y) v Q(Z)
        let p = ctx.pred("P", 1);
        let q = ctx.pred("Q", 1);

        let x = ctx.var("X");
        let y = ctx.var("Y");
        let z = ctx.var("Z");

        let clause = Clause::new(vec![
            Literal::positive(p, vec![x.clone()]),
            Literal::positive(p, vec![y.clone()]),
            Literal::positive(q, vec![z.clone()]),
        ]);

        let selector = SelectAll;
        let results = factoring(&clause, 0, &selector);

        // With SelectAll, both P literals are selected
        // P(X) factors with P(Y), and P(Y) factors with P(X)
        // Both produce the same clause: P(X) v Q(Z) (with appropriate substitution)
        assert_eq!(results.len(), 2);
        for r in &results {
            if let StateChange::Add(clause, _, _) = r {
                assert_eq!(clause.literals.len(), 2);
            } else {
                panic!("Expected StateChange::Add");
            }
        }
    }
}
