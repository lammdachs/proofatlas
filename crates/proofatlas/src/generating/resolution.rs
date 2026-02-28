//! Binary resolution inference rule

use super::common::{
    collect_scoped_literals_except, remove_duplicate_literals, unify_atoms_scoped,
};
use crate::state::{SaturationState, StateChange, GeneratingInference, VerificationError};
use crate::logic::{Clause, Interner, Position, PredicateId};
use crate::logic::clause_manager::ClauseManager;
use crate::index::SelectedLiteralIndex;
use crate::selection::LiteralSelector;
use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::Ordering as AtomicOrdering;

/// Apply binary resolution between two clauses using literal selection
pub fn resolution(
    clause1: &Clause,
    clause2: &Clause,
    idx1: usize,
    idx2: usize,
    selector: &dyn LiteralSelector,
    interner: &mut Interner,
) -> Vec<StateChange> {
    let mut results = Vec::new();

    // Get selected literals from both clauses
    let selected1 = selector.select(clause1);
    let selected2 = selector.select(clause2);

    // If no literals are selected in either clause, no resolution is possible
    if selected1.is_empty() || selected2.is_empty() {
        return results;
    }

    // Only try to resolve SELECTED literals (no variable renaming needed — scoped unification)
    for &i in &selected1 {
        let lit1 = &clause1.literals[i];

        for &j in &selected2 {
            let lit2 = &clause2.literals[j];

            // Check if literals have opposite polarity and same predicate
            if lit1.polarity != lit2.polarity && lit1.predicate == lit2.predicate {
                // Try to unify the atoms with scoped variables
                if let Ok(mgu) = unify_atoms_scoped(lit1.predicate, &lit1.args, 0, lit2.predicate, &lit2.args, 1) {
                    // Collect side literals from both clauses
                    let mut renaming = std::collections::HashMap::new();
                    let mut new_literals = collect_scoped_literals_except(clause1, &[i], 0, &mgu, &mut renaming, interner);
                    new_literals.extend(collect_scoped_literals_except(clause2, &[j], 1, &mgu, &mut renaming, interner));

                    // Remove duplicates
                    new_literals = remove_duplicate_literals(new_literals);

                    let new_clause = Clause::new(new_literals);

                    // Orientation and normalization handled by apply_change
                    results.push(StateChange::Add(
                        Arc::new(new_clause),
                        "Resolution".into(),
                        vec![Position::clause(idx1), Position::clause(idx2)],
                    ));
                }
            }
        }
    }

    results
}

/// Resolution inference rule.
///
/// Owns a SelectedLiteralIndex for candidate filtering.
pub struct ResolutionRule {
    selected_literals: SelectedLiteralIndex,
}

impl ResolutionRule {
    pub fn new(literal_selector: Arc<dyn LiteralSelector>, eq_pred_id: Option<PredicateId>) -> Self {
        ResolutionRule {
            selected_literals: SelectedLiteralIndex::new(literal_selector, eq_pred_id),
        }
    }
}

impl GeneratingInference for ResolutionRule {
    fn name(&self) -> &str {
        "Resolution"
    }

    fn verify(
        &self,
        conclusion: &Clause,
        premises: &[Position],
        state: &SaturationState,
        cm: &ClauseManager,
    ) -> Result<(), VerificationError> {
        use crate::logic::ordering::orient_equalities::orient_clause_equalities;

        if premises.len() != 2 {
            return Err(VerificationError::InvalidConclusion {
                step_idx: 0,
                rule: "Resolution".into(),
                reason: format!("expected 2 premises, got {}", premises.len()),
            });
        }

        let c1 = &state.clauses[premises[0].clause];
        let c2 = &state.clauses[premises[1].clause];
        let interner = &cm.interner;

        for (i, lit1) in c1.literals.iter().enumerate() {
            for (j, lit2) in c2.literals.iter().enumerate() {
                if lit1.polarity != lit2.polarity && lit1.predicate == lit2.predicate {
                    if let Ok(mgu) = super::common::unify_atoms_scoped(
                        lit1.predicate, &lit1.args, 0,
                        lit2.predicate, &lit2.args, 1,
                    ) {
                        let mut int = interner.clone();
                        let mut renaming = std::collections::HashMap::new();
                        let mut new_lits = super::common::collect_scoped_literals_except(
                            c1, &[i], 0, &mgu, &mut renaming, &mut int,
                        );
                        new_lits.extend(super::common::collect_scoped_literals_except(
                            c2, &[j], 1, &mgu, &mut renaming, &mut int,
                        ));
                        new_lits = super::common::remove_duplicate_literals(new_lits);

                        let mut reconstructed = Clause::new(new_lits);
                        reconstructed.normalize_variables(&mut int);
                        orient_clause_equalities(&mut reconstructed, &int);

                        if conclusion.literals.len() == reconstructed.literals.len()
                            && conclusion.literals.iter().all(|cl| reconstructed.literals.contains(cl))
                        {
                            return Ok(());
                        }
                    }
                }
            }
        }

        Err(VerificationError::InvalidConclusion {
            step_idx: 0,
            rule: "Resolution".into(),
            reason: "no complementary literal pair produces this resolvent".into(),
        })
    }

    fn generate(
        &mut self,
        given_idx: usize,
        state: &SaturationState,
        cm: &mut ClauseManager,
    ) -> Vec<StateChange> {
        let given = &state.clauses[given_idx];
        let cancel = cm.cancel.clone();
        let start_time = cm.start_time;
        let timeout = cm.timeout;
        let memory_limit = cm.memory_limit;
        let baseline_rss = cm.baseline_rss_mb;
        let selector = cm.literal_selector.as_ref();
        let interner = &mut cm.interner;
        let mut changes = Vec::new();

        let stopped = || -> bool {
            if cancel.load(AtomicOrdering::Relaxed) { return true; }
            if let Some(start) = start_time {
                if start.elapsed() > timeout {
                    cancel.store(true, AtomicOrdering::Relaxed);
                    return true;
                }
            }
            if let Some(limit) = memory_limit {
                if let Some(rss) = crate::config::process_memory_mb() {
                    if rss.saturating_sub(baseline_rss) >= limit {
                        cancel.store(true, AtomicOrdering::Relaxed);
                        return true;
                    }
                }
            }
            false
        };

        let sli = &self.selected_literals;

        // Collect unique candidate clause indices from index
        let given_selected = selector.select(given);
        let mut candidate_set: BTreeSet<usize> = BTreeSet::new();
        for &lit_idx in &given_selected {
            let lit = &given.literals[lit_idx];
            // Look for clauses with complementary polarity
            for &(clause_idx, _) in sli.candidates_by_predicate(lit.predicate.id, !lit.polarity) {
                candidate_set.insert(clause_idx);
            }
        }

        for &partner_idx in &candidate_set {
            if stopped() { break; }
            if let Some(partner) = state.clauses.get(partner_idx) {
                changes.extend(resolution(given, partner, given_idx, partner_idx, selector, interner));
                if partner_idx != given_idx {
                    changes.extend(resolution(partner, given, partner_idx, given_idx, selector, interner));
                }
            }
        }

        // Self-resolution if given wasn't already a candidate
        if !candidate_set.contains(&given_idx) && !stopped() {
            changes.extend(resolution(given, given, given_idx, given_idx, selector, interner));
        }

        changes
    }

    fn on_activate(&mut self, idx: usize, clause: &Arc<Clause>) {
        self.selected_literals.on_activate(idx, clause);
    }

    fn on_delete(&mut self, idx: usize, clause: &Arc<Clause>) {
        self.selected_literals.on_delete(idx, clause);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::{Constant, Literal, PredicateSymbol, Term, Variable};
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
    fn test_resolution_with_select_all() {
        let mut ctx = TestContext::new();

        // P(a) v Q(X)
        // ~P(a) v R(b)
        // Should resolve to Q(X) v R(b)

        let p = ctx.pred("P", 1);
        let q = ctx.pred("Q", 1);
        let r = ctx.pred("R", 1);

        let a = ctx.const_("a");
        let b = ctx.const_("b");
        let x = ctx.var("X");

        let clause1 = Clause::new(vec![
            Literal::positive(p, vec![a.clone()]),
            Literal::positive(q, vec![x.clone()]),
        ]);

        let clause2 = Clause::new(vec![
            Literal::negative(p, vec![a.clone()]),
            Literal::positive(r, vec![b.clone()]),
        ]);

        let selector = SelectAll;
        let results = resolution(&clause1, &clause2, 0, 1, &selector, &mut ctx.interner);
        assert_eq!(results.len(), 1);
        if let StateChange::Add(clause, rule, _) = &results[0] {
            assert_eq!(clause.literals.len(), 2);
            assert_eq!(rule, "Resolution");
        } else {
            panic!("Expected StateChange::Add");
        }
    }
}
