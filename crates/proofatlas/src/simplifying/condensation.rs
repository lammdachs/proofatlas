//! Clause condensation: removes redundant literals from a clause.
//!
//! A clause C can be condensed if there exists a substitution σ such that
//! Cσ has strictly fewer distinct literals than C. This happens when σ maps
//! one literal onto another (collapsing duplicates).
//!
//! Algorithm: for each pair of literals (Li, Lj) with the same predicate and
//! polarity, try to match Li onto Lj. If σ exists, apply σ to all of C and
//! remove duplicate literals. If the result is smaller, return it.
//!
//! Example: P(X) | P(a) condenses to P(a) because σ = {X→a} makes both
//! literals identical: P(a) | P(a) = P(a).
//!
//! Condensation is a self-simplification: it only looks at the clause itself,
//! requires no index, and all lifecycle methods are no-ops.

use crate::logic::clause_manager::ClauseManager;
use crate::logic::{Clause, Literal, Position, Substitution};
use crate::simplifying::subsumption::{subsumes, subsumes_unit};
use crate::state::{SaturationState, SimplifyingInference, StateChange, VerificationError};
use std::sync::Arc;

/// Try to match literal `from` onto literal `to` (one-way: only `from`'s variables bind).
/// Returns the matching substitution if successful.
fn match_literal(from: &Literal, to: &Literal) -> Option<Substitution> {
    if from.predicate != to.predicate || from.polarity != to.polarity {
        return None;
    }
    if from.args.len() != to.args.len() {
        return None;
    }

    let mut subst = Substitution::new();
    for (pattern, target) in from.args.iter().zip(&to.args) {
        if match_term_into(pattern, target, &mut subst).is_err() {
            return None;
        }
    }
    Some(subst)
}

/// One-way match accumulating into an existing substitution.
fn match_term_into(
    pattern: &crate::logic::Term,
    target: &crate::logic::Term,
    subst: &mut Substitution,
) -> Result<(), ()> {
    use crate::logic::Term;
    match (pattern, target) {
        (Term::Variable(v), t) => {
            if let Some(bound) = subst.get(v.id) {
                if bound == t { Ok(()) } else { Err(()) }
            } else {
                subst.insert(*v, t.clone());
                Ok(())
            }
        }
        (Term::Constant(c1), Term::Constant(c2)) => {
            if c1.id == c2.id { Ok(()) } else { Err(()) }
        }
        (Term::Function(f1, args1), Term::Function(f2, args2)) => {
            if f1.id != f2.id || args1.len() != args2.len() {
                return Err(());
            }
            for (a1, a2) in args1.iter().zip(args2) {
                match_term_into(a1, a2, subst)?;
            }
            Ok(())
        }
        _ => Err(()),
    }
}

/// Clause condensation rule.
///
/// Stateless — no index needed, all lifecycle methods use default no-ops.
pub struct CondensationRule;

impl CondensationRule {
    pub fn new() -> Self {
        CondensationRule
    }
}

impl SimplifyingInference for CondensationRule {
    fn name(&self) -> &str {
        "Condensation"
    }

    fn simplify_forward(
        &mut self,
        clause_idx: usize,
        state: &SaturationState,
        _cm: &ClauseManager,
    ) -> Option<StateChange> {
        let clause = &state.clauses[clause_idx];
        let lits = &clause.literals;

        // Need at least 2 literals for condensation to be possible
        if lits.len() < 2 {
            return None;
        }

        // Try to match each pair of same-predicate, same-polarity literals
        for i in 0..lits.len() {
            for j in 0..lits.len() {
                if i == j {
                    continue;
                }

                // Try to match Li onto Lj: find σ where Liσ = Lj
                if let Some(sigma) = match_literal(&lits[i], &lits[j]) {
                    // Apply σ to all literals and collect unique results
                    let mut new_lits: Vec<Literal> = Vec::with_capacity(lits.len());
                    for lit in lits {
                        let new_lit = lit.apply_substitution(&sigma);
                        if !new_lits.contains(&new_lit) {
                            new_lits.push(new_lit);
                        }
                    }

                    if new_lits.len() < lits.len() {
                        return Some(StateChange::Simplify(
                            clause_idx,
                            Some(Arc::new(Clause::new(new_lits))),
                            "Condensation".into(),
                            vec![Position::clause(clause_idx)],
                        ));
                    }
                }
            }
        }

        None
    }

    fn verify(
        &self,
        clause_idx: usize,
        replacement: Option<&Clause>,
        premises: &[Position],
        state: &SaturationState,
        _cm: &ClauseManager,
    ) -> Result<(), VerificationError> {
        let replacement = replacement.ok_or_else(|| VerificationError::InvalidConclusion {
            step_idx: 0,
            rule: "Condensation".into(),
            reason: "condensation must produce a replacement clause".into(),
        })?;

        if premises.len() != 1 {
            return Err(VerificationError::InvalidConclusion {
                step_idx: 0,
                rule: "Condensation".into(),
                reason: format!("expected 1 premise, got {}", premises.len()),
            });
        }

        let original = &state.clauses[clause_idx];

        // Replacement must be strictly smaller
        if replacement.literals.len() >= original.literals.len() {
            return Err(VerificationError::InvalidConclusion {
                step_idx: 0,
                rule: "Condensation".into(),
                reason: "replacement must have fewer literals than original".into(),
            });
        }

        // Replacement must subsume original (this is always true for valid condensation,
        // since the replacement is Cσ and C ⊨ Cσ, thus Cσ subsumes C)
        if replacement.literals.len() == 1 {
            if !subsumes_unit(replacement, original) {
                return Err(VerificationError::InvalidConclusion {
                    step_idx: 0,
                    rule: "Condensation".into(),
                    reason: "replacement does not subsume original".into(),
                });
            }
        } else if !subsumes(replacement, original) {
            return Err(VerificationError::InvalidConclusion {
                step_idx: 0,
                rule: "Condensation".into(),
                reason: "replacement does not subsume original".into(),
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::{
        Constant, FunctionSymbol, Interner, Literal, PredicateSymbol, Term, Variable,
    };

    struct Ctx {
        interner: Interner,
    }

    impl Ctx {
        fn new() -> Self {
            Ctx {
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

        fn func(&mut self, name: &str, args: Vec<Term>) -> Term {
            let id = self.interner.intern_function(name);
            Term::Function(FunctionSymbol::new(id, args.len() as u8), args)
        }

        fn pred(&mut self, name: &str, arity: u8) -> PredicateSymbol {
            let id = self.interner.intern_predicate(name);
            PredicateSymbol::new(id, arity)
        }
    }

    fn make_state(clauses: Vec<Clause>) -> SaturationState {
        let len = clauses.len();
        SaturationState {
            clauses: clauses.into_iter().map(Arc::new).collect(),
            processed: indexmap::IndexSet::new(),
            unprocessed: indexmap::IndexSet::new(),
            new: (0..len).collect(),
            event_log: Vec::new(),
            current_iteration: 0,
            initial_clause_count: len,
        }
    }

    #[test]
    fn test_condensation_px_pa() {
        // P(X) | P(a) -> P(a) because σ = {X→a} makes P(X) = P(a)
        let mut ctx = Ctx::new();
        let p = ctx.pred("P", 1);
        let x = ctx.var("X");
        let a = ctx.const_("a");

        let clause = Clause::new(vec![
            Literal::positive(p, vec![x]),
            Literal::positive(p, vec![a.clone()]),
        ]);

        let cm = ClauseManager::new(ctx.interner, Arc::new(crate::selection::SelectAll));
        let state = make_state(vec![clause]);
        let mut rule = CondensationRule::new();

        let result = rule.simplify_forward(0, &state, &cm);
        assert!(result.is_some(), "should condense P(X) | P(a)");

        if let Some(StateChange::Simplify(idx, Some(replacement), name, _)) = result {
            assert_eq!(idx, 0);
            assert_eq!(name, "Condensation");
            assert_eq!(replacement.literals.len(), 1);
            assert_eq!(replacement.literals[0].args, vec![a]);
        } else {
            panic!("unexpected state change");
        }
    }

    #[test]
    fn test_no_condensation_distinct_predicates() {
        // P(a) | Q(b) - different predicates, no condensation possible
        let mut ctx = Ctx::new();
        let p = ctx.pred("P", 1);
        let q = ctx.pred("Q", 1);
        let a = ctx.const_("a");
        let b = ctx.const_("b");

        let clause = Clause::new(vec![
            Literal::positive(p, vec![a]),
            Literal::positive(q, vec![b]),
        ]);

        let cm = ClauseManager::new(ctx.interner, Arc::new(crate::selection::SelectAll));
        let state = make_state(vec![clause]);
        let mut rule = CondensationRule::new();

        let result = rule.simplify_forward(0, &state, &cm);
        assert!(result.is_none(), "should not condense P(a) | Q(b)");
    }

    #[test]
    fn test_no_condensation_ground_same_pred() {
        // P(a) | P(b) - ground, same predicate, but different args; no σ collapses them
        let mut ctx = Ctx::new();
        let p = ctx.pred("P", 1);
        let a = ctx.const_("a");
        let b = ctx.const_("b");

        let clause = Clause::new(vec![
            Literal::positive(p, vec![a]),
            Literal::positive(p, vec![b]),
        ]);

        let cm = ClauseManager::new(ctx.interner, Arc::new(crate::selection::SelectAll));
        let state = make_state(vec![clause]);
        let mut rule = CondensationRule::new();

        let result = rule.simplify_forward(0, &state, &cm);
        assert!(result.is_none(), "should not condense P(a) | P(b)");
    }

    #[test]
    fn test_no_condensation_unit() {
        // P(a) - single literal, nothing to condense
        let mut ctx = Ctx::new();
        let p = ctx.pred("P", 1);
        let a = ctx.const_("a");

        let clause = Clause::new(vec![Literal::positive(p, vec![a])]);

        let cm = ClauseManager::new(ctx.interner, Arc::new(crate::selection::SelectAll));
        let state = make_state(vec![clause]);
        let mut rule = CondensationRule::new();

        let result = rule.simplify_forward(0, &state, &cm);
        assert!(result.is_none());
    }

    #[test]
    fn test_condensation_three_literals() {
        // P(X) | P(a) | Q(b) -> P(a) | Q(b) because σ={X→a} collapses P(X) into P(a)
        let mut ctx = Ctx::new();
        let p = ctx.pred("P", 1);
        let q = ctx.pred("Q", 1);
        let x = ctx.var("X");
        let a = ctx.const_("a");
        let b = ctx.const_("b");

        let clause = Clause::new(vec![
            Literal::positive(p, vec![x]),
            Literal::positive(p, vec![a.clone()]),
            Literal::positive(q, vec![b.clone()]),
        ]);

        let cm = ClauseManager::new(ctx.interner, Arc::new(crate::selection::SelectAll));
        let state = make_state(vec![clause]);
        let mut rule = CondensationRule::new();

        let result = rule.simplify_forward(0, &state, &cm);
        assert!(result.is_some());

        if let Some(StateChange::Simplify(_, Some(replacement), _, _)) = result {
            assert_eq!(replacement.literals.len(), 2);
        }
    }

    #[test]
    fn test_condensation_with_function() {
        // P(f(X)) | P(f(a)) -> P(f(a))
        let mut ctx = Ctx::new();
        let p = ctx.pred("P", 1);
        let x = ctx.var("X");
        let a = ctx.const_("a");
        let fx = ctx.func("f", vec![x]);
        let fa = ctx.func("f", vec![a]);

        let clause = Clause::new(vec![
            Literal::positive(p, vec![fx]),
            Literal::positive(p, vec![fa.clone()]),
        ]);

        let cm = ClauseManager::new(ctx.interner, Arc::new(crate::selection::SelectAll));
        let state = make_state(vec![clause]);
        let mut rule = CondensationRule::new();

        let result = rule.simplify_forward(0, &state, &cm);
        assert!(result.is_some());

        if let Some(StateChange::Simplify(_, Some(replacement), _, _)) = result {
            assert_eq!(replacement.literals.len(), 1);
            assert_eq!(replacement.literals[0].args, vec![fa]);
        }
    }

    #[test]
    fn test_condensation_propagates_to_other_literals() {
        // P(X) | P(a) | Q(X) -> P(a) | Q(a) because σ={X→a} applies to ALL literals
        let mut ctx = Ctx::new();
        let p = ctx.pred("P", 1);
        let q = ctx.pred("Q", 1);
        let x = ctx.var("X");
        let a = ctx.const_("a");

        let clause = Clause::new(vec![
            Literal::positive(p, vec![x.clone()]),
            Literal::positive(p, vec![a.clone()]),
            Literal::positive(q, vec![x]),
        ]);

        let cm = ClauseManager::new(ctx.interner, Arc::new(crate::selection::SelectAll));
        let state = make_state(vec![clause]);
        let mut rule = CondensationRule::new();

        let result = rule.simplify_forward(0, &state, &cm);
        assert!(result.is_some());

        if let Some(StateChange::Simplify(_, Some(replacement), _, _)) = result {
            assert_eq!(replacement.literals.len(), 2);
            // Q(X) should become Q(a) under σ={X→a}
            let q_lit = replacement
                .literals
                .iter()
                .find(|l| l.predicate == q)
                .unwrap();
            assert_eq!(q_lit.args, vec![a]);
        }
    }

    #[test]
    fn test_no_condensation_different_polarity() {
        // P(X) | ~P(a) - same predicate but different polarity
        let mut ctx = Ctx::new();
        let p = ctx.pred("P", 1);
        let x = ctx.var("X");
        let a = ctx.const_("a");

        let clause = Clause::new(vec![
            Literal::positive(p, vec![x]),
            Literal::negative(p, vec![a]),
        ]);

        let cm = ClauseManager::new(ctx.interner, Arc::new(crate::selection::SelectAll));
        let state = make_state(vec![clause]);
        let mut rule = CondensationRule::new();

        let result = rule.simplify_forward(0, &state, &cm);
        assert!(result.is_none());
    }
}
