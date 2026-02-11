//! Demodulation - rewriting terms using unit equalities
//!
//! This module merges the demodulation algorithm (rewriting terms using unit
//! equalities) with the DemodulationRule adapter for the saturation framework.

use crate::logic::{Clause, Interner, KBOConfig, Literal, Position, PredicateId, Term, TermOrdering, KBO};
use crate::logic::match_term;
use crate::logic::clause_manager::ClauseManager;
use crate::logic::ordering::orient_equalities::orient_clause_equalities;
use crate::index::IndexRegistry;
use crate::state::{SaturationState, SimplifyingInference, StateChange};

// =============================================================================
// Demodulation Algorithm
// =============================================================================

/// Apply demodulation using a unit equality to rewrite terms in another clause
pub fn demodulate(
    unit_eq: &Clause,
    target: &Clause,
    unit_idx: usize,
    target_idx: usize,
    interner: &Interner,
) -> Vec<StateChange> {
    let mut results = Vec::new();

    // Unit equality must have exactly one positive equality literal
    if unit_eq.literals.len() != 1 {
        return results;
    }

    let unit_lit = &unit_eq.literals[0];
    if !unit_lit.polarity || !unit_lit.is_equality(interner) {
        return results;
    }

    // Get left and right sides of the equality
    let (lhs, rhs) = match &unit_lit.args[..] {
        [l, r] => (l, r),
        _ => return results,
    };

    // Initialize KBO for ordering checks
    let kbo = KBO::new(KBOConfig::default());

    // Only try rewriting lhs -> rhs if lhs ≻ rhs
    match kbo.compare(lhs, rhs) {
        TermOrdering::Greater => {
            // Try rewriting lhs -> rhs
            if let Some(new_clause) = demodulate_clause(target, lhs, rhs, &kbo) {
                let mut new_clause = new_clause;
                new_clause.id = None;
                results.push(StateChange::Add(
                    new_clause,
                    "Demodulation".into(),
                    vec![Position::clause(target_idx), Position::clause(unit_idx)],
                ));
            }
        }
        TermOrdering::Less => {
            // Try rewriting rhs -> lhs
            if let Some(new_clause) = demodulate_clause(target, rhs, lhs, &kbo) {
                let mut new_clause = new_clause;
                new_clause.id = None;
                results.push(StateChange::Add(
                    new_clause,
                    "Demodulation".into(),
                    vec![Position::clause(target_idx), Position::clause(unit_idx)],
                ));
            }
        }
        _ => {
            // If equal or incomparable, don't apply demodulation
        }
    }

    results
}

/// Demodulate a clause by rewriting all occurrences of lhs to rhs
fn demodulate_clause(clause: &Clause, lhs: &Term, rhs: &Term, kbo: &KBO) -> Option<Clause> {
    let mut changed = false;
    let new_literals: Vec<_> = clause
        .literals
        .iter()
        .map(|lit| {
            let new_lit = rewrite_literal(lit, lhs, rhs, kbo);
            if new_lit != *lit {
                changed = true;
            }
            new_lit
        })
        .collect();

    if changed {
        Some(Clause {
            literals: new_literals,
            id: clause.id,
            role: clause.role,
            age: clause.age,
        })
    } else {
        None
    }
}

/// Rewrite a literal by replacing occurrences of lhs with rhs
fn rewrite_literal(lit: &Literal, lhs: &Term, rhs: &Term, kbo: &KBO) -> Literal {
    Literal {
        predicate: lit.predicate,
        args: lit
            .args
            .iter()
            .map(|term| rewrite_term(term, lhs, rhs, kbo))
            .collect(),
        polarity: lit.polarity,
    }
}

/// Rewrite a term by replacing occurrences of lhs with rhs
/// Only performs the rewrite if the ordering constraint lσ ≻ rσ is satisfied
fn rewrite_term(term: &Term, lhs: &Term, rhs: &Term, kbo: &KBO) -> Term {
    // Try to match the entire term with lhs using one-way matching
    // Only variables in lhs can be substituted
    if let Ok(subst) = match_term(lhs, term) {
        // Apply substitution to both sides
        let lhs_instance = lhs.apply_substitution(&subst);
        let rhs_instance = rhs.apply_substitution(&subst);

        // Check ordering constraint: lσ ≻ rσ
        if let TermOrdering::Greater = kbo.compare(&lhs_instance, &rhs_instance) {
            return rhs_instance;
        }
    }

    // Otherwise, recursively rewrite subterms
    match term {
        Term::Variable(_) | Term::Constant(_) => term.clone(),
        Term::Function(f, args) => Term::Function(
            *f,
            args.iter()
                .map(|arg| rewrite_term(arg, lhs, rhs, kbo))
                .collect(),
        ),
    }
}

// =============================================================================
// DemodulationRule (rule adapter)
// =============================================================================

/// Demodulation rule (rewriting with unit equalities).
///
/// Stateless rule that queries the UnitEqualitiesIndex from the IndexRegistry.
pub struct DemodulationRule {
    /// Predicate ID for equality (cached for performance, None if "=" not interned)
    equality_pred_id: Option<PredicateId>,
}

impl DemodulationRule {
    pub fn new(interner: &Interner) -> Self {
        DemodulationRule {
            equality_pred_id: interner.get_predicate("="),
        }
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

impl Default for DemodulationRule {
    fn default() -> Self {
        DemodulationRule {
            equality_pred_id: None,
        }
    }
}

impl SimplifyingInference for DemodulationRule {
    fn name(&self) -> &str {
        "Demodulation"
    }

    fn simplify_forward(
        &self,
        clause_idx: usize,
        state: &SaturationState,
        cm: &ClauseManager,
        indices: &IndexRegistry,
    ) -> Option<StateChange> {
        let clause = &state.clauses[clause_idx];
        let interner = &cm.interner;

        // Query unit equalities from IndexRegistry
        let unit_eq_index = indices.unit_equalities()?;

        // Try to demodulate using each unit equality
        for &unit_idx in unit_eq_index.iter() {
            if let Some(unit_clause) = state.clauses.get(unit_idx) {
                let results = demodulate(unit_clause, clause, unit_idx, clause_idx, interner);
                if !results.is_empty() {
                    if let StateChange::Add(ref conclusion, _, _) = results[0] {
                        let mut simplified_clause = conclusion.clone();
                        orient_clause_equalities(&mut simplified_clause, interner);

                        return Some(StateChange::Simplify(
                            clause_idx,
                            Some(simplified_clause),
                            "Demodulation".into(),
                            vec![Position::clause(clause_idx), Position::clause(unit_idx)],
                        ));
                    }
                }
            }
        }
        None
    }

    fn simplify_backward(
        &self,
        clause_idx: usize,
        state: &SaturationState,
        cm: &ClauseManager,
        _indices: &IndexRegistry,
    ) -> Vec<StateChange> {
        let clause = &state.clauses[clause_idx];
        let interner = &cm.interner;

        // Only unit equalities can backward-demodulate
        if !self.is_unit_equality(clause) {
            return vec![];
        }

        let mut changes = Vec::new();

        // Try to demodulate each clause in U∪P
        for &target_idx in state.unprocessed.iter().chain(state.processed.iter()) {
            if target_idx == clause_idx {
                continue;
            }

            if let Some(target_clause) = state.clauses.get(target_idx) {
                let results = demodulate(clause, target_clause, clause_idx, target_idx, interner);
                if !results.is_empty() {
                    if let StateChange::Add(ref conclusion, _, _) = results[0] {
                        let mut simplified_clause = conclusion.clone();
                        orient_clause_equalities(&mut simplified_clause, interner);

                        changes.push(StateChange::Simplify(
                            target_idx,
                            Some(simplified_clause),
                            "Demodulation".into(),
                            vec![Position::clause(target_idx), Position::clause(clause_idx)],
                        ));
                    }
                }
            }
        }

        changes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::{Constant, FunctionSymbol, PredicateSymbol};

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

        fn func(&mut self, name: &str, args: Vec<Term>) -> Term {
            let id = self.interner.intern_function(name);
            Term::Function(FunctionSymbol::new(id, args.len() as u8), args)
        }

        fn pred(&mut self, name: &str, arity: u8) -> PredicateSymbol {
            let id = self.interner.intern_predicate(name);
            PredicateSymbol::new(id, arity)
        }
    }

    #[test]
    fn test_demodulation_basic() {
        let mut ctx = TestContext::new();

        // Unit equality: f(a) = b
        let a = ctx.const_("a");
        let b = ctx.const_("b");
        let fa = ctx.func("f", vec![a.clone()]);

        // Create equality literal manually
        let eq_pred = ctx.pred("=", 2);

        let unit_eq = Clause::new(vec![Literal::positive(eq_pred, vec![fa.clone(), b.clone()])]);

        // Target clause: P(f(a))
        let p = ctx.pred("P", 1);
        let target = Clause::new(vec![Literal::positive(p, vec![fa.clone()])]);

        let results = demodulate(&unit_eq, &target, 0, 1, &ctx.interner);
        assert_eq!(results.len(), 1);

        // Should produce P(b)
        let expected = Clause::new(vec![Literal::positive(p, vec![b.clone()])]);
        if let StateChange::Add(clause, _, _) = &results[0] {
            assert_eq!(clause.literals, expected.literals);
        } else {
            panic!("Expected StateChange::Add");
        }
    }
}
