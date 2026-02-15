//! Equality factoring inference rule

use super::common::{collect_literals_except, is_ordered_greater};
use crate::logic::{Clause, Interner, Literal, Position, PredicateSymbol, KBO};
use crate::state::{SaturationState, StateChange, GeneratingInference};
use crate::logic::clause_manager::ClauseManager;
use crate::logic::ordering::orient_equalities::orient_clause_equalities;
use crate::index::IndexRegistry;
use crate::selection::LiteralSelector;
use crate::logic::unify;
use std::sync::Arc;

/// Apply equality factoring rule
/// From l = r v s = t v C where sigma = mgu(l, s), l = r is selected
/// Constraints: l*sigma not smaller than r*sigma, l*sigma not smaller than t*sigma, r*sigma not smaller than t*sigma
/// Derive (l = r v r != t v C)*sigma
pub fn equality_factoring(
    clause: &Clause,
    idx: usize,
    selector: &dyn LiteralSelector,
    interner: &mut Interner,
    kbo: &KBO,
) -> Vec<StateChange> {
    let mut results = Vec::new();
    let selected = selector.select(clause);

    // Find all positive equality literals
    let positive_eq_literals: Vec<(usize, &Literal)> = clause
        .literals
        .iter()
        .enumerate()
        .filter(|(_, lit)| lit.polarity && lit.is_equality(interner))
        .collect();

    if positive_eq_literals.len() < 2 {
        return results;
    }

    // Try to factor each pair of positive equality literals
    for i in 0..positive_eq_literals.len() {
        let (idx1, lit1) = positive_eq_literals[i];

        // Only consider if lit1 is selected
        if !selected.contains(&idx1) {
            continue;
        }

        let (s1, t1) = get_equality_terms(lit1).unwrap();

        for j in i + 1..positive_eq_literals.len() {
            let (idx2, lit2) = positive_eq_literals[j];
            let (s2, t2) = get_equality_terms(lit2).unwrap();

            // Try to unify l with s (s1 with s2)
            if let Ok(sigma) = unify(s1, s2) {
                // Apply substitution
                // Using naming from quick reference: l=s1, r=t1, s=s2, t=t2
                let l_sigma = s1.apply_substitution(&sigma);
                let r_sigma = t1.apply_substitution(&sigma);
                let _s_sigma = s2.apply_substitution(&sigma);
                let t_sigma = t2.apply_substitution(&sigma);

                // Check ordering constraints: l*sigma not smaller than r*sigma, l*sigma not smaller than t*sigma, r*sigma not smaller than t*sigma
                // "not smaller" means Greater or Incomparable
                let l_not_smaller_r = is_ordered_greater(&l_sigma, &r_sigma, &kbo);
                let l_not_smaller_t = is_ordered_greater(&l_sigma, &t_sigma, &kbo);
                let r_not_smaller_t = is_ordered_greater(&r_sigma, &t_sigma, &kbo);

                if l_not_smaller_r && l_not_smaller_t && r_not_smaller_t {
                    // Build the conclusion: (l = r v r != t v C)*sigma
                    let mut new_literals = Vec::new();

                    // Add r != t
                    let eq_symbol = PredicateSymbol::new(interner.intern_predicate("="), 2);
                    let neq_literal = Literal::negative(eq_symbol, vec![r_sigma.clone(), t_sigma.clone()]);
                    new_literals.push(neq_literal);

                    // Add l = r (the first equality literal)
                    let eq_literal = Literal::positive(eq_symbol, vec![l_sigma, r_sigma]);
                    new_literals.push(eq_literal);

                    // Add all other literals from C (except the two equalities we're factoring)
                    new_literals.extend(collect_literals_except(clause, &[idx1, idx2], &sigma));

                    let mut conclusion = Clause::new(new_literals);
                    orient_clause_equalities(&mut conclusion, interner);

                    results.push(StateChange::Add(
                        Arc::new(conclusion),
                        "EqualityFactoring".into(),
                        vec![Position::clause(idx)],
                    ));
                }
            }
        }
    }

    results
}

/// Get the two terms from an equality literal (assumes already verified to be equality)
fn get_equality_terms(lit: &Literal) -> Option<(&crate::logic::Term, &crate::logic::Term)> {
    if lit.predicate.arity == 2 && lit.args.len() == 2 {
        Some((&lit.args[0], &lit.args[1]))
    } else {
        None
    }
}

/// Equality factoring inference rule.
///
/// Factors positive equalities.
pub struct EqualityFactoringRule;

impl EqualityFactoringRule {
    pub fn new() -> Self {
        EqualityFactoringRule
    }
}

impl Default for EqualityFactoringRule {
    fn default() -> Self {
        Self::new()
    }
}

impl GeneratingInference for EqualityFactoringRule {
    fn name(&self) -> &str {
        "EqualityFactoring"
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
        let kbo = &cm.term_ordering;
        let interner = &mut cm.interner;
        equality_factoring(given, given_idx, selector, interner, kbo)
    }
}
