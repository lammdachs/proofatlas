//! Equality factoring inference rule

use super::common::{collect_literals_except, is_ordered_greater};
use crate::logic::{Clause, Interner, Literal, Position, PredicateSymbol, Term, KBO};
use crate::state::{SaturationState, StateChange, GeneratingInference, VerificationError};
use crate::logic::clause_manager::ClauseManager;
use crate::selection::LiteralSelector;
use crate::logic::unify;
use std::sync::Arc;

/// Apply equality factoring rule
/// From l = r v s = t v C where sigma = mgu(l, s), l = r is selected
/// Constraints: l*sigma not smaller than r*sigma, l*sigma not smaller than t*sigma, r*sigma not smaller than t*sigma
/// Derive (r != t v s = t v C)*sigma
///
/// Every ordered pair of distinct positive equalities is tried, with both
/// orientations of each, since literal order in a clause is syntactic and an
/// equation with incomparable sides is stored in either orientation.
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

    let eq_symbol = PredicateSymbol::new(interner.intern_predicate("="), 2);

    for &(idx1, lit1) in &positive_eq_literals {
        // Only consider if lit1 is selected
        if !selected.contains(&idx1) {
            continue;
        }

        for &(idx2, lit2) in &positive_eq_literals {
            if idx2 == idx1 {
                continue;
            }

            for (l, r) in orientations(lit1) {
                for (s, t) in orientations(lit2) {
                    let Ok(sigma) = unify(l, s) else { continue };

                    let l_sigma = l.apply_substitution(&sigma);
                    let r_sigma = r.apply_substitution(&sigma);
                    let s_sigma = s.apply_substitution(&sigma);
                    let t_sigma = t.apply_substitution(&sigma);

                    // "not smaller" means Greater or Incomparable
                    if !(is_ordered_greater(&l_sigma, &r_sigma, kbo)
                        && is_ordered_greater(&l_sigma, &t_sigma, kbo)
                        && is_ordered_greater(&r_sigma, &t_sigma, kbo))
                    {
                        continue;
                    }

                    let mut new_literals = vec![
                        Literal::negative(eq_symbol, vec![r_sigma, t_sigma.clone()]),
                        Literal::positive(eq_symbol, vec![s_sigma, t_sigma]),
                    ];
                    new_literals.extend(collect_literals_except(clause, &[idx1, idx2], &sigma));

                    // Orientation and normalization handled by apply_change
                    results.push(StateChange::Add(
                        Arc::new(Clause::new(new_literals)),
                        "EqualityFactoring".into(),
                        vec![Position::clause(idx)],
                    ));
                }
            }
        }
    }

    results
}

/// Both orientations (lhs, rhs) of an equality literal (assumes already verified to be equality)
fn orientations(lit: &Literal) -> [(&Term, &Term); 2] {
    [(&lit.args[0], &lit.args[1]), (&lit.args[1], &lit.args[0])]
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

    fn verify(
        &self,
        conclusion: &Clause,
        premises: &[Position],
        state: &SaturationState,
        cm: &ClauseManager,
    ) -> Result<(), VerificationError> {
        use crate::state::VerificationError;

        if premises.len() != 1 {
            return Err(VerificationError::InvalidConclusion {
                step_idx: 0,
                rule: "EqualityFactoring".into(),
                reason: format!("expected 1 premise, got {}", premises.len()),
            });
        }

        let premise = &state.clauses[premises[0].clause];
        let interner = &cm.interner;

        // Find pairs of positive equality literals with unifiable sides
        let positive_eq_lits: Vec<(usize, &crate::logic::Literal)> = premise
            .literals
            .iter()
            .enumerate()
            .filter(|(_, lit)| lit.polarity && lit.is_equality(interner))
            .collect();

        if positive_eq_lits.len() < 2 {
            return Err(VerificationError::InvalidConclusion {
                step_idx: 0,
                rule: "EqualityFactoring".into(),
                reason: "premise must have at least 2 positive equalities".into(),
            });
        }

        let eq_symbol = crate::logic::PredicateSymbol::new(
            interner.get_predicate("=").unwrap_or_else(|| {
                let mut int = interner.clone();
                int.intern_predicate("=")
            }),
            2,
        );

        for &(idx1, lit1) in &positive_eq_lits {
            for &(idx2, lit2) in &positive_eq_lits {
                if idx2 == idx1 {
                    continue;
                }

                for (l, r) in orientations(lit1) {
                    for (s, t) in orientations(lit2) {
                        let Ok(sigma) = crate::logic::unify(l, s) else { continue };

                        // Build the expected conclusion: (r != t v s = t v C)*sigma
                        let t_sigma = t.apply_substitution(&sigma);
                        let mut expected_lits = vec![
                            crate::logic::Literal::negative(
                                eq_symbol,
                                vec![r.apply_substitution(&sigma), t_sigma.clone()],
                            ),
                            crate::logic::Literal::positive(
                                eq_symbol,
                                vec![s.apply_substitution(&sigma), t_sigma],
                            ),
                        ];
                        expected_lits.extend(super::common::collect_literals_except(
                            premise, &[idx1, idx2], &sigma,
                        ));

                        let mut int = interner.clone();
                        let mut reconstructed = Clause::new(expected_lits);
                        reconstructed.normalize(&mut int);
                        crate::logic::ordering::orient_equalities::orient_clause_equalities(&mut reconstructed, &int);

                        // Check if conclusion matches
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
            rule: "EqualityFactoring".into(),
            reason: "no pair of positive equalities produces this conclusion".into(),
        })
    }

    fn generate(
        &mut self,
        given_idx: usize,
        state: &SaturationState,
        cm: &mut ClauseManager,
    ) -> Vec<StateChange> {
        let given = &state.clauses[given_idx];
        let selector = cm.literal_selector.as_ref();
        let kbo = &cm.term_ordering;
        let interner = &mut cm.interner;
        equality_factoring(given, given_idx, selector, interner, kbo)
    }
}
