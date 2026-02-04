//! Equality resolution inference rule

use super::common::InferenceResult;
use crate::fol::Clause;
use super::derivation::Derivation;
use crate::selection::LiteralSelector;
use crate::unification::unify;

/// Apply equality resolution rule using literal selection
/// From ~s = t, if we can unify s and t, derive the remaining clause
pub fn equality_resolution(
    clause: &Clause,
    idx: usize,
    selector: &dyn LiteralSelector,
) -> Vec<InferenceResult> {
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
        if !lit.polarity && lit.atom.is_equality() {
            if let [ref s, ref t] = lit.atom.args.as_slice() {
                // Try to unify s and t
                if let Ok(mgu) = unify(s, t) {
                    // Apply substitution to all other literals
                    let mut new_literals = Vec::new();
                    for (j, other_lit) in clause.literals.iter().enumerate() {
                        if i != j {
                            new_literals.push(other_lit.apply_substitution(&mgu));
                        }
                    }

                    // The negative equality disappears, leaving the remaining literals
                    let new_clause = Clause::new(new_literals);

                    results.push(InferenceResult {
                        derivation: Derivation::equality_resolution(idx),
                        conclusion: new_clause,
                    });
                }
            }
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fol::{Atom, Constant, Literal, PredicateSymbol, Term};
    use crate::selection::SelectAll;

    #[test]
    fn test_equality_resolution_with_select_all() {
        // Test ~a = a should resolve to empty clause
        let eq_pred = PredicateSymbol {
            name: "=".to_string(),
            arity: 2,
        };
        let a = Term::Constant(Constant {
            name: "a".to_string(),
        });

        let clause = Clause::new(vec![Literal::negative(Atom {
            predicate: eq_pred.clone(),
            args: vec![a.clone(), a.clone()],
        })]);

        let selector = SelectAll;
        let results = equality_resolution(&clause, 0, &selector);
        assert_eq!(results.len(), 1);
        assert!(results[0].conclusion.is_empty());
        assert_eq!(results[0].derivation, Derivation::equality_resolution(0));
    }
}
