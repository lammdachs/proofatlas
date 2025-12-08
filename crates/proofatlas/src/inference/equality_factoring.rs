//! Equality factoring inference rule

use super::common::{InferenceResult, InferenceRule};
use crate::core::{
    Atom, Clause, KBOConfig, Literal, PredicateSymbol, TermOrdering as Ordering, KBO,
};
use super::LiteralSelector;
use crate::unification::unify;

/// Apply equality factoring rule
/// From s ≈ t ∨ s' ≈ t' ∨ C where σ = mgu(s, s'), s ≈ t is selected, sσ ⪯̸ tσ, s'σ ⪯̸ t'σ
/// Derive (t ≉ t' ∨ s ≈ t ∨ C)σ
pub fn equality_factoring(
    clause: &Clause,
    idx: usize,
    selector: &dyn LiteralSelector,
) -> Vec<InferenceResult> {
    let mut results = Vec::new();
    let selected = selector.select(clause);
    let kbo = KBO::new(KBOConfig::default());

    // Find all positive equality literals
    let positive_eq_literals: Vec<(usize, &Literal)> = clause
        .literals
        .iter()
        .enumerate()
        .filter(|(_, lit)| lit.polarity && is_equality(&lit.atom))
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

        let (s1, t1) = get_equality_terms(&lit1.atom).unwrap();

        for j in i + 1..positive_eq_literals.len() {
            let (idx2, lit2) = positive_eq_literals[j];
            let (s2, t2) = get_equality_terms(&lit2.atom).unwrap();

            // Try to unify s1 with s2
            if let Ok(sigma) = unify(s1, s2) {
                // Apply substitution
                let s1_sigma = s1.apply_substitution(&sigma);
                let t1_sigma = t1.apply_substitution(&sigma);
                let s2_sigma = s2.apply_substitution(&sigma);
                let t2_sigma = t2.apply_substitution(&sigma);

                // Check ordering constraints: sσ ⪯̸ tσ, s'σ ⪯̸ t'σ
                // This means sσ > tσ or sσ ‖ tσ (and same for s' and t')
                let s1_not_smaller = matches!(
                    kbo.compare(&s1_sigma, &t1_sigma),
                    Ordering::Greater | Ordering::Incomparable
                );
                let s2_not_smaller = matches!(
                    kbo.compare(&s2_sigma, &t2_sigma),
                    Ordering::Greater | Ordering::Incomparable
                );

                if s1_not_smaller && s2_not_smaller {
                    // Build the conclusion: (t ≉ t' ∨ s ≈ t ∨ C)σ
                    let mut new_literals = Vec::new();

                    // Add t ≉ t'
                    let eq_symbol = PredicateSymbol {
                        name: "=".to_string(),
                        arity: 2,
                    };
                    let neq_literal = Literal::negative(Atom {
                        predicate: eq_symbol.clone(),
                        args: vec![t1_sigma.clone(), t2_sigma.clone()],
                    });
                    new_literals.push(neq_literal);

                    // Add s ≈ t (the first equality literal)
                    let eq_literal = Literal::positive(Atom {
                        predicate: eq_symbol,
                        args: vec![s1_sigma, t1_sigma],
                    });
                    new_literals.push(eq_literal);

                    // Add all other literals from C (except the two equalities we're factoring)
                    for (k, lit) in clause.literals.iter().enumerate() {
                        if k != idx1 && k != idx2 {
                            new_literals.push(lit.apply_substitution(&sigma));
                        }
                    }

                    let conclusion = Clause::new(new_literals);

                    results.push(InferenceResult {
                        rule: InferenceRule::EqualityFactoring,
                        premises: vec![idx],
                        conclusion,
                    });
                }
            }
        }
    }

    results
}

/// Check if an atom is an equality
fn is_equality(atom: &Atom) -> bool {
    atom.predicate.name == "=" && atom.predicate.arity == 2
}

/// Get the two terms from an equality atom
fn get_equality_terms(atom: &Atom) -> Option<(&crate::core::Term, &crate::core::Term)> {
    if is_equality(atom) && atom.args.len() == 2 {
        Some((&atom.args[0], &atom.args[1]))
    } else {
        None
    }
}
