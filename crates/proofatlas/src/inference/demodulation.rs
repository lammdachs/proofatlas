//! Demodulation - rewriting terms using unit equalities

use super::common::InferenceResult;
use crate::fol::{Atom, Clause, KBOConfig, Literal, Term, TermOrdering, KBO};
use super::derivation::Derivation;
use crate::unification::match_term;

/// Apply demodulation using a unit equality to rewrite terms in another clause
pub fn demodulate(
    unit_eq: &Clause,
    target: &Clause,
    unit_idx: usize,
    target_idx: usize,
) -> Vec<InferenceResult> {
    let mut results = Vec::new();

    // Unit equality must have exactly one positive equality literal
    if unit_eq.literals.len() != 1 {
        return results;
    }

    let unit_lit = &unit_eq.literals[0];
    if !unit_lit.polarity || !unit_lit.atom.is_equality() {
        return results;
    }

    // Get left and right sides of the equality
    let (lhs, rhs) = match &unit_lit.atom.args[..] {
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
                results.push(InferenceResult {
                    derivation: Derivation {
                        rule_name: "Demodulation".into(),
                        premises: vec![unit_idx, target_idx],
                    },
                    conclusion: new_clause,
                });
            }
        }
        TermOrdering::Less => {
            // Try rewriting rhs -> lhs
            if let Some(new_clause) = demodulate_clause(target, rhs, lhs, &kbo) {
                let mut new_clause = new_clause;
                new_clause.id = None;
                results.push(InferenceResult {
                    derivation: Derivation {
                        rule_name: "Demodulation".into(),
                        premises: vec![unit_idx, target_idx],
                    },
                    conclusion: new_clause,
                });
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
        polarity: lit.polarity,
        atom: rewrite_atom(&lit.atom, lhs, rhs, kbo),
    }
}

/// Rewrite an atom by replacing occurrences of lhs with rhs
fn rewrite_atom(atom: &Atom, lhs: &Term, rhs: &Term, kbo: &KBO) -> Atom {
    Atom {
        predicate: atom.predicate.clone(),
        args: atom
            .args
            .iter()
            .map(|term| rewrite_term(term, lhs, rhs, kbo))
            .collect(),
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
            f.clone(),
            args.iter()
                .map(|arg| rewrite_term(arg, lhs, rhs, kbo))
                .collect(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fol::{Constant, FunctionSymbol, PredicateSymbol};

    #[test]
    fn test_demodulation_basic() {
        // Unit equality: f(a) = b
        let f = FunctionSymbol {
            name: "f".to_string(),
            arity: 1,
        };
        let a = Term::Constant(Constant {
            name: "a".to_string(),
        });
        let b = Term::Constant(Constant {
            name: "b".to_string(),
        });
        let fa = Term::Function(f.clone(), vec![a.clone()]);

        // Create equality atom manually
        let eq_pred = PredicateSymbol {
            name: "=".to_string(),
            arity: 2,
        };
        let eq_atom = Atom {
            predicate: eq_pred,
            args: vec![fa.clone(), b.clone()],
        };

        let unit_eq = Clause::new(vec![Literal::positive(eq_atom)]);

        // Target clause: P(f(a))
        let p = PredicateSymbol {
            name: "P".to_string(),
            arity: 1,
        };
        let target = Clause::new(vec![Literal::positive(Atom {
            predicate: p.clone(),
            args: vec![fa.clone()],
        })]);

        let results = demodulate(&unit_eq, &target, 0, 1);
        assert_eq!(results.len(), 1);

        // Should produce P(b)
        let expected = Clause::new(vec![Literal::positive(Atom {
            predicate: p.clone(),
            args: vec![b.clone()],
        })]);
        assert_eq!(results[0].conclusion.literals, expected.literals);
    }
}
