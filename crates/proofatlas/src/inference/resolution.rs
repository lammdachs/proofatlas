//! Binary resolution inference rule

use super::common::{
    remove_duplicate_literals, rename_clause_variables, unify_atoms, InferenceResult,
};
use crate::fol::Clause;
use super::derivation::Derivation;
use crate::selection::LiteralSelector;

/// Apply binary resolution between two clauses using literal selection
pub fn resolution(
    clause1: &Clause,
    clause2: &Clause,
    idx1: usize,
    idx2: usize,
    selector: &dyn LiteralSelector,
) -> Vec<InferenceResult> {
    let mut results = Vec::new();

    // Get selected literals from both clauses
    let selected1 = selector.select(clause1);
    let selected2 = selector.select(clause2);

    // If no literals are selected in either clause, no resolution is possible
    if selected1.is_empty() || selected2.is_empty() {
        return results;
    }

    // Rename variables in clause2 to avoid conflicts
    let renamed_clause2 = rename_clause_variables(clause2, &format!("c{}", idx2));

    // Only try to resolve SELECTED literals
    for &i in &selected1 {
        let lit1 = &clause1.literals[i];

        for &j in &selected2 {
            let lit2 = &renamed_clause2.literals[j];

            // Check if literals have opposite polarity and same predicate
            if lit1.polarity != lit2.polarity && lit1.atom.predicate == lit2.atom.predicate {
                // Try to unify the atoms
                if let Ok(mgu) = unify_atoms(&lit1.atom, &lit2.atom) {
                    // Build resolvent
                    let mut new_literals = Vec::new();

                    // Add literals from clause1 except the resolved one
                    for (k, lit) in clause1.literals.iter().enumerate() {
                        if k != i {
                            new_literals.push(lit.apply_substitution(&mgu));
                        }
                    }

                    // Add literals from clause2 except the resolved one
                    for (k, lit) in renamed_clause2.literals.iter().enumerate() {
                        if k != j {
                            new_literals.push(lit.apply_substitution(&mgu));
                        }
                    }

                    // Remove duplicates
                    new_literals = remove_duplicate_literals(new_literals);

                    let new_clause = Clause::new(new_literals);

                    // Don't generate tautologies
                    if !new_clause.is_tautology() {
                        results.push(InferenceResult {
                            derivation: Derivation {
                                rule_name: "Resolution".into(),
                                premises: vec![idx1, idx2],
                            },
                            conclusion: new_clause,
                        });
                    }
                }
            }
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fol::{Atom, Constant, Literal, PredicateSymbol, Term, Variable};
    use crate::selection::SelectAll;

    #[test]
    fn test_resolution_with_select_all() {
        // P(a) ∨ Q(X)
        // ~P(a) ∨ R(b)
        // Should resolve to Q(X) ∨ R(b)

        let p = PredicateSymbol {
            name: "P".to_string(),
            arity: 1,
        };
        let q = PredicateSymbol {
            name: "Q".to_string(),
            arity: 1,
        };
        let r = PredicateSymbol {
            name: "R".to_string(),
            arity: 1,
        };

        let a = Term::Constant(Constant {
            name: "a".to_string(),
        });
        let b = Term::Constant(Constant {
            name: "b".to_string(),
        });
        let x = Term::Variable(Variable {
            name: "X".to_string(),
        });

        let clause1 = Clause::new(vec![
            Literal::positive(Atom {
                predicate: p.clone(),
                args: vec![a.clone()],
            }),
            Literal::positive(Atom {
                predicate: q.clone(),
                args: vec![x.clone()],
            }),
        ]);

        let clause2 = Clause::new(vec![
            Literal::negative(Atom {
                predicate: p.clone(),
                args: vec![a.clone()],
            }),
            Literal::positive(Atom {
                predicate: r.clone(),
                args: vec![b.clone()],
            }),
        ]);

        let selector = SelectAll;
        let results = resolution(&clause1, &clause2, 0, 1, &selector);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].conclusion.literals.len(), 2);
    }
}
