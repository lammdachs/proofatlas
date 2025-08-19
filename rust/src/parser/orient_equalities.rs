//! Equality orientation for preprocessing
//!
//! Orients equality literals so that the larger term (according to KBO)
//! is on the left side. This improves superposition performance.

use crate::core::{Clause, KBOConfig, TermOrdering, KBO};

/// Orient all equality literals in a clause
pub fn orient_clause_equalities(clause: &mut Clause) {
    let kbo = KBO::new(KBOConfig::default());

    for literal in &mut clause.literals {
        if literal.atom.is_equality() && literal.atom.args.len() == 2 {
            let left = &literal.atom.args[0];
            let right = &literal.atom.args[1];

            // Compare terms using KBO
            match kbo.compare(left, right) {
                TermOrdering::Less => {
                    // Right is larger, swap the arguments
                    literal.atom.args.swap(0, 1);
                }
                _ => {
                    // Keep as is (Greater, Equal, or Incomparable)
                    // For Incomparable, we keep the original order
                }
            }
        }
    }
}

/// Orient equalities in all clauses
pub fn orient_all_equalities(clauses: &mut [Clause]) {
    for clause in clauses {
        orient_clause_equalities(clause);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Atom, Constant, Literal, PredicateSymbol, Term};

    #[test]
    fn test_orient_simple_equality() {
        let a = Term::Constant(Constant {
            name: "a".to_string(),
        });
        let b = Term::Constant(Constant {
            name: "b".to_string(),
        });

        let eq_pred = PredicateSymbol {
            name: "=".to_string(),
            arity: 2,
        };

        // Create clause: a = b
        let mut clause = Clause {
            literals: vec![Literal {
                atom: Atom {
                    predicate: eq_pred.clone(),
                    args: vec![a.clone(), b.clone()],
                },
                polarity: true,
            }],
            id: None,
        };

        orient_clause_equalities(&mut clause);

        // Should be reoriented to b = a (since b > a alphabetically)
        assert_eq!(clause.literals[0].atom.args[0], b);
        assert_eq!(clause.literals[0].atom.args[1], a);
    }

    #[test]
    fn test_keep_correct_orientation() {
        let c = Term::Constant(Constant {
            name: "c".to_string(),
        });
        let a = Term::Constant(Constant {
            name: "a".to_string(),
        });

        let eq_pred = PredicateSymbol {
            name: "=".to_string(),
            arity: 2,
        };

        // Create clause: c = a (already correctly oriented)
        let mut clause = Clause {
            literals: vec![Literal {
                atom: Atom {
                    predicate: eq_pred.clone(),
                    args: vec![c.clone(), a.clone()],
                },
                polarity: true,
            }],
            id: None,
        };

        orient_clause_equalities(&mut clause);

        // Should remain c = a
        assert_eq!(clause.literals[0].atom.args[0], c);
        assert_eq!(clause.literals[0].atom.args[1], a);
    }
}
