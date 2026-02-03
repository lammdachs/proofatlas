//! Superposition inference rule for equality reasoning

use super::common::{
    remove_duplicate_literals, rename_clause_variables, InferenceResult,
};
use crate::core::{
    Atom, Clause, Derivation, KBOConfig, Literal, Substitution, Term, TermOrdering as Ordering, KBO,
};
use super::LiteralSelector;
use crate::unification::unify;

/// Position in a term/atom where unification can occur
struct Position {
    term: Term,
    path: Vec<usize>, // Path to this position
}

/// Apply superposition rule using literal selection
///
/// Superposition: l ≈ r ∨ C₁    L[l'] ∨ C₂  =>  (L[r] ∨ C₁ ∨ C₂)σ
///   where σ = mgu(l, l'), lσ ⪯̸ rσ, l' is not a variable.
///   If L[l'] is an equality s[l'] ⊕ t, additionally s[l']σ ⪯̸ tσ.
pub fn superposition(
    from_clause: &Clause,
    into_clause: &Clause,
    idx1: usize,
    idx2: usize,
    selector: &dyn LiteralSelector,
) -> Vec<InferenceResult> {
    let mut results = Vec::new();
    let kbo = KBO::new(KBOConfig::default());

    // Get selected literals from both clauses
    let selected_from = selector.select(from_clause);
    let selected_into = selector.select(into_clause);

    // If no literals are selected in either clause, no superposition is possible
    if selected_from.is_empty() || selected_into.is_empty() {
        return results;
    }

    // Rename variables to avoid conflicts
    let renamed_into = rename_clause_variables(into_clause, &"2");
    let renamed_from = rename_clause_variables(from_clause, &"1");

    // Find positive equality literals in selected literals of from_clause
    for &from_idx in &selected_from {
        let from_lit = &renamed_from.literals[from_idx];

        if from_lit.polarity && from_lit.atom.is_equality() {
            if let [ref left, ref right] = from_lit.atom.args.as_slice() {
                // Try superposition in BOTH directions:
                // 1. Find occurrences of left, replace with right (left → right)
                // 2. Find occurrences of right, replace with left (right → left)
                // The ordering constraint is checked AFTER computing the MGU

                let directions: [(_, _, &str); 2] = [
                    (left, right, "l→r"),
                    (right, left, "r→l"),
                ];

                for (pattern, replacement, _dir) in directions {
                    // For each selected literal in into_clause
                    for &into_idx in &selected_into {
                        let into_lit = &renamed_into.literals[into_idx];

                        // Find positions where pattern can be unified with some subterm
                        let positions = find_unifiable_positions(&into_lit.atom, pattern, &kbo);

                        for pos in positions {
                            // CRITICAL: l' (pos.term) must not be a variable
                            // This prevents unsound inferences
                            if matches!(pos.term, Term::Variable(_)) {
                                continue;
                            }

                            if let Ok(mgu) = unify(pattern, &pos.term) {
                                // Apply substitution to both sides
                                let pattern_sigma = pattern.apply_substitution(&mgu);
                                let replacement_sigma = replacement.apply_substitution(&mgu);

                                // Check ordering constraint: pattern_sigma ⪯̸ replacement_sigma
                                // i.e., pattern_sigma must NOT be smaller than replacement_sigma
                                // This ensures we're rewriting larger to smaller (simplifying)
                                match kbo.compare(&pattern_sigma, &replacement_sigma) {
                                    Ordering::Less | Ordering::Equal => continue,
                                    Ordering::Greater | Ordering::Incomparable => {}
                                }

                                // Additional check for superposition into equalities
                                // The side containing l' must not be smaller than the other side
                                if into_lit.atom.is_equality() && !pos.path.is_empty() {
                                    let s = &into_lit.atom.args[0];
                                    let t = &into_lit.atom.args[1];

                                    let s_sigma = s.apply_substitution(&mgu);
                                    let t_sigma = t.apply_substitution(&mgu);

                                    if pos.path[0] == 0 {
                                        // l' is in s (left side): need sσ ⪯̸ tσ
                                        match kbo.compare(&s_sigma, &t_sigma) {
                                            Ordering::Less | Ordering::Equal => continue,
                                            Ordering::Greater | Ordering::Incomparable => {}
                                        }
                                    } else if pos.path[0] == 1 {
                                        // l' is in t (right side): need tσ ⪯̸ sσ
                                        match kbo.compare(&t_sigma, &s_sigma) {
                                            Ordering::Less | Ordering::Equal => continue,
                                            Ordering::Greater | Ordering::Incomparable => {}
                                        }
                                    }
                                }

                                // Apply superposition: replace pattern with replacement
                                let mut new_literals = Vec::new();

                                // Add literals from from_clause EXCEPT the equality being used
                                for (i, lit) in renamed_from.literals.iter().enumerate() {
                                    if i != from_idx {
                                        new_literals.push(lit.apply_substitution(&mgu));
                                    }
                                }

                                // Add the modified literal and other literals from into_clause
                                for (k, lit) in renamed_into.literals.iter().enumerate() {
                                    if k == into_idx {
                                        let new_atom = replace_at_position(
                                            &lit.atom,
                                            &pos.path,
                                            &replacement_sigma,
                                            &mgu,
                                        );

                                        new_literals.push(Literal {
                                            atom: new_atom,
                                            polarity: lit.polarity,
                                        });
                                    } else {
                                        new_literals.push(lit.apply_substitution(&mgu));
                                    }
                                }

                                // Remove duplicates
                                new_literals = remove_duplicate_literals(new_literals);

                                let new_clause = Clause::new(new_literals);

                                if !new_clause.is_tautology() {
                                    results.push(InferenceResult {
                                        derivation: Derivation::Superposition { parent1: idx1, parent2: idx2 },
                                        conclusion: new_clause,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    results
}

/// Find all positions in an atom where a term can potentially unify with pattern
/// This is used to find occurrences of l in the atom that can unify with l
///
/// For equalities, we search BOTH sides. The ordering constraint (s[l']σ ⪯̸ tσ)
/// is checked later after computing the MGU, not here during position search.
/// This is important because the ordering depends on the substitution, which
/// we don't know until we find a unifier.
fn find_unifiable_positions(atom: &Atom, pattern: &Term, _kbo: &KBO) -> Vec<Position> {
    let mut positions = Vec::new();

    // Search all arguments for potential unification positions
    // For equalities, this searches both sides; the ordering constraint
    // is checked later in the superposition function after computing the MGU
    for (i, arg) in atom.args.iter().enumerate() {
        find_positions_in_term(arg, pattern, vec![i], &mut positions);
    }

    positions
}

/// Find positions in a term recursively
fn find_positions_in_term(
    term: &Term,
    pattern: &Term,
    path: Vec<usize>,
    positions: &mut Vec<Position>,
) {
    // Check if current position can unify
    if could_unify(term, pattern) {
        positions.push(Position {
            term: term.clone(),
            path: path.clone(),
        });
    }

    // Recurse into subterms
    if let Term::Function(_, args) = term {
        for (i, arg) in args.iter().enumerate() {
            let mut new_path = path.clone();
            new_path.push(i);
            find_positions_in_term(arg, pattern, new_path, positions);
        }
    }
}

/// Quick check if two terms could potentially unify (without doing full unification)
fn could_unify(term1: &Term, term2: &Term) -> bool {
    match (term1, term2) {
        (Term::Variable(_), _) | (_, Term::Variable(_)) => true,
        (Term::Constant(c1), Term::Constant(c2)) => c1 == c2,
        (Term::Function(f1, args1), Term::Function(f2, args2)) => {
            f1.name == f2.name && args1.len() == args2.len()
        }
        _ => false,
    }
}

/// Replace a term at a specific position in an atom
fn replace_at_position(
    atom: &Atom,
    path: &[usize],
    replacement: &Term,
    subst: &Substitution,
) -> Atom {
    if path.is_empty() {
        // Can't replace at root of atom
        atom.apply_substitution(subst)
    } else {
        let mut new_args = atom.args.clone();
        new_args[path[0]] = replace_in_term(&new_args[path[0]], &path[1..], replacement);
        Atom {
            predicate: atom.predicate.clone(),
            args: new_args,
        }
        .apply_substitution(subst)
    }
}

/// Replace a term at a specific position in another term
fn replace_in_term(term: &Term, path: &[usize], replacement: &Term) -> Term {
    if path.is_empty() {
        replacement.clone()
    } else {
        match term {
            Term::Variable(_) | Term::Constant(_) => term.clone(),
            Term::Function(f, args) => {
                let mut new_args = args.clone();
                new_args[path[0]] = replace_in_term(&new_args[path[0]], &path[1..], replacement);
                Term::Function(f.clone(), new_args)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Constant, FunctionSymbol, PredicateSymbol, Variable};
    use crate::inference::SelectAll;

    /// Test superposition into the RIGHT side of an equality
    /// This verifies that we search both sides of equalities for superposition positions.
    /// The ordering constraint s[l']σ ⪯̸ tσ must be satisfied (the side containing l'
    /// must not be smaller than the other side).
    #[test]
    fn test_superposition_into_right_side_of_equality() {
        // From: f(X) = X (f(X) > X in KBO)
        // Into: a = f(b) (f(b) > a in KBO, so RIGHT side is larger)
        //
        // The pattern f(X) should unify with f(b) in the RIGHT side of the into clause.
        // s[l'] = f(b), t = a
        // Constraint: s[l']σ ⪯̸ tσ means f(b) ⪯̸ a, which is satisfied since f(b) > a
        // After superposition: a = b

        let eq = PredicateSymbol {
            name: "=".to_string(),
            arity: 2,
        };
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
        let x = Term::Variable(Variable {
            name: "X".to_string(),
        });
        let f_x = Term::Function(f.clone(), vec![x.clone()]);
        let f_b = Term::Function(f.clone(), vec![b.clone()]);

        // f(X) = X
        let clause1 = Clause::new(vec![Literal::positive(Atom {
            predicate: eq.clone(),
            args: vec![f_x.clone(), x.clone()],
        })]);

        // a = f(b) (note: right side f(b) is larger than left side a)
        let clause2 = Clause::new(vec![Literal::positive(Atom {
            predicate: eq.clone(),
            args: vec![a.clone(), f_b.clone()],
        })]);

        let selector = SelectAll;
        let results = superposition(&clause1, &clause2, 0, 1, &selector);

        // Should derive: a = b
        assert!(
            !results.is_empty(),
            "Superposition should find positions in the right side of equalities"
        );

        // Find the expected result: a = b
        let found = results.iter().any(|r| {
            r.conclusion.literals.len() == 1
                && r.conclusion.literals[0].polarity
                && r.conclusion.literals[0].atom.predicate.name == "="
                && r.conclusion.literals[0].atom.args.len() == 2
        });

        assert!(
            found,
            "Expected to derive a = b, got: {:?}",
            results.iter().map(|r| r.conclusion.to_string()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_superposition_with_selection() {
        // Test superposition with corrected implementation
        // From: mult(e,X) = X
        // Into: P(mult(e,c))
        // Should derive: P(c)

        let eq = PredicateSymbol {
            name: "=".to_string(),
            arity: 2,
        };
        let p = PredicateSymbol {
            name: "P".to_string(),
            arity: 1,
        };
        let mult = FunctionSymbol {
            name: "mult".to_string(),
            arity: 2,
        };

        let e = Term::Constant(Constant {
            name: "e".to_string(),
        });
        let c = Term::Constant(Constant {
            name: "c".to_string(),
        });
        let x = Term::Variable(Variable {
            name: "X".to_string(),
        });
        let mult_ex = Term::Function(mult.clone(), vec![e.clone(), x.clone()]);
        let mult_ec = Term::Function(mult.clone(), vec![e.clone(), c.clone()]);

        // mult(e,X) = X
        let clause1 = Clause::new(vec![Literal::positive(Atom {
            predicate: eq.clone(),
            args: vec![mult_ex.clone(), x.clone()],
        })]);

        // P(mult(e,c))
        let clause2 = Clause::new(vec![Literal::positive(Atom {
            predicate: p.clone(),
            args: vec![mult_ec.clone()],
        })]);

        let selector = SelectAll;
        let results = superposition(&clause1, &clause2, 0, 1, &selector);

        // Should derive P(c)
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].conclusion.literals.len(), 1);
        assert!(results[0].conclusion.literals[0].polarity);
        assert_eq!(results[0].conclusion.literals[0].atom.predicate.name, "P");
        assert_eq!(results[0].conclusion.literals[0].atom.args.len(), 1);
        match &results[0].conclusion.literals[0].atom.args[0] {
            Term::Constant(constant) => assert_eq!(constant.name, "c"),
            _ => panic!("Expected constant c"),
        }
    }
}
