//! Superposition inference rule for equality reasoning

use super::common::{
    remove_duplicate_literals, rename_clause_variables, InferenceResult, InferenceRule,
};
use crate::core::{
    Atom, Clause, KBOConfig, Literal, Substitution, Term, TermOrdering as Ordering, KBO,
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
/// Implements both Superposition 1 (into predicate) and Superposition 2 (into equality):
/// - Superposition 1: l ≈ r ∨ C₁    P[l'] ∨ C₂  =>  (P[r] ∨ C₁ ∨ C₂)σ
///   where σ = mgu(l, l'), l ⪯̸ r, l' is not a variable
/// - Superposition 2: l ≈ r ∨ C₁    s[l'] ⊕ t ∨ C₂  =>  (s[r] ⊕ t ∨ C₁ ∨ C₂)σ
///   where σ = mgu(l, l'), l ⪯̸ r, l' is not a variable, s[l'] ⪯̸ t
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
            if let [ref l, ref r] = from_lit.atom.args.as_slice() {
                // Use the equality l ≈ r in its given orientation
                // The ordering constraint l ⪯̸ r is an invariant of the program
                // Standard superposition: find occurrences of l and replace with r

                // For each selected literal in into_clause
                for &into_idx in &selected_into {
                    let into_lit = &renamed_into.literals[into_idx];

                    // Find positions where l can be unified with some subterm in the target literal
                    // This is the correct formulation - we look for the LARGER term and replace with the SMALLER
                    let positions = find_unifiable_positions(&into_lit.atom, l, &kbo);

                    // Positions are already filtered during search, so we can use them directly

                    for pos in positions {
                        // CRITICAL: l' (pos.term) must not be a variable
                        // This prevents unsound inferences like mult(inv(X),X) = mult(e,e)
                        if matches!(pos.term, Term::Variable(_)) {
                            continue;
                        }

                        if let Ok(mgu) = unify(l, &pos.term) {
                            // Apply substitution to both sides
                            let l_sigma = l.apply_substitution(&mgu);
                            let r_sigma = r.apply_substitution(&mgu);

                            // Check ordering constraint: l_sigma ⪯̸ r_sigma
                            // This is equivalent to: ¬(l_sigma ≤ r_sigma)

                            match kbo.compare(&l_sigma, &r_sigma) {
                                Ordering::Less | Ordering::Equal => continue, // l_sigma ≤ r_sigma, skip
                                Ordering::Greater | Ordering::Incomparable => {} // l_sigma ⪯̸ r_sigma, proceed
                            }

                            // Additional check for superposition into equalities
                            // According to the calculus, for s[l'] ⊕ t, we need s[l']σ ⪯̸ tσ
                            // This means: if l' is in s (left side), check s[l']σ ⪯̸ tσ
                            //            if l' is in t (right side), check sσ ⪯̸ t[l']σ
                            if into_lit.atom.is_equality() && !pos.path.is_empty() {
                                let s = &into_lit.atom.args[0];
                                let t = &into_lit.atom.args[1];

                                if pos.path[0] == 0 {
                                    // l' is in s (left side), so we have s[l'] = t
                                    // Need to check s[l']σ ⪯̸ tσ
                                    let s_sigma = s.apply_substitution(&mgu);
                                    let t_sigma = t.apply_substitution(&mgu);

                                    match kbo.compare(&s_sigma, &t_sigma) {
                                        Ordering::Less | Ordering::Equal => continue, // s[l']σ ≤ tσ, skip
                                        Ordering::Greater | Ordering::Incomparable => {} // s[l']σ ⪯̸ tσ, proceed
                                    }
                                } else if pos.path[0] == 1 {
                                    // l' is in t (right side), so we have s = t[l']
                                    // Need to check sσ ⪯̸ t[l']σ
                                    let s_sigma = s.apply_substitution(&mgu);
                                    let t_sigma = t.apply_substitution(&mgu);

                                    match kbo.compare(&s_sigma, &t_sigma) {
                                        Ordering::Less | Ordering::Equal => continue, // sσ ≤ t[l']σ, skip
                                        Ordering::Greater | Ordering::Incomparable => {} // sσ ⪯̸ t[l']σ, proceed
                                    }
                                }
                            }

                            // Apply superposition according to the calculus:
                            // Result is (s[r] ⊕ t ∨ C₁ ∨ C₂)σ where:
                            // - s[r] ⊕ t is the modified literal from into_clause (l' replaced by r)
                            // - C₁ are the other literals from from_clause (excluding l ≈ r)
                            // - C₂ are the other literals from into_clause (excluding s[l'] ⊕ t)
                            let mut new_literals = Vec::new();

                            // Add C₁: literals from from_clause EXCEPT the equality l ≈ r being used
                            // IMPORTANT: Use renamed_from to ensure substitution applies correctly
                            for (i, lit) in renamed_from.literals.iter().enumerate() {
                                if i != from_idx {
                                    new_literals.push(lit.apply_substitution(&mgu));
                                }
                            }

                            // Add the modified literal s[r] ⊕ t and C₂ (other literals from into_clause)
                            for (k, lit) in renamed_into.literals.iter().enumerate() {
                                if k == into_idx {
                                    // This is the literal s[l'] ⊕ t that we're modifying
                                    // Replace the occurrence of l with r at the position
                                    // IMPORTANT: Apply MGU to r before replacement to ensure all variables are substituted
                                    let r_sigma = r.apply_substitution(&mgu);
                                    let new_atom =
                                        replace_at_position(&lit.atom, &pos.path, &r_sigma, &mgu);

                                    new_literals.push(Literal {
                                        atom: new_atom,
                                        polarity: lit.polarity,
                                    });
                                } else {
                                    // Add other literals from C₂
                                    new_literals.push(lit.apply_substitution(&mgu));
                                }
                            }

                            // Remove duplicates
                            new_literals = remove_duplicate_literals(new_literals);

                            let new_clause = Clause::new(new_literals);

                            if !new_clause.is_tautology() {
                                results.push(InferenceResult {
                                    rule: InferenceRule::Superposition,
                                    premises: vec![idx1, idx2],
                                    conclusion: new_clause,
                                });
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
/// For equalities, we only search in the maximal side(s) based on ordering
fn find_unifiable_positions(atom: &Atom, pattern: &Term, kbo: &KBO) -> Vec<Position> {
    let mut positions = Vec::new();

    if atom.is_equality() && atom.args.len() == 2 {
        let left = &atom.args[0];
        let right = &atom.args[1];

        // Always search the left side (due to orientation invariant, left is never smaller)
        find_positions_in_term(left, pattern, vec![0], &mut positions);

        // Only search the right side if the terms are incomparable
        // If left > right, we don't need to search the right side
        if matches!(kbo.compare(left, right), Ordering::Incomparable) {
            find_positions_in_term(right, pattern, vec![1], &mut positions);
        }
    } else {
        // For non-equalities, search in all arguments
        for (i, arg) in atom.args.iter().enumerate() {
            find_positions_in_term(arg, pattern, vec![i], &mut positions);
        }
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
