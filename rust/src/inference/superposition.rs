//! Superposition inference rule for equality reasoning

use crate::core::{Clause, Literal, Atom, Term, Substitution, KBO, KBOConfig, TermOrdering as Ordering};
use crate::selection::LiteralSelector;
use crate::unification::unify;
use super::common::{InferenceResult, InferenceRule, rename_clause_variables, remove_duplicate_literals};

/// Position in a term/atom where unification can occur
struct Position {
    term: Term,
    path: Vec<usize>, // Path to this position
}

/// Apply superposition rule using literal selection
/// 
/// Implements both Superposition 1 (into predicate) and Superposition 2 (into equality):
/// - Superposition 1: l ≈ r ∨ C₁    P[l'] ∨ C₂  =>  (P[r] ∨ C₁ ∨ C₂)σ
/// - Superposition 2: l ≈ r ∨ C₁    s[l'] ⊕ t ∨ C₂  =>  (s[r] ⊕ t ∨ C₁ ∨ C₂)σ
/// 
/// where σ = mgu(l, l'), and ordering constraints must be satisfied
pub fn superposition(
    from_clause: &Clause, 
    into_clause: &Clause, 
    idx1: usize, 
    idx2: usize,
    selector: &dyn LiteralSelector
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
    let renamed_into = rename_clause_variables(into_clause, &format!("c{}", idx2));
    
    // Find positive equality literals in selected literals of from_clause
    for &from_idx in &selected_from {
        let from_lit = &from_clause.literals[from_idx];
        
        if from_lit.polarity && from_lit.atom.is_equality() {
            if let [ref l, ref r] = from_lit.atom.args.as_slice() {
                // Use the equality l ≈ r in its given orientation (already pre-oriented with l ≥ r)
                // Standard superposition: find occurrences of r and replace with l
                
                // For each selected literal in into_clause
                for &into_idx in &selected_into {
                    let into_lit = &renamed_into.literals[into_idx];
                    
                    // Find positions where r can be unified with some subterm in the target literal
                    let positions = find_unifiable_positions(&into_lit.atom, r);
                    
                    for pos in positions {
                        if let Ok(mgu) = unify(r, &pos.term) {
                            // Verify ordering constraint: lσ ≻ rσ
                            // This should always hold due to preprocessing, but we check as a safety measure
                            let l_sigma = l.apply_substitution(&mgu);
                            let r_sigma = r.apply_substitution(&mgu);
                            
                            if kbo.compare(&l_sigma, &r_sigma) != Ordering::Greater {
                                continue; // Skip if ordering constraint not satisfied
                            }
                            
                            // For Superposition 2 (into equality), check additional constraint
                            if into_lit.atom.is_equality() && into_lit.atom.args.len() == 2 {
                                // into_lit has form s[l'] ⊕ t where l' is at some position in s
                                // Note: We're checking if the whole term s[l'] > t, not just s > t
                                let s_with_l_prime_sigma = &into_lit.atom.args[0].apply_substitution(&mgu);
                                let t_sigma = &into_lit.atom.args[1].apply_substitution(&mgu);
                                
                                // Check s[l']σ ≻ tσ
                                if kbo.compare(s_with_l_prime_sigma, t_sigma) != Ordering::Greater {
                                    continue;
                                }
                                
                                // Additional check: if we're at the top level (path = [0]) and rσ = s[r']σ,
                                // then we're replacing the entire left side with lσ
                                // We need lσ ≻ s[r']σ for this to be valid
                                if pos.path == vec![0] && r_sigma == *s_with_l_prime_sigma {
                                    if kbo.compare(&l_sigma, s_with_l_prime_sigma) != Ordering::Greater {
                                        continue;
                                    }
                                }
                            }
                            
                            // Apply superposition according to the calculus:
                            // Result is (s[l] ⊕ t ∨ C₁ ∨ C₂)σ where:
                            // - s[l] ⊕ t is the modified literal from into_clause (r replaced by l)
                            // - C₁ are the other literals from from_clause (excluding l ≈ r)
                            // - C₂ are the other literals from into_clause (excluding s[l'] ⊕ t)
                            let mut new_literals = Vec::new();
                            
                            // Add C₁: literals from from_clause EXCEPT the equality l ≈ r being used
                            for (i, lit) in from_clause.literals.iter().enumerate() {
                                if i != from_idx {
                                    new_literals.push(lit.apply_substitution(&mgu));
                                }
                            }
                            
                            // Add the modified literal s[r] ⊕ t and C₂ (other literals from into_clause)
                            for (k, lit) in renamed_into.literals.iter().enumerate() {
                                if k == into_idx {
                                    // This is the literal s[l'] ⊕ t that we're modifying
                                    // Replace the occurrence of r with l at the position
                                    // IMPORTANT: Apply MGU to l before replacement to ensure all variables are substituted
                                    let l_sigma = l.apply_substitution(&mgu);
                                    let new_atom = replace_at_position(&lit.atom, &pos.path, &l_sigma, &mgu);
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
/// This is used to find occurrences of l' in the atom that can unify with l
/// For equalities, we only look in the first argument (the larger side after orientation)
fn find_unifiable_positions(atom: &Atom, pattern: &Term) -> Vec<Position> {
    let mut positions = Vec::new();
    
    if atom.is_equality() && atom.args.len() == 2 {
        // For equalities, only search in the first argument (larger side)
        // This ensures we don't violate ordering constraints
        find_positions_in_term(&atom.args[0], pattern, vec![0], &mut positions);
    } else {
        // For non-equalities, search in all arguments
        for (i, arg) in atom.args.iter().enumerate() {
            find_positions_in_term(arg, pattern, vec![i], &mut positions);
        }
    }
    
    positions
}

/// Find positions in a term recursively
fn find_positions_in_term(term: &Term, pattern: &Term, path: Vec<usize>, positions: &mut Vec<Position>) {
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
fn replace_at_position(atom: &Atom, path: &[usize], replacement: &Term, subst: &Substitution) -> Atom {
    if path.is_empty() {
        // Can't replace at root of atom
        atom.apply_substitution(subst)
    } else {
        let mut new_args = atom.args.clone();
        new_args[path[0]] = replace_in_term(&new_args[path[0]], &path[1..], replacement);
        Atom {
            predicate: atom.predicate.clone(),
            args: new_args,
        }.apply_substitution(subst)
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
    use crate::core::{Constant, Variable, PredicateSymbol, FunctionSymbol};
    use crate::selection::SelectAll;
    
    #[test]
    fn test_superposition_with_selection() {
        // Test superposition
        // mult(e,e) = e
        // mult(e,X) = X
        // Should derive mult(e,mult(e,X)) = X
        
        let eq = PredicateSymbol { name: "=".to_string(), arity: 2 };
        let mult = FunctionSymbol { name: "mult".to_string(), arity: 2 };
        
        let e = Term::Constant(Constant { name: "e".to_string() });
        let x = Term::Variable(Variable { name: "X".to_string() });
        let mult_ee = Term::Function(mult.clone(), vec![e.clone(), e.clone()]);
        let mult_ex = Term::Function(mult.clone(), vec![e.clone(), x.clone()]);
        // mult(e,e) = e
        let _clause1 = Clause::new(vec![
            Literal::positive(Atom { 
                predicate: eq.clone(), 
                args: vec![mult_ee.clone(), e.clone()]
            })
        ]);
        
        // mult(e,X) = X
        let clause2 = Clause::new(vec![
            Literal::positive(Atom { 
                predicate: eq.clone(), 
                args: vec![mult_ex.clone(), x.clone()]
            })
        ]);
        
        let selector = SelectAll;
        let results = superposition(&clause2, &clause2, 1, 1, &selector);
        
        // Should derive mult(e,mult(e,X)) = X and possibly other valid inferences
        assert!(results.len() >= 1);
        
        // Check that at least one result is mult(e,mult(e,X)) = X
        let found_expected = results.iter().any(|res| {
            if res.conclusion.literals.len() != 1 || !res.conclusion.literals[0].polarity {
                return false;
            }
            
            if let [lhs, rhs] = &res.conclusion.literals[0].atom.args[..] {
                // Check if LHS matches mult(e,mult(e,X)) pattern
                if let Term::Function(f, args) = lhs {
                    if f.name == "mult" && args.len() == 2 {
                        if let Term::Constant(c) = &args[0] {
                            if c.name == "e" {
                                if let Term::Function(f2, args2) = &args[1] {
                                    if f2.name == "mult" && args2.len() == 2 {
                                        if let (Term::Constant(c2), Term::Variable(_)) = (&args2[0], &args2[1]) {
                                            if c2.name == "e" && matches!(rhs, Term::Variable(_)) {
                                                return true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            false
        });
        
        assert!(found_expected, "Expected to find mult(e,mult(e,X)) = X in results");
    }
}