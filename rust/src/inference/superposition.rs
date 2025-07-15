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
///   where σ = mgu(l, l'), l ⪯̸ r, l' is not a variable
/// - Superposition 2: l ≈ r ∨ C₁    s[l'] ⊕ t ∨ C₂  =>  (s[r] ⊕ t ∨ C₁ ∨ C₂)σ
///   where σ = mgu(l, l'), l ⪯̸ r, l' is not a variable, s[l'] ⪯̸ t
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
    
    #[cfg(debug_assertions)]
    if idx1 == 2 && idx2 == 2 {
        eprintln!("DEBUG: Self-superposition [2,2]");
        eprintln!("  from_clause: {}", from_clause);
        eprintln!("  into_clause: {}", into_clause);
        eprintln!("  selected_from: {:?}", selected_from);
        eprintln!("  selected_into: {:?}", selected_into);
    }
    
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
                // Use the equality l ≈ r in its given orientation
                // The ordering constraint l ⪯̸ r will be checked after substitution
                // Standard superposition: find occurrences of l and replace with r
                
                // For each selected literal in into_clause
                for &into_idx in &selected_into {
                    let into_lit = &renamed_into.literals[into_idx];
                    
                    // Find positions where l can be unified with some subterm in the target literal
                    // This is the correct formulation - we look for the LARGER term and replace with the SMALLER
                    let positions = find_unifiable_positions(&into_lit.atom, l);
                    
                    
                    // Filter positions: for equalities, only keep positions in the maximal side
                    let valid_positions: Vec<_> = if into_lit.atom.is_equality() && into_lit.atom.args.len() == 2 {
                        let left = &into_lit.atom.args[0];
                        let right = &into_lit.atom.args[1];
                        let left_not_smaller = matches!(kbo.compare(left, right), Ordering::Greater | Ordering::Incomparable);
                        let right_not_smaller = matches!(kbo.compare(right, left), Ordering::Greater | Ordering::Incomparable);
                        
                        #[cfg(debug_assertions)]
                        if (idx1 == 0 && idx2 == 1) || (idx1 == 1 && idx2 == 0) || (idx1 == 0 && idx2 == 4) {
                            eprintln!("DEBUG: Filtering positions for equality {}", into_lit.atom);
                            eprintln!("  Left: {}, Right: {}", left, right);
                            eprintln!("  left_not_smaller: {}, right_not_smaller: {}", left_not_smaller, right_not_smaller);
                            eprintln!("  Positions before filter: {:?}", positions.iter().map(|p| &p.path).collect::<Vec<_>>());
                        }
                        
                        
                        positions.into_iter().filter(|pos| {
                            let keep = if left_not_smaller && pos.path.get(0) == Some(&0) {
                                true // Position is in left side, which is not smaller
                            } else if right_not_smaller && pos.path.get(0) == Some(&1) {
                                true // Position is in right side, which is not smaller
                            } else {
                                false // Position is in smaller side
                            };
                            
                            
                            keep
                        }).collect()
                    } else {
                        positions // For non-equalities, keep all positions
                    };
                    
                    for pos in valid_positions {
                        // CRITICAL: l' (pos.term) must not be a variable
                        // This prevents unsound inferences like mult(inv(X),X) = mult(e,e)
                        if matches!(pos.term, Term::Variable(_)) {
                            continue;
                        }
                        
                        #[cfg(debug_assertions)]
                        if idx1 == 2 && idx2 == 2 {
                            eprintln!("DEBUG [2->2]: Trying position {:?} with term {}", pos.path, pos.term);
                            eprintln!("  l = {}, pos.term = {}", l, pos.term);
                        }
                        
                        if let Ok(mgu) = unify(l, &pos.term) {
                            #[cfg(debug_assertions)]
                            if (idx1 == 1 && idx2 == 5) || (idx1 == 2 && idx2 == 5) || (idx1 == 0 && idx2 == 4) {
                                eprintln!("\nDEBUG Inference [{},{}]:", idx1, idx2);
                                eprintln!("  From: {} (l={}, r={})", from_clause, l, r);
                                eprintln!("  Into: {}", into_clause);
                                eprintln!("  Into renamed: {}", renamed_into);
                                eprintln!("  Position {:?}: {}", pos.path, pos.term);
                                eprintln!("  MGU: {:?}", mgu);
                            }
                            // CRITICAL CHECK: If r is a variable, we must ensure that after substitution,
                            // the instantiated r is still smaller than the instantiated l
                            // This prevents using equations "backwards" where a variable on the smaller side
                            // matches a term that's actually larger than the left side
                            
                            // Apply substitution to both sides
                            let l_sigma = l.apply_substitution(&mgu);
                            let r_sigma = r.apply_substitution(&mgu);
                            
                            // Check ordering constraint: l ⪯̸ r (which means l > r or l ‖ r)
                            // After substitution, we need l_sigma ⪯̸ r_sigma
                            // This is equivalent to: ¬(l_sigma ≤ r_sigma)
                            
                            match kbo.compare(&l_sigma, &r_sigma) {
                                Ordering::Less | Ordering::Equal => continue, // l_sigma ≤ r_sigma, skip
                                Ordering::Greater | Ordering::Incomparable => {} // l_sigma ⪯̸ r_sigma, proceed
                            }
                            // Additional ordering check already performed above
                            
                            // For equalities (both positive and negative), check ordering constraints
                            if into_lit.atom.is_equality() && into_lit.atom.args.len() == 2 {
                                // Get the two sides of the equality
                                let left_side = &into_lit.atom.args[0].apply_substitution(&mgu);
                                let right_side = &into_lit.atom.args[1].apply_substitution(&mgu);
                                
                                // Determine which argument contains the position
                                let pos_in_left = pos.path.get(0) == Some(&0);
                                let pos_in_right = pos.path.get(0) == Some(&1);
                                
                                if pos_in_left {
                                    // Position is in the left side - check if left ⪯̸ right (left > right or left ‖ right)
                                    match kbo.compare(left_side, right_side) {
                                        Ordering::Less | Ordering::Equal => continue, // left ≤ right, skip
                                        Ordering::Greater | Ordering::Incomparable => {} // left ⪯̸ right, proceed
                                    }
                                } else if pos_in_right {
                                    // Position is in the right side - check if right ⪯̸ left (right > left or right ‖ left)
                                    match kbo.compare(right_side, left_side) {
                                        Ordering::Less | Ordering::Equal => continue, // right ≤ left, skip
                                        Ordering::Greater | Ordering::Incomparable => {} // right ⪯̸ left, proceed
                                    }
                                }
                                
                                // Additional check for positive equalities: if we're at the top level,
                                // ensure the result maintains the ordering
                                if into_lit.polarity && pos.path.len() == 1 {
                                    // We're replacing at the top level of one side
                                    let new_atom = replace_at_position(&into_lit.atom, &pos.path, &l, &mgu);
                                    let new_left = &new_atom.args[0];
                                    let new_right = &new_atom.args[1];
                                    
                                    // The result must maintain proper orientation
                                    if kbo.compare(new_left, new_right) != Ordering::Greater && 
                                       kbo.compare(new_right, new_left) != Ordering::Greater {
                                        // Neither side is larger - would create unorientable equality
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
                                    // Replace the occurrence of l with r at the position
                                    // IMPORTANT: Apply MGU to r before replacement to ensure all variables are substituted
                                    let r_sigma = r.apply_substitution(&mgu);
                                    let new_atom = replace_at_position(&lit.atom, &pos.path, &r_sigma, &mgu);
                                    
                                    
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
/// For equalities, we need to determine which side is maximal and only search there
fn find_unifiable_positions(atom: &Atom, pattern: &Term) -> Vec<Position> {
    let mut positions = Vec::new();
    
    if atom.is_equality() && atom.args.len() == 2 {
        // For equalities, we need to check which side is larger
        // We'll do this check in the main superposition function since we need KBO
        // For now, search in both sides but mark which side each position is in
        find_positions_in_term(&atom.args[0], pattern, vec![0], &mut positions);
        find_positions_in_term(&atom.args[1], pattern, vec![1], &mut positions);
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
        // Test superposition with corrected implementation
        // From: mult(e,X) = X
        // Into: P(mult(e,c)) 
        // Should derive: P(c)
        
        let eq = PredicateSymbol { name: "=".to_string(), arity: 2 };
        let p = PredicateSymbol { name: "P".to_string(), arity: 1 };
        let mult = FunctionSymbol { name: "mult".to_string(), arity: 2 };
        
        let e = Term::Constant(Constant { name: "e".to_string() });
        let c = Term::Constant(Constant { name: "c".to_string() });
        let x = Term::Variable(Variable { name: "X".to_string() });
        let mult_ex = Term::Function(mult.clone(), vec![e.clone(), x.clone()]);
        let mult_ec = Term::Function(mult.clone(), vec![e.clone(), c.clone()]);
        
        // mult(e,X) = X
        let clause1 = Clause::new(vec![
            Literal::positive(Atom { 
                predicate: eq.clone(), 
                args: vec![mult_ex.clone(), x.clone()]
            })
        ]);
        
        // P(mult(e,c))
        let clause2 = Clause::new(vec![
            Literal::positive(Atom { 
                predicate: p.clone(), 
                args: vec![mult_ec.clone()]
            })
        ]);
        
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