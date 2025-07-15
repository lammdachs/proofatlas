//! Demodulation inference rule for term rewriting

use crate::core::{Clause, Literal, Atom, Term, KBO, KBOConfig, TermOrdering as Ordering};
use crate::unification::match_terms;
use super::common::{InferenceResult, InferenceRule};

/// Apply demodulation rule using unit equalities
/// From l ≈ r and P[t] ∨ C where t matches lσ for some substitution σ, and lσ ≻ rσ
/// Derive P[t'] ∨ C where t' is t with the matched subterm replaced by rσ
/// This is a simplifying inference that replaces terms with simpler ones
pub fn demodulate(unit_eq: &Clause, target: &Clause, unit_idx: usize, target_idx: usize) -> Vec<InferenceResult> {
    let mut results = Vec::new();
    let kbo = KBO::new(KBOConfig::default());
    
    // Check if unit_eq is a unit equality
    if unit_eq.literals.len() != 1 || !unit_eq.literals[0].polarity || !unit_eq.literals[0].atom.is_equality() {
        return results;
    }
    
    let eq_lit = &unit_eq.literals[0];
    if let [ref l, ref r] = eq_lit.atom.args.as_slice() {
        // Check ordering constraint: l must be greater than r
        match kbo.compare(l, r) {
            Ordering::Greater => {}, // Good, l > r
            _ => return results, // Skip if l ≤ r or incomparable
        }
        
        // Try to find matches of l in the target clause
        let mut clause_modified = false;
        let mut new_literals = Vec::new();
        
        for literal in target.literals.iter() {
            if let Some(new_atom) = demodulate_atom(&literal.atom, l, r, &kbo) {
                clause_modified = true;
                new_literals.push(Literal {
                    atom: new_atom,
                    polarity: literal.polarity,
                });
            } else {
                new_literals.push(literal.clone());
            }
        }
        
        if clause_modified {
            let conclusion = Clause::new(new_literals);
            
            // Apply demodulation - the ordering constraint in demodulate_term already ensures
            // we're moving toward a normal form
            results.push(InferenceResult {
                rule: InferenceRule::Demodulation,
                premises: vec![unit_idx, target_idx],
                conclusion,
            });
        }
    }
    
    results
}

/// Demodulate an atom by replacing matches of l with r
fn demodulate_atom(atom: &Atom, l: &Term, r: &Term, kbo: &KBO) -> Option<Atom> {
    let mut modified = false;
    let mut new_args = Vec::new();
    
    for arg in &atom.args {
        if let Some(new_term) = demodulate_term(arg, l, r, kbo) {
            modified = true;
            new_args.push(new_term);
        } else {
            new_args.push(arg.clone());
        }
    }
    
    if modified {
        Some(Atom {
            predicate: atom.predicate.clone(),
            args: new_args,
        })
    } else {
        None
    }
}

/// Demodulate a term by replacing matches of l with r
fn demodulate_term(term: &Term, l: &Term, r: &Term, kbo: &KBO) -> Option<Term> {
    // Try to match l with term
    if let Ok(sigma) = match_terms(l, term) {
        // Apply substitution to both l and r
        let l_sigma = l.apply_substitution(&sigma);
        let r_sigma = r.apply_substitution(&sigma);
        
        // Check ordering constraint: lσ ≻ rσ
        match kbo.compare(&l_sigma, &r_sigma) {
            Ordering::Greater => {
                // Good, we can replace
                return Some(r_sigma);
            }
            _ => {} // Can't apply this match
        }
    }
    
    // If no match at this level, recursively check subterms
    match term {
        Term::Variable(_) | Term::Constant(_) => None,
        Term::Function(f, args) => {
            let mut modified = false;
            let mut new_args = Vec::new();
            
            for arg in args {
                if let Some(new_arg) = demodulate_term(arg, l, r, kbo) {
                    modified = true;
                    new_args.push(new_arg);
                } else {
                    new_args.push(arg.clone());
                }
            }
            
            if modified {
                Some(Term::Function(f.clone(), new_args))
            } else {
                None
            }
        }
    }
}

