//! Subsumption checking for clause redundancy elimination

use crate::core::{Clause, Literal};
use std::collections::HashSet;

/// Check if a clause is subsumed by any existing clause
pub fn is_subsumed(clause: &Clause, existing_clauses: &[Clause]) -> bool {
    for existing in existing_clauses {
        if subsumes(existing, clause) {
            return true;
        }
    }
    false
}

/// Check if clause1 subsumes clause2
/// clause1 subsumes clause2 if all literals in clause1 appear in clause2
pub fn subsumes(clause1: &Clause, clause2: &Clause) -> bool {
    // Simple subsumption check without variable matching
    // TODO: Implement proper subsumption with variable matching
    
    if clause1.literals.len() > clause2.literals.len() {
        return false;
    }
    
    'outer: for lit1 in &clause1.literals {
        for lit2 in &clause2.literals {
            if literals_match(lit1, lit2) {
                continue 'outer;
            }
        }
        return false;
    }
    
    true
}

/// Check if we already have an identical clause
pub fn has_duplicate(clause: &Clause, existing_clauses: &[Clause]) -> bool {
    for existing in existing_clauses {
        if clauses_identical(clause, existing) {
            return true;
        }
    }
    false
}

/// Check if two literals match (same atom and polarity)
fn literals_match(lit1: &Literal, lit2: &Literal) -> bool {
    lit1.polarity == lit2.polarity && lit1.atom == lit2.atom
}

/// Check if two clauses are identical (same literals in any order)
fn clauses_identical(clause1: &Clause, clause2: &Clause) -> bool {
    if clause1.literals.len() != clause2.literals.len() {
        return false;
    }
    
    let set1: HashSet<_> = clause1.literals.iter().collect();
    let set2: HashSet<_> = clause2.literals.iter().collect();
    
    set1 == set2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{PredicateSymbol, Atom, Term, Constant};
    
    #[test]
    fn test_subsumption() {
        let p = PredicateSymbol { name: "P".to_string(), arity: 1 };
        let q = PredicateSymbol { name: "Q".to_string(), arity: 1 };
        let a = Term::Constant(Constant { name: "a".to_string() });
        
        // P(a) subsumes P(a) ∨ Q(a)
        let clause1 = Clause::new(vec![
            Literal::positive(Atom { predicate: p.clone(), args: vec![a.clone()] })
        ]);
        
        let clause2 = Clause::new(vec![
            Literal::positive(Atom { predicate: p.clone(), args: vec![a.clone()] }),
            Literal::positive(Atom { predicate: q.clone(), args: vec![a.clone()] })
        ]);
        
        assert!(subsumes(&clause1, &clause2));
        assert!(!subsumes(&clause2, &clause1));
    }
}