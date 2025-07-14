//! Subsumption checking for clause redundancy elimination

use crate::core::{Clause, Literal, Term, Variable, Substitution};
use std::collections::{HashSet, HashMap};

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
/// clause1 subsumes clause2 if there exists a substitution σ such that
/// clause1σ ⊆ clause2 (all literals in clause1σ appear in clause2)
pub fn subsumes(clause1: &Clause, clause2: &Clause) -> bool {
    // Quick check: if clause1 has more literals, it can't subsume clause2
    if clause1.literals.len() > clause2.literals.len() {
        return false;
    }
    
    // Special case: if clause1 is empty, it subsumes everything
    if clause1.literals.is_empty() {
        return true;
    }
    
    // Try to find a substitution that makes clause1 a subset of clause2
    subsumes_with_matching(clause1, clause2)
}

/// Check subsumption with proper variable matching
fn subsumes_with_matching(clause1: &Clause, clause2: &Clause) -> bool {
    // We need to find an assignment of literals from clause1 to clause2
    // such that all can be unified with a consistent substitution
    
    let n1 = clause1.literals.len();
    let n2 = clause2.literals.len();
    
    // Try all possible mappings of clause1 literals to clause2 literals
    let mut mapping = vec![0; n1];
    
    loop {
        // Check if current mapping works
        if check_mapping(&clause1.literals, &clause2.literals, &mapping) {
            return true;
        }
        
        // Generate next mapping
        if !next_mapping(&mut mapping, n2) {
            break;
        }
    }
    
    false
}

/// Check if a specific mapping of literals works with a consistent substitution
fn check_mapping(lits1: &[Literal], lits2: &[Literal], mapping: &[usize]) -> bool {
    // Check for duplicate mappings (each literal in clause2 can match at most one from clause1)
    let mut used = HashSet::new();
    for &idx in mapping {
        if !used.insert(idx) {
            return false;
        }
    }
    
    // Try to build a consistent substitution
    let mut subst = Substitution::new();
    
    for (i, lit1) in lits1.iter().enumerate() {
        let lit2 = &lits2[mapping[i]];
        
        // Literals must have same polarity
        if lit1.polarity != lit2.polarity {
            return false;
        }
        
        // Try to unify the atoms
        if let Ok(mgu) = unify_atoms_with_subst(&lit1.atom, &lit2.atom, &subst) {
            // Check if new substitution is consistent with existing one
            for (var, term) in mgu.map.iter() {
                if let Some(existing_term) = subst.map.get(var) {
                    // Variable already has a binding, check if consistent
                    if existing_term != term {
                        return false;
                    }
                } else {
                    subst.map.insert(var.clone(), term.clone());
                }
            }
        } else {
            return false;
        }
    }
    
    true
}

/// Unify two atoms with an existing substitution
/// Only allows substitutions for variables from atom1, not atom2
fn unify_atoms_with_subst(
    atom1: &crate::core::Atom, 
    atom2: &crate::core::Atom,
    existing_subst: &Substitution
) -> Result<Substitution, ()> {
    // Atoms must have same predicate
    if atom1.predicate != atom2.predicate {
        return Err(());
    }
    
    // Apply existing substitution first
    let atom1_subst = crate::core::Atom {
        predicate: atom1.predicate.clone(),
        args: atom1.args.iter()
            .map(|t| t.apply_substitution(existing_subst))
            .collect(),
    };
    
    // Now match the arguments - only allowing substitutions from atom1
    let mut subst = Substitution::new();
    for (arg1, arg2) in atom1_subst.args.iter().zip(atom2.args.iter()) {
        if let Err(()) = match_terms_directed(arg1, arg2, &mut subst) {
            return Err(());
        }
    }
    
    Ok(subst)
}

/// Match term1 with term2, only allowing substitutions for variables in term1
fn match_terms_directed(term1: &Term, term2: &Term, subst: &mut Substitution) -> Result<(), ()> {
    let term1_subst = term1.apply_substitution(subst);
    
    match (&term1_subst, term2) {
        (Term::Variable(v1), t2) => {
            // Check if v1 already has a binding
            if let Some(existing) = subst.map.get(v1) {
                // Must match the existing binding
                if existing == t2 {
                    Ok(())
                } else {
                    Err(())
                }
            } else {
                // Add new binding
                subst.map.insert(v1.clone(), t2.clone());
                Ok(())
            }
        }
        (Term::Constant(c1), Term::Constant(c2)) => {
            if c1 == c2 {
                Ok(())
            } else {
                Err(())
            }
        }
        (Term::Function(f1, args1), Term::Function(f2, args2)) => {
            if f1 == f2 && args1.len() == args2.len() {
                for (a1, a2) in args1.iter().zip(args2.iter()) {
                    match_terms_directed(a1, a2, subst)?;
                }
                Ok(())
            } else {
                Err(())
            }
        }
        _ => Err(()),
    }
}

/// Generate next mapping in lexicographic order
fn next_mapping(mapping: &mut [usize], max_val: usize) -> bool {
    for i in (0..mapping.len()).rev() {
        if mapping[i] + 1 < max_val {
            mapping[i] += 1;
            return true;
        }
        mapping[i] = 0;
    }
    false
}

/// Check if we already have an identical clause (modulo variable renaming)
pub fn has_duplicate(clause: &Clause, existing_clauses: &[Clause]) -> bool {
    for existing in existing_clauses {
        if clauses_variant(clause, existing) {
            return true;
        }
    }
    false
}

/// Check if two clauses are variants (identical up to variable renaming)
fn clauses_variant(clause1: &Clause, clause2: &Clause) -> bool {
    if clause1.literals.len() != clause2.literals.len() {
        return false;
    }
    
    // A variant means clause1 subsumes clause2 AND clause2 subsumes clause1
    subsumes(clause1, clause2) && subsumes(clause2, clause1)
}

/// Normalize a clause by renaming variables to a canonical form
/// This is useful for detecting exact duplicates
pub fn normalize_clause(clause: &Clause) -> Clause {
    let mut var_map = HashMap::new();
    let mut next_var = 0;
    
    let normalized_literals = clause.literals.iter()
        .map(|lit| Literal {
            atom: crate::core::Atom {
                predicate: lit.atom.predicate.clone(),
                args: lit.atom.args.iter()
                    .map(|term| normalize_term(term, &mut var_map, &mut next_var))
                    .collect(),
            },
            polarity: lit.polarity,
        })
        .collect();
    
    Clause {
        literals: normalized_literals,
        id: clause.id,
    }
}

fn normalize_term(
    term: &Term, 
    var_map: &mut HashMap<String, usize>,
    next_var: &mut usize
) -> Term {
    match term {
        Term::Variable(v) => {
            let normalized_name = if let Some(&idx) = var_map.get(&v.name) {
                format!("X{}", idx)
            } else {
                let idx = *next_var;
                *next_var += 1;
                var_map.insert(v.name.clone(), idx);
                format!("X{}", idx)
            };
            Term::Variable(Variable { name: normalized_name })
        }
        Term::Constant(c) => Term::Constant(c.clone()),
        Term::Function(f, args) => Term::Function(
            f.clone(),
            args.iter()
                .map(|arg| normalize_term(arg, var_map, next_var))
                .collect(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{PredicateSymbol, Atom, Term, Constant};
    
    #[test]
    fn test_subsumption_simple() {
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
    
    #[test]
    fn test_subsumption_with_variables() {
        let p = PredicateSymbol { name: "P".to_string(), arity: 1 };
        let x = Term::Variable(Variable { name: "X".to_string() });
        let a = Term::Constant(Constant { name: "a".to_string() });
        
        // P(X) subsumes P(a)
        let clause1 = Clause::new(vec![
            Literal::positive(Atom { predicate: p.clone(), args: vec![x.clone()] })
        ]);
        
        let clause2 = Clause::new(vec![
            Literal::positive(Atom { predicate: p.clone(), args: vec![a.clone()] })
        ]);
        
        assert!(subsumes(&clause1, &clause2));
        assert!(!subsumes(&clause2, &clause1));
    }
    
    #[test]
    fn test_variant_detection() {
        let p = PredicateSymbol { name: "P".to_string(), arity: 2 };
        let x = Term::Variable(Variable { name: "X".to_string() });
        let y = Term::Variable(Variable { name: "Y".to_string() });
        let x_prime = Term::Variable(Variable { name: "X'".to_string() });
        let y_prime = Term::Variable(Variable { name: "Y'".to_string() });
        
        // P(X,Y) and P(X',Y') are variants
        let clause1 = Clause::new(vec![
            Literal::positive(Atom { 
                predicate: p.clone(), 
                args: vec![x.clone(), y.clone()] 
            })
        ]);
        
        let clause2 = Clause::new(vec![
            Literal::positive(Atom { 
                predicate: p.clone(), 
                args: vec![x_prime.clone(), y_prime.clone()] 
            })
        ]);
        
        assert!(clauses_variant(&clause1, &clause2));
    }
}