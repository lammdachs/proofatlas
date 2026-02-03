//! Common types and utilities for inference rules

use crate::fol::{Atom, Clause, Literal, Substitution, Term, Variable};
use super::derivation::Derivation;
use crate::unification::unify;
use std::collections::HashSet;

/// Result of an inference rule application
#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub derivation: Derivation,
    pub conclusion: Clause,
}

/// Rename all variables in a clause to avoid conflicts
pub fn rename_clause_variables(clause: &Clause, suffix: &str) -> Clause {
    Clause {
        literals: clause
            .literals
            .iter()
            .map(|lit| Literal {
                atom: Atom {
                    predicate: lit.atom.predicate.clone(),
                    args: lit
                        .atom
                        .args
                        .iter()
                        .map(|arg| rename_variables(arg, suffix))
                        .collect(),
                },
                polarity: lit.polarity,
            })
            .collect(),
        id: clause.id,
        role: clause.role,
        age: clause.age,
    }
}

/// Rename variables in a term
pub fn rename_variables(term: &Term, suffix: &str) -> Term {
    match term {
        Term::Variable(v) => Term::Variable(Variable {
            name: format!("{}_{}", v.name, suffix),
        }),
        Term::Constant(c) => Term::Constant(c.clone()),
        Term::Function(f, args) => Term::Function(
            f.clone(),
            args.iter()
                .map(|arg| rename_variables(arg, suffix))
                .collect(),
        ),
    }
}

/// Unify two atoms
pub fn unify_atoms(atom1: &Atom, atom2: &Atom) -> Result<Substitution, ()> {
    if atom1.predicate != atom2.predicate || atom1.args.len() != atom2.args.len() {
        return Err(());
    }

    let mut subst = Substitution::new();
    for (arg1, arg2) in atom1.args.iter().zip(atom2.args.iter()) {
        // Apply current substitution before unifying
        let arg1_subst = arg1.apply_substitution(&subst);
        let arg2_subst = arg2.apply_substitution(&subst);

        if let Ok(mgu) = unify(&arg1_subst, &arg2_subst) {
            subst = subst.compose(&mgu);
        } else {
            return Err(());
        }
    }

    Ok(subst)
}

/// Remove duplicate literals from a list
pub fn remove_duplicate_literals(literals: Vec<Literal>) -> Vec<Literal> {
    let mut seen = HashSet::new();
    let mut result = Vec::new();

    for lit in literals {
        if seen.insert((lit.atom.clone(), lit.polarity)) {
            result.push(lit);
        }
    }

    result
}
