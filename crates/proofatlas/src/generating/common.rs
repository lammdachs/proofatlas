//! Common types and utilities for inference rules

use crate::logic::{Clause, Interner, Literal, PredicateSymbol, Substitution, Term, TermOrdering, KBO};
use crate::logic::unify;
use crate::logic::unification::scoped::{
    ScopedSubstitution, ScopedVar, flatten_scoped, unify_scoped_extend,
};
use std::collections::HashMap;

/// Unify the predicate/args of two literals (or atoms).
///
/// Returns the most general unifier if the predicates match and all
/// argument pairs are unifiable, otherwise returns Err.
pub fn unify_atoms(
    pred1: PredicateSymbol,
    args1: &[Term],
    pred2: PredicateSymbol,
    args2: &[Term],
) -> Result<Substitution, ()> {
    if pred1 != pred2 || args1.len() != args2.len() {
        return Err(());
    }

    let mut subst = Substitution::new();
    for (arg1, arg2) in args1.iter().zip(args2.iter()) {
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

/// Remove duplicate literals from a list.
/// Uses equality comparison instead of HashSet to avoid cloning each literal.
/// O(n^2) but n is typically 2-10 for clause literals.
pub fn remove_duplicate_literals(literals: Vec<Literal>) -> Vec<Literal> {
    let mut result = Vec::with_capacity(literals.len());
    for lit in literals {
        if !result.contains(&lit) {
            result.push(lit);
        }
    }
    result
}

/// Collect literals from a clause, excluding specified indices, with substitution applied.
///
/// Used by inference rules to collect side literals while excluding the literals
/// being resolved/factored/etc.
pub fn collect_literals_except(
    clause: &Clause,
    exclude: &[usize],
    subst: &Substitution,
) -> Vec<Literal> {
    clause
        .literals
        .iter()
        .enumerate()
        .filter(|(i, _)| !exclude.contains(i))
        .map(|(_, lit)| lit.apply_substitution(subst))
        .collect()
}

/// Check if t1 is "ordered greater" than t2 according to KBO.
///
/// Returns true if t1 > t2 or t1 and t2 are incomparable.
/// This is used for the ordering constraint (not smaller than),
/// which means Greater or Incomparable.
pub fn is_ordered_greater(t1: &Term, t2: &Term, kbo: &KBO) -> bool {
    matches!(
        kbo.compare(t1, t2),
        TermOrdering::Greater | TermOrdering::Incomparable
    )
}

/// Unify atoms from two different clauses using scoped variables.
///
/// Returns a ScopedSubstitution if the predicates match and all
/// argument pairs are unifiable across the two scopes.
/// Uses direct Term unification to avoid intermediate ScopedTerm allocation.
pub fn unify_atoms_scoped(
    pred1: PredicateSymbol,
    args1: &[Term],
    scope1: u8,
    pred2: PredicateSymbol,
    args2: &[Term],
    scope2: u8,
) -> Result<ScopedSubstitution, ()> {
    if pred1 != pred2 || args1.len() != args2.len() {
        return Err(());
    }

    let mut subst = ScopedSubstitution::new();
    for (arg1, arg2) in args1.iter().zip(args2.iter()) {
        unify_scoped_extend(arg1, scope1, arg2, scope2, &mut subst)?;
    }

    Ok(subst)
}

/// Collect literals from a clause, excluding indices, applying scoped subst + flatten.
pub fn collect_scoped_literals_except(
    clause: &Clause,
    exclude: &[usize],
    scope: u8,
    subst: &ScopedSubstitution,
    renaming: &mut HashMap<ScopedVar, crate::logic::VariableId>,
    interner: &mut Interner,
) -> Vec<Literal> {
    clause
        .literals
        .iter()
        .enumerate()
        .filter(|(i, _)| !exclude.contains(i))
        .map(|(_, lit)| {
            Literal {
                predicate: lit.predicate,
                args: lit.args.iter().map(|arg| flatten_scoped(arg, scope, subst, renaming, interner)).collect(),
                polarity: lit.polarity,
            }
        })
        .collect()
}
