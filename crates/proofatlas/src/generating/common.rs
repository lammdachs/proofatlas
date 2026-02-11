//! Common types and utilities for inference rules

use crate::logic::{Clause, Interner, Literal, PredicateSymbol, Substitution, Term, TermOrdering, Variable, KBO};
use crate::logic::unify;
use std::collections::HashSet;

/// Rename all variables in a clause to avoid conflicts
pub fn rename_clause_variables(clause: &Clause, suffix: &str, interner: &mut Interner) -> Clause {
    Clause {
        literals: clause
            .literals
            .iter()
            .map(|lit| Literal {
                predicate: lit.predicate,
                args: lit
                    .args
                    .iter()
                    .map(|arg| rename_variables(arg, suffix, interner))
                    .collect(),
                polarity: lit.polarity,
            })
            .collect(),
        id: clause.id,
        role: clause.role,
        age: clause.age,
        derivation_rule: clause.derivation_rule,
    }
}

/// Rename variables in a term
pub fn rename_variables(term: &Term, suffix: &str, interner: &mut Interner) -> Term {
    match term {
        Term::Variable(v) => {
            let old_name = interner.resolve_variable(v.id);
            let new_name = format!("{}_{}", old_name, suffix);
            let new_id = interner.intern_variable(&new_name);
            Term::Variable(Variable::new(new_id))
        }
        Term::Constant(c) => Term::Constant(*c),
        Term::Function(f, args) => Term::Function(
            *f,
            args.iter()
                .map(|arg| rename_variables(arg, suffix, interner))
                .collect(),
        ),
    }
}

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

/// Remove duplicate literals from a list
pub fn remove_duplicate_literals(literals: Vec<Literal>) -> Vec<Literal> {
    let mut seen = HashSet::new();
    let mut result = Vec::new();

    for lit in literals {
        if seen.insert(lit.clone()) {
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
