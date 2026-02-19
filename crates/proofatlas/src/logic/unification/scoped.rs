//! Scoped variables for binary inference rules
//!
//! Instead of renaming variables in partner clauses before unification,
//! scoped variables distinguish variables from different parent clauses
//! by pairing each VariableId with a scope tag (u8). This eliminates
//! the O(clause_size) clone + intern cost of `rename_clause_variables`
//! for every unification attempt (most of which fail).
//!
//! The substitution maps `ScopedVar` to `(Term, u8)` pairs, meaning
//! "this term with all its variables interpreted at this scope."
//! This avoids any intermediate `ScopedTerm` representation — unification
//! works directly on `(&Term, scope)` pairs with zero allocation on the
//! common failure path.

use crate::logic::interner::{Interner, VariableId};
use crate::logic::core::term::{Term, Variable};
use std::collections::HashMap;

/// A variable tagged with its parent clause scope
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ScopedVar {
    pub scope: u8,
    pub id: VariableId,
}

/// Substitution mapping scoped variables to (Term, scope) pairs.
/// The scope indicates what scope all variables in the Term belong to.
pub type ScopedSubstitution = HashMap<ScopedVar, (Term, u8)>;

/// Unify two concrete terms at different scopes.
pub fn unify_scoped(
    t1: &Term,
    scope1: u8,
    t2: &Term,
    scope2: u8,
) -> Result<ScopedSubstitution, ()> {
    let mut subst = ScopedSubstitution::new();
    unify_terms(t1, scope1, t2, scope2, &mut subst)?;
    Ok(subst)
}

/// Extend an existing scoped substitution with a new unification constraint.
pub fn unify_scoped_extend(
    t1: &Term,
    scope1: u8,
    t2: &Term,
    scope2: u8,
    subst: &mut ScopedSubstitution,
) -> Result<(), ()> {
    unify_terms(t1, scope1, t2, scope2, subst)
}

/// Core Robinson unification on (Term, scope) pairs.
/// Zero allocation on the failure path for non-variable terms.
fn unify_terms(
    t1: &Term,
    s1: u8,
    t2: &Term,
    s2: u8,
    subst: &mut ScopedSubstitution,
) -> Result<(), ()> {
    match (t1, t2) {
        (Term::Variable(v1), Term::Variable(v2)) => {
            let sv1 = ScopedVar { scope: s1, id: v1.id };
            let sv2 = ScopedVar { scope: s2, id: v2.id };
            if sv1 == sv2 {
                return Ok(());
            }
            match (subst.get(&sv1).cloned(), subst.get(&sv2).cloned()) {
                (Some((bt1, bs1)), Some((bt2, bs2))) => {
                    unify_terms(&bt1, bs1, &bt2, bs2, subst)
                }
                (Some((bt, bs)), None) => {
                    if occurs_in_term(&sv2, &bt, bs, subst) {
                        return Err(());
                    }
                    subst.insert(sv2, (bt, bs));
                    Ok(())
                }
                (None, Some((bt, bs))) => {
                    if occurs_in_term(&sv1, &bt, bs, subst) {
                        return Err(());
                    }
                    subst.insert(sv1, (bt, bs));
                    Ok(())
                }
                (None, None) => {
                    subst.insert(sv1, (Term::Variable(*v2), s2));
                    Ok(())
                }
            }
        }
        (Term::Variable(v1), _) => {
            let sv1 = ScopedVar { scope: s1, id: v1.id };
            match subst.get(&sv1).cloned() {
                Some((bound, bs)) => unify_terms(&bound, bs, t2, s2, subst),
                None => {
                    if occurs_in_term(&sv1, t2, s2, subst) {
                        return Err(());
                    }
                    subst.insert(sv1, (t2.clone(), s2));
                    Ok(())
                }
            }
        }
        (_, Term::Variable(v2)) => {
            let sv2 = ScopedVar { scope: s2, id: v2.id };
            match subst.get(&sv2).cloned() {
                Some((bound, bs)) => unify_terms(t1, s1, &bound, bs, subst),
                None => {
                    if occurs_in_term(&sv2, t1, s1, subst) {
                        return Err(());
                    }
                    subst.insert(sv2, (t1.clone(), s1));
                    Ok(())
                }
            }
        }
        (Term::Constant(c1), Term::Constant(c2)) => {
            if c1.id == c2.id {
                Ok(())
            } else {
                Err(())
            }
        }
        (Term::Function(f1, args1), Term::Function(f2, args2)) => {
            if f1.id != f2.id || args1.len() != args2.len() {
                return Err(());
            }
            for (a1, a2) in args1.iter().zip(args2.iter()) {
                unify_terms(a1, s1, a2, s2, subst)?;
            }
            Ok(())
        }
        _ => Err(()),
    }
}

/// Occurs check: does `var` occur in `term` at `scope` (after chasing bindings)?
fn occurs_in_term(var: &ScopedVar, term: &Term, scope: u8, subst: &ScopedSubstitution) -> bool {
    match term {
        Term::Variable(v) => {
            let sv = ScopedVar { scope, id: v.id };
            if sv == *var {
                return true;
            }
            if let Some((bound, bs)) = subst.get(&sv) {
                occurs_in_term(var, bound, *bs, subst)
            } else {
                false
            }
        }
        Term::Constant(_) => false,
        Term::Function(_, args) => args.iter().any(|a| occurs_in_term(var, a, scope, subst)),
    }
}

/// Apply scoped substitution to a concrete term and flatten in one pass.
///
/// Walks the concrete Term, chasing bindings as needed, producing a
/// concrete Term with scope-0 variables keeping original IDs and
/// other-scope variables getting fresh `{name}_{scope}` names.
pub fn flatten_scoped(
    term: &Term,
    scope: u8,
    subst: &ScopedSubstitution,
    renaming: &mut HashMap<ScopedVar, VariableId>,
    interner: &mut Interner,
) -> Term {
    match term {
        Term::Variable(v) => {
            let sv = ScopedVar { scope, id: v.id };
            match subst.get(&sv) {
                Some((bound, bs)) => flatten_scoped(bound, *bs, subst, renaming, interner),
                None => resolve_scoped_var(sv, renaming, interner),
            }
        }
        Term::Constant(c) => Term::Constant(*c),
        Term::Function(f, args) => Term::Function(
            *f,
            args.iter()
                .map(|a| flatten_scoped(a, scope, subst, renaming, interner))
                .collect(),
        ),
    }
}

/// Resolve a free scoped variable to a concrete Term::Variable.
/// Scope 0 keeps original VariableId, other scopes get fresh interned names.
fn resolve_scoped_var(
    sv: ScopedVar,
    renaming: &mut HashMap<ScopedVar, VariableId>,
    interner: &mut Interner,
) -> Term {
    if let Some(&vid) = renaming.get(&sv) {
        Term::Variable(Variable::new(vid))
    } else if sv.scope == 0 {
        renaming.insert(sv, sv.id);
        Term::Variable(Variable::new(sv.id))
    } else {
        let old_name = interner.resolve_variable(sv.id).to_owned();
        let new_name = format!("{}_{}", old_name, sv.scope);
        let new_id = interner.intern_variable(&new_name);
        renaming.insert(sv, new_id);
        Term::Variable(Variable::new(new_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::core::term::{Constant, FunctionSymbol};
    use crate::logic::Interner;

    struct Ctx {
        interner: Interner,
    }

    impl Ctx {
        fn new() -> Self {
            Ctx { interner: Interner::new() }
        }

        fn var(&mut self, name: &str) -> Term {
            let id = self.interner.intern_variable(name);
            Term::Variable(Variable::new(id))
        }

        fn const_(&mut self, name: &str) -> Term {
            let id = self.interner.intern_constant(name);
            Term::Constant(Constant::new(id))
        }

        fn func(&mut self, name: &str, args: Vec<Term>) -> Term {
            let id = self.interner.intern_function(name);
            Term::Function(FunctionSymbol::new(id, args.len() as u8), args)
        }
    }

    #[test]
    fn test_scoped_unify_same_var_different_scopes() {
        // X (scope 0) and X (scope 1) are DIFFERENT variables
        let mut ctx = Ctx::new();
        let x = ctx.var("X");
        let a = ctx.const_("a");

        // f(X, a) at scope 0 vs f(a, X) at scope 1
        let t1 = ctx.func("f", vec![x.clone(), a.clone()]);
        let t2 = ctx.func("f", vec![a.clone(), x.clone()]);

        let mgu = unify_scoped(&t1, 0, &t2, 1).unwrap();

        // Both X@0 and X@1 should resolve to a
        let mut renaming = HashMap::new();
        let x0_resolved = flatten_scoped(&x, 0, &mgu, &mut renaming, &mut ctx.interner);
        let x1_resolved = flatten_scoped(&x, 1, &mgu, &mut renaming, &mut ctx.interner);
        assert_eq!(x0_resolved, a);
        assert_eq!(x1_resolved, a);
    }

    #[test]
    fn test_scoped_unify_shared_var() {
        // Clause1: P(X, f(X))  at scope 0
        // Clause2: P(a, Y)     at scope 1
        // MGU: X@0 = a, Y@1 = f(a)
        let mut ctx = Ctx::new();
        let x = ctx.var("X");
        let y = ctx.var("Y");
        let a = ctx.const_("a");
        let f_x = ctx.func("f", vec![x.clone()]);

        // Unify the arguments pairwise using unify_scoped_extend
        let mut subst = ScopedSubstitution::new();
        unify_scoped_extend(&x, 0, &a, 1, &mut subst).unwrap();
        unify_scoped_extend(&f_x, 0, &y, 1, &mut subst).unwrap();

        // Now flatten f(X)@0 under the substitution: should give f(a)
        let mut renaming = HashMap::new();
        let result = flatten_scoped(&f_x, 0, &subst, &mut renaming, &mut ctx.interner);
        assert_eq!(result, ctx.func("f", vec![a.clone()]));
    }

    #[test]
    fn test_flatten_preserves_scope0_ids() {
        let mut ctx = Ctx::new();
        let x = ctx.var("X");
        let x_id = if let Term::Variable(v) = &x { v.id } else { panic!() };

        // Empty substitution: X@0 should keep its original VariableId
        let subst = ScopedSubstitution::new();
        let mut renaming = HashMap::new();
        let result = flatten_scoped(&x, 0, &subst, &mut renaming, &mut ctx.interner);

        if let Term::Variable(v) = result {
            assert_eq!(v.id, x_id);
        } else {
            panic!("Expected variable");
        }
    }

    #[test]
    fn test_flatten_renames_scope1() {
        let mut ctx = Ctx::new();
        let x = ctx.var("X");
        let x_id = if let Term::Variable(v) = &x { v.id } else { panic!() };

        // Empty substitution: X@1 should get a fresh name "X_1"
        let subst = ScopedSubstitution::new();
        let mut renaming = HashMap::new();
        let result = flatten_scoped(&x, 1, &subst, &mut renaming, &mut ctx.interner);

        if let Term::Variable(v) = result {
            assert_ne!(v.id, x_id);
            assert_eq!(ctx.interner.resolve_variable(v.id), "X_1");
        } else {
            panic!("Expected variable");
        }
    }

    #[test]
    fn test_unify_fails_occurs_check() {
        let mut ctx = Ctx::new();
        let x = ctx.var("X");
        let f_x = ctx.func("f", vec![x.clone()]);

        // X@0 vs f(X)@0 — same scope, occurs check should fail
        assert!(unify_scoped(&x, 0, &f_x, 0).is_err());
    }

    #[test]
    fn test_unify_function_clash() {
        let mut ctx = Ctx::new();
        let a = ctx.const_("a");
        let t1 = ctx.func("f", vec![a.clone()]);
        let t2 = ctx.func("g", vec![a.clone()]);

        assert!(unify_scoped(&t1, 0, &t2, 1).is_err());
    }

    #[test]
    fn test_self_resolution_same_var_names() {
        // Simulates self-resolution where both clauses are the same
        // P(X) v ~P(f(X)) resolved with itself:
        // Clause at scope 0: P(X@0) v ~P(f(X@0))
        // Clause at scope 1: P(X@1) v ~P(f(X@1))
        // Resolve P(X@0) with ~P(f(X@1)):
        //   X@0 = f(X@1)
        let mut ctx = Ctx::new();
        let x = ctx.var("X");
        let f_x = ctx.func("f", vec![x.clone()]);

        let mgu = unify_scoped(&x, 0, &f_x, 1).unwrap();

        // Flatten: X@0 should give f(X_1) since X@0 = f(X)@1
        let mut renaming = HashMap::new();
        let result = flatten_scoped(&x, 0, &mgu, &mut renaming, &mut ctx.interner);
        if let Term::Function(_, args) = &result {
            if let Term::Variable(v) = &args[0] {
                assert_eq!(ctx.interner.resolve_variable(v.id), "X_1");
            } else {
                panic!("Expected variable in f(...)");
            }
        } else {
            panic!("Expected function f(X_1)");
        }
    }
}
