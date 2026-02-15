//! Scoped variables for binary inference rules
//!
//! Instead of renaming variables in partner clauses before unification,
//! scoped variables distinguish variables from different parent clauses
//! by pairing each VariableId with a scope tag (u8). This eliminates
//! the O(clause_size) clone + intern cost of `rename_clause_variables`
//! for every unification attempt (most of which fail).

use crate::logic::interner::{ConstantId, FunctionId, Interner, VariableId};
use crate::logic::core::term::{Constant, FunctionSymbol, Term, Variable};
use std::collections::HashMap;

/// A variable tagged with its parent clause scope
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ScopedVar {
    pub scope: u8,
    pub id: VariableId,
}

/// A term with scoped variables
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScopedTerm {
    Variable(ScopedVar),
    Constant(ConstantId),
    Function(FunctionId, Vec<ScopedTerm>),
}

/// Substitution mapping scoped variables to scoped terms
pub type ScopedSubstitution = HashMap<ScopedVar, ScopedTerm>;

/// Lift a concrete Term into a ScopedTerm at the given scope
pub fn lift(term: &Term, scope: u8) -> ScopedTerm {
    match term {
        Term::Variable(v) => ScopedTerm::Variable(ScopedVar { scope, id: v.id }),
        Term::Constant(c) => ScopedTerm::Constant(c.id),
        Term::Function(f, args) => {
            ScopedTerm::Function(f.id, args.iter().map(|a| lift(a, scope)).collect())
        }
    }
}

/// Apply a scoped substitution to a concrete term at the given scope,
/// producing a ScopedTerm
pub fn apply_scoped(term: &Term, scope: u8, subst: &ScopedSubstitution) -> ScopedTerm {
    apply_scoped_term(&lift(term, scope), subst)
}

/// Apply a scoped substitution to a ScopedTerm (chase bindings to fixpoint)
pub fn apply_scoped_term(term: &ScopedTerm, subst: &ScopedSubstitution) -> ScopedTerm {
    match term {
        ScopedTerm::Variable(sv) => {
            if let Some(bound) = subst.get(sv) {
                // Chase the binding (it may itself be a variable)
                apply_scoped_term(bound, subst)
            } else {
                term.clone()
            }
        }
        ScopedTerm::Constant(_) => term.clone(),
        ScopedTerm::Function(f, args) => {
            ScopedTerm::Function(*f, args.iter().map(|a| apply_scoped_term(a, subst)).collect())
        }
    }
}

/// Occurs check: does `var` occur in `term` (after chasing bindings)?
fn scoped_occurs_check(var: &ScopedVar, term: &ScopedTerm, subst: &ScopedSubstitution) -> bool {
    match term {
        ScopedTerm::Variable(sv) => {
            if sv == var {
                return true;
            }
            if let Some(bound) = subst.get(sv) {
                scoped_occurs_check(var, bound, subst)
            } else {
                false
            }
        }
        ScopedTerm::Constant(_) => false,
        ScopedTerm::Function(_, args) => args.iter().any(|a| scoped_occurs_check(var, a, subst)),
    }
}

/// Unify two concrete terms at different scopes
pub fn unify_scoped(
    t1: &Term,
    scope1: u8,
    t2: &Term,
    scope2: u8,
) -> Result<ScopedSubstitution, ()> {
    let st1 = lift(t1, scope1);
    let st2 = lift(t2, scope2);
    let mut subst = ScopedSubstitution::new();
    unify_scoped_terms(&st1, &st2, &mut subst).map(|()| subst)
}

/// Robinson unification on ScopedTerms
pub fn unify_scoped_terms(
    t1: &ScopedTerm,
    t2: &ScopedTerm,
    subst: &mut ScopedSubstitution,
) -> Result<(), ()> {
    // Chase bindings
    let t1 = apply_scoped_term(t1, subst);
    let t2 = apply_scoped_term(t2, subst);

    match (&t1, &t2) {
        _ if t1 == t2 => Ok(()),

        (ScopedTerm::Variable(v), t) | (t, ScopedTerm::Variable(v)) => {
            if scoped_occurs_check(v, t, subst) {
                Err(())
            } else {
                subst.insert(*v, t.clone());
                Ok(())
            }
        }

        (ScopedTerm::Constant(c1), ScopedTerm::Constant(c2)) => {
            if c1 == c2 { Ok(()) } else { Err(()) }
        }

        (ScopedTerm::Function(f1, args1), ScopedTerm::Function(f2, args2)) => {
            if f1 != f2 || args1.len() != args2.len() {
                return Err(());
            }
            for (a1, a2) in args1.iter().zip(args2.iter()) {
                unify_scoped_terms(a1, a2, subst)?;
            }
            Ok(())
        }

        _ => Err(()),
    }
}

/// Convert a ScopedTerm back to a concrete Term.
///
/// - Scope 0 variables keep their original VariableId
/// - Other scopes get fresh interned names (`{name}_{scope}`)
///   only for free variables that survive after substitution application
pub fn flatten(
    term: &ScopedTerm,
    renaming: &mut HashMap<ScopedVar, VariableId>,
    interner: &mut Interner,
) -> Term {
    match term {
        ScopedTerm::Variable(sv) => {
            if let Some(&vid) = renaming.get(sv) {
                Term::Variable(Variable::new(vid))
            } else if sv.scope == 0 {
                // Scope 0: keep original VariableId
                renaming.insert(*sv, sv.id);
                Term::Variable(Variable::new(sv.id))
            } else {
                // Other scopes: intern fresh name
                let old_name = interner.resolve_variable(sv.id).to_owned();
                let new_name = format!("{}_{}", old_name, sv.scope);
                let new_id = interner.intern_variable(&new_name);
                renaming.insert(*sv, new_id);
                Term::Variable(Variable::new(new_id))
            }
        }
        ScopedTerm::Constant(cid) => Term::Constant(Constant::new(*cid)),
        ScopedTerm::Function(fid, args) => {
            // Look up arity from args length
            let arity = args.len() as u8;
            Term::Function(
                FunctionSymbol::new(*fid, arity),
                args.iter().map(|a| flatten(a, renaming, interner)).collect(),
            )
        }
    }
}

/// Apply scoped substitution to a concrete term and flatten in one pass.
///
/// Equivalent to `flatten(apply_scoped(term, scope, subst), renaming, interner)`
/// but avoids the intermediate ScopedTerm allocation in simple cases.
pub fn flatten_scoped(
    term: &Term,
    scope: u8,
    subst: &ScopedSubstitution,
    renaming: &mut HashMap<ScopedVar, VariableId>,
    interner: &mut Interner,
) -> Term {
    // Apply then flatten — the intermediate ScopedTerm is short-lived
    let scoped = apply_scoped(term, scope, subst);
    flatten(&scoped, renaming, interner)
}

#[cfg(test)]
mod tests {
    use super::*;
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

        // X@0 should map to a, X@1 should map to a
        let x_id = if let Term::Variable(v) = &x { v.id } else { panic!() };
        let a_lifted = lift(&a, 0);

        let x0_resolved = apply_scoped_term(
            &ScopedTerm::Variable(ScopedVar { scope: 0, id: x_id }),
            &mgu,
        );
        let x1_resolved = apply_scoped_term(
            &ScopedTerm::Variable(ScopedVar { scope: 1, id: x_id }),
            &mgu,
        );

        assert_eq!(x0_resolved, a_lifted);
        assert_eq!(x1_resolved, a_lifted);
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

        // Unify the arguments pairwise: first X@0 = a@1, then f(X)@0 = Y@1
        // We'll unify the full P(X, f(X)) vs P(a, Y) by unifying args pairwise
        let mut subst = ScopedSubstitution::new();

        // Unify X@0 with a@1
        let sx = lift(&x, 0);
        let sa = lift(&a, 1);
        unify_scoped_terms(&sx, &sa, &mut subst).unwrap();

        // Unify f(X)@0 with Y@1
        let sf_x = lift(&f_x, 0);
        let sy = lift(&y, 1);
        unify_scoped_terms(&sf_x, &sy, &mut subst).unwrap();

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

        // X@0 should resolve to f(X@1)
        let x_id = if let Term::Variable(v) = &x { v.id } else { panic!() };
        let x0_resolved = apply_scoped_term(
            &ScopedTerm::Variable(ScopedVar { scope: 0, id: x_id }),
            &mgu,
        );
        // Should be f(X@1) — not f(X@0) or f(f(X@1))
        let f_id = if let Term::Function(f, _) = &f_x { f.id } else { panic!() };
        assert_eq!(
            x0_resolved,
            ScopedTerm::Function(f_id, vec![ScopedTerm::Variable(ScopedVar { scope: 1, id: x_id })])
        );

        // Flatten: X@1 should get renamed to X_1
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
