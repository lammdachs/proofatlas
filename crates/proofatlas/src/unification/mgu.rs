//! Most General Unifier (MGU) computation

use crate::fol::{ConstantId, FunctionId, Interner, Substitution, Term, Variable, VariableId};
use std::collections::HashSet;

/// Result of a unification attempt
pub type UnificationResult = Result<Substitution, UnificationError>;

/// Errors that can occur during unification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnificationError {
    /// Occurs check failed - variable occurs in term
    OccursCheck(Variable, Term),
    /// Function symbols don't match
    FunctionClash(FunctionId, FunctionId),
    /// Arities don't match
    ArityMismatch(usize, usize),
    /// Constant symbols don't match
    ConstantClash(ConstantId, ConstantId),
    /// Function-constant clash
    FunctionConstantClash(FunctionId, ConstantId),
}

/// Unify two terms, returning a most general unifier (MGU) if one exists
pub fn unify(term1: &Term, term2: &Term) -> UnificationResult {
    let mut subst = Substitution::new();
    unify_with_subst(term1, term2, &mut subst)?;
    Ok(subst)
}

/// Unify two terms with an existing substitution
fn unify_with_subst(
    term1: &Term,
    term2: &Term,
    subst: &mut Substitution,
) -> Result<(), UnificationError> {
    let t1 = term1.apply_substitution(subst);
    let t2 = term2.apply_substitution(subst);

    match (&t1, &t2) {
        // Same term - nothing to do
        _ if t1 == t2 => Ok(()),

        // Variable cases
        (Term::Variable(v), t) | (t, Term::Variable(v)) => {
            if occurs_check(v, t) {
                Err(UnificationError::OccursCheck(*v, t.clone()))
            } else {
                // Use normalized insert to ensure all substitutions are propagated
                subst.insert_normalized(*v, t.clone());
                Ok(())
            }
        }

        // Constant clash
        (Term::Constant(c1), Term::Constant(c2)) => {
            Err(UnificationError::ConstantClash(c1.id, c2.id))
        }

        // Function terms
        (Term::Function(f1, args1), Term::Function(f2, args2)) => {
            if f1.id != f2.id {
                return Err(UnificationError::FunctionClash(f1.id, f2.id));
            }
            if args1.len() != args2.len() {
                return Err(UnificationError::ArityMismatch(args1.len(), args2.len()));
            }

            // Unify arguments pairwise
            for (arg1, arg2) in args1.iter().zip(args2.iter()) {
                unify_with_subst(arg1, arg2, subst)?;
            }
            Ok(())
        }

        // Function-Constant clash
        (Term::Function(f, _), Term::Constant(c)) | (Term::Constant(c), Term::Function(f, _)) => {
            Err(UnificationError::FunctionConstantClash(f.id, c.id))
        }
    }
}

/// Check if variable occurs in term (occurs check)
fn occurs_check(var: &Variable, term: &Term) -> bool {
    match term {
        Term::Variable(v) => v.id == var.id,
        Term::Constant(_) => false,
        Term::Function(_, args) => args.iter().any(|arg| occurs_check(var, arg)),
    }
}

/// Rename variables in a term to avoid conflicts
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

/// Get all variables in a term
pub fn variables_in_term(term: &Term) -> HashSet<Variable> {
    match term {
        Term::Variable(v) => {
            let mut set = HashSet::new();
            set.insert(*v);
            set
        }
        Term::Constant(_) => HashSet::new(),
        Term::Function(_, args) => args.iter().flat_map(variables_in_term).collect(),
    }
}

/// Get all variable IDs in a term
pub fn variable_ids_in_term(term: &Term) -> HashSet<VariableId> {
    match term {
        Term::Variable(v) => {
            let mut set = HashSet::new();
            set.insert(v.id);
            set
        }
        Term::Constant(_) => HashSet::new(),
        Term::Function(_, args) => args.iter().flat_map(variable_ids_in_term).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fol::{Constant, FunctionSymbol};

    /// Test context for building terms with interned symbols
    struct TestContext {
        interner: Interner,
    }

    impl TestContext {
        fn new() -> Self {
            TestContext {
                interner: Interner::new(),
            }
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

        fn var_id(&mut self, name: &str) -> VariableId {
            self.interner.intern_variable(name)
        }
    }

    #[test]
    fn test_unify_variables() {
        let mut ctx = TestContext::new();
        let x = ctx.var("X");
        let y = ctx.var("Y");

        let result = unify(&x, &y).unwrap();
        assert_eq!(result.map.len(), 1);
    }

    #[test]
    fn test_unify_constant_variable() {
        let mut ctx = TestContext::new();
        let x_id = ctx.var_id("X");
        let x = ctx.var("X");
        let a = ctx.const_("a");

        let result = unify(&x, &a).unwrap();
        assert_eq!(result.map.len(), 1);
        assert_eq!(result.map.get(&x_id), Some(&a));
    }

    #[test]
    fn test_unify_functions() {
        let mut ctx = TestContext::new();
        let x = ctx.var("X");
        let y = ctx.var("Y");
        let a = ctx.const_("a");
        let t1 = ctx.func("f", vec![x, y]);
        let a2 = ctx.const_("a");
        let a3 = ctx.const_("a");
        let t2 = ctx.func("f", vec![a2, a3]);

        let result = unify(&t1, &t2).unwrap();
        assert_eq!(result.map.len(), 2);
    }

    #[test]
    fn test_occurs_check() {
        let mut ctx = TestContext::new();
        let x = ctx.var("X");
        let x2 = ctx.var("X");
        let fx = ctx.func("f", vec![x2]);

        let result = unify(&x, &fx);
        assert!(matches!(result, Err(UnificationError::OccursCheck(_, _))));
    }

    #[test]
    fn test_rename_variables() {
        let mut ctx = TestContext::new();
        let x = ctx.var("X");
        let a = ctx.const_("a");
        let term = ctx.func("f", vec![x, a]);

        let renamed = rename_variables(&term, "1", &mut ctx.interner);

        // Check that the renamed variable has a new name
        if let Term::Function(_, args) = &renamed {
            if let Term::Variable(v) = &args[0] {
                assert_eq!(ctx.interner.resolve_variable(v.id), "X_1");
            } else {
                panic!("Expected variable");
            }
        } else {
            panic!("Expected function");
        }
    }
}
