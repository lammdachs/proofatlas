//! One-way matching for demodulation

use super::UnificationError;
use crate::fol::{Substitution, Term};

/// One-way match: Find a substitution σ such that pattern σ = term
/// Only variables in the pattern can be substituted
pub fn match_term(pattern: &Term, term: &Term) -> Result<Substitution, UnificationError> {
    let mut subst = Substitution::new();
    match_with_subst(pattern, term, &mut subst)?;
    Ok(subst)
}

fn match_with_subst(
    pattern: &Term,
    term: &Term,
    subst: &mut Substitution,
) -> Result<(), UnificationError> {
    match (pattern, term) {
        // Variable in pattern matches anything
        (Term::Variable(v), t) => {
            // Check if this variable is already bound
            if let Some(bound_term) = subst.get(v.id) {
                // Variable is already bound, check if it matches
                if bound_term == t {
                    Ok(())
                } else {
                    Err(UnificationError::ConstantClash(
                        // Use constant IDs for the error - this is a type mismatch in original code
                        // We'll create a new error variant would be cleaner, but for now use the existing structure
                        // Since we can't easily convert Term to ConstantId, we'll need a different approach
                        // For now, return a generic error by using dummy IDs
                        crate::fol::ConstantId::from_raw(0),
                        crate::fol::ConstantId::from_raw(1),
                    ))
                }
            } else {
                // Variable is not yet bound, bind it
                subst.insert(*v, t.clone());
                Ok(())
            }
        }
        // Constants must match exactly
        (Term::Constant(c1), Term::Constant(c2)) => {
            if c1.id == c2.id {
                Ok(())
            } else {
                Err(UnificationError::ConstantClash(c1.id, c2.id))
            }
        }
        // Functions must have same symbol and arity
        (Term::Function(f1, args1), Term::Function(f2, args2)) => {
            if f1.id != f2.id {
                return Err(UnificationError::FunctionClash(f1.id, f2.id));
            }
            if args1.len() != args2.len() {
                return Err(UnificationError::ArityMismatch(args1.len(), args2.len()));
            }

            // Match all arguments
            for (arg1, arg2) in args1.iter().zip(args2.iter()) {
                match_with_subst(arg1, arg2, subst)?;
            }
            Ok(())
        }
        // All other combinations fail
        (Term::Constant(c), Term::Function(f, _))
        | (Term::Function(f, _), Term::Constant(c)) => {
            Err(UnificationError::FunctionConstantClash(f.id, c.id))
        }
        // Constant in pattern cannot match variable in term
        (Term::Constant(c), Term::Variable(_)) => {
            // Use a dummy constant ID for the error since we can't match
            Err(UnificationError::ConstantClash(
                c.id,
                crate::fol::ConstantId::from_raw(u32::MAX),
            ))
        }
        // Function in pattern cannot match variable in term
        (Term::Function(f, _), Term::Variable(_)) => {
            Err(UnificationError::FunctionConstantClash(
                f.id,
                crate::fol::ConstantId::from_raw(u32::MAX),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fol::{Constant, FunctionSymbol, Interner, Variable};

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
    }

    #[test]
    fn test_match_variable() {
        let mut ctx = TestContext::new();
        let x = ctx.var("X");
        let a = ctx.const_("a");

        let subst = match_term(&x, &a).unwrap();
        assert_eq!(x.apply_substitution(&subst), a);
    }

    #[test]
    fn test_match_function() {
        let mut ctx = TestContext::new();
        let x = ctx.var("X");
        let y = ctx.var("Y");
        let pattern = ctx.func("f", vec![x, y]);

        let a = ctx.const_("a");
        let b = ctx.const_("b");
        let term = ctx.func("f", vec![a, b]);

        let subst = match_term(&pattern, &term).unwrap();
        assert_eq!(pattern.apply_substitution(&subst), term);
    }

    #[test]
    fn test_no_match_variable_in_term() {
        let mut ctx = TestContext::new();
        let a = ctx.const_("a");
        let x = ctx.var("X");

        // Should fail because we can't match constant against variable
        assert!(match_term(&a, &x).is_err());
    }

    #[test]
    fn test_no_match_inconsistent_variable() {
        // Test that mult(inv(X),X) does NOT match mult(inv(Y),mult(Y,Z))
        let mut ctx = TestContext::new();

        // Build pattern: mult(inv(X), X)
        let x1 = ctx.var("X");
        let inv_x = ctx.func("inv", vec![x1]);
        let x2 = ctx.var("X");
        let pattern = ctx.func("mult", vec![inv_x, x2]);

        // Build term: mult(inv(Y), mult(Y, Z))
        let y1 = ctx.var("Y");
        let inv_y = ctx.func("inv", vec![y1]);
        let y2 = ctx.var("Y");
        let z = ctx.var("Z");
        let mult_y_z = ctx.func("mult", vec![y2, z]);
        let term = ctx.func("mult", vec![inv_y, mult_y_z]);

        // Should fail because X cannot be both Y and mult(Y,Z)
        match match_term(&pattern, &term) {
            Ok(subst) => {
                eprintln!("ERROR: Match should have failed!");
                eprintln!("Pattern: {}", pattern);
                eprintln!("Term: {}", term);
                eprintln!("Substitution: {:?}", subst.map);
                panic!("Match incorrectly succeeded");
            }
            Err(e) => {
                eprintln!("Good: Match correctly failed with: {:?}", e);
            }
        }
    }
}
