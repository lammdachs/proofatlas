//! One-way matching for demodulation

use super::UnificationError;
use crate::core::{Substitution, Term};

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
            if let Some(bound_term) = subst.map.get(v) {
                // Variable is already bound, check if it matches
                if bound_term == t {
                    Ok(())
                } else {
                    Err(UnificationError::ConstantClash(
                        bound_term.to_string(),
                        t.to_string(),
                    ))
                }
            } else {
                // Variable is not yet bound, bind it
                subst.insert(v.clone(), t.clone());
                Ok(())
            }
        }
        // Constants must match exactly
        (Term::Constant(c1), Term::Constant(c2)) => {
            if c1 == c2 {
                Ok(())
            } else {
                Err(UnificationError::ConstantClash(
                    c1.name.clone(),
                    c2.name.clone(),
                ))
            }
        }
        // Functions must have same symbol and arity
        (Term::Function(f1, args1), Term::Function(f2, args2)) => {
            if f1.name != f2.name {
                return Err(UnificationError::FunctionClash(
                    f1.name.clone(),
                    f2.name.clone(),
                ));
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
        _ => Err(UnificationError::ConstantClash(
            pattern.to_string(),
            term.to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Constant, FunctionSymbol, Variable};

    #[test]
    fn test_match_variable() {
        let x = Term::Variable(Variable {
            name: "X".to_string(),
        });
        let a = Term::Constant(Constant {
            name: "a".to_string(),
        });

        let subst = match_term(&x, &a).unwrap();
        assert_eq!(x.apply_substitution(&subst), a);
    }

    #[test]
    fn test_match_function() {
        let f = FunctionSymbol {
            name: "f".to_string(),
            arity: 2,
        };
        let x = Term::Variable(Variable {
            name: "X".to_string(),
        });
        let y = Term::Variable(Variable {
            name: "Y".to_string(),
        });
        let a = Term::Constant(Constant {
            name: "a".to_string(),
        });
        let b = Term::Constant(Constant {
            name: "b".to_string(),
        });

        let pattern = Term::Function(f.clone(), vec![x.clone(), y.clone()]);
        let term = Term::Function(f.clone(), vec![a.clone(), b.clone()]);

        let subst = match_term(&pattern, &term).unwrap();
        assert_eq!(pattern.apply_substitution(&subst), term);
    }

    #[test]
    fn test_no_match_variable_in_term() {
        let a = Term::Constant(Constant {
            name: "a".to_string(),
        });
        let x = Term::Variable(Variable {
            name: "X".to_string(),
        });

        // Should fail because we can't match constant against variable
        assert!(match_term(&a, &x).is_err());
    }

    #[test]
    fn test_no_match_inconsistent_variable() {
        // Test that mult(inv(X),X) does NOT match mult(inv(Y),mult(Y,Z))
        let mult = FunctionSymbol {
            name: "mult".to_string(),
            arity: 2,
        };
        let inv = FunctionSymbol {
            name: "inv".to_string(),
            arity: 1,
        };
        let x = Term::Variable(Variable {
            name: "X".to_string(),
        });
        let y = Term::Variable(Variable {
            name: "Y".to_string(),
        });
        let z = Term::Variable(Variable {
            name: "Z".to_string(),
        });

        let inv_x = Term::Function(inv.clone(), vec![x.clone()]);
        let pattern = Term::Function(mult.clone(), vec![inv_x, x.clone()]);

        let inv_y = Term::Function(inv.clone(), vec![y.clone()]);
        let mult_y_z = Term::Function(mult.clone(), vec![y.clone(), z.clone()]);
        let term = Term::Function(mult.clone(), vec![inv_y, mult_y_z]);

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
