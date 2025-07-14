//! Most General Unifier (MGU) computation

use crate::core::{Term, Variable, Substitution};
use std::collections::HashSet;

/// Result of a unification attempt
pub type UnificationResult = Result<Substitution, UnificationError>;

/// Errors that can occur during unification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnificationError {
    /// Occurs check failed - variable occurs in term
    OccursCheck(Variable, Term),
    /// Function symbols don't match
    FunctionClash(String, String),
    /// Arities don't match
    ArityMismatch(usize, usize),
    /// Constant symbols don't match
    ConstantClash(String, String),
}

/// Unify two terms, returning a most general unifier (MGU) if one exists
pub fn unify(term1: &Term, term2: &Term) -> UnificationResult {
    let mut subst = Substitution::new();
    unify_with_subst(term1, term2, &mut subst)?;
    Ok(subst)
}

/// Unify two terms with an existing substitution
fn unify_with_subst(term1: &Term, term2: &Term, subst: &mut Substitution) -> Result<(), UnificationError> {
    let t1 = term1.apply_substitution(subst);
    let t2 = term2.apply_substitution(subst);
    
    match (&t1, &t2) {
        // Same term - nothing to do
        _ if t1 == t2 => Ok(()),
        
        // Variable cases
        (Term::Variable(v), t) | (t, Term::Variable(v)) => {
            if occurs_check(v, t) {
                Err(UnificationError::OccursCheck(v.clone(), t.clone()))
            } else {
                // Use normalized insert to ensure all substitutions are propagated
                subst.insert_normalized(v.clone(), t.clone());
                Ok(())
            }
        }
        
        // Constant clash
        (Term::Constant(c1), Term::Constant(c2)) => {
            Err(UnificationError::ConstantClash(c1.name.clone(), c2.name.clone()))
        }
        
        // Function terms
        (Term::Function(f1, args1), Term::Function(f2, args2)) => {
            if f1.name != f2.name {
                return Err(UnificationError::FunctionClash(f1.name.clone(), f2.name.clone()));
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
        (Term::Function(f, _), Term::Constant(c)) |
        (Term::Constant(c), Term::Function(f, _)) => {
            Err(UnificationError::FunctionClash(f.name.clone(), c.name.clone()))
        }
    }
}

/// Check if variable occurs in term (occurs check)
fn occurs_check(var: &Variable, term: &Term) -> bool {
    match term {
        Term::Variable(v) => v == var,
        Term::Constant(_) => false,
        Term::Function(_, args) => {
            args.iter().any(|arg| occurs_check(var, arg))
        }
    }
}

/// Rename variables in a term to avoid conflicts
pub fn rename_variables(term: &Term, suffix: &str) -> Term {
    match term {
        Term::Variable(v) => {
            Term::Variable(Variable {
                name: format!("{}_{}", v.name, suffix),
            })
        }
        Term::Constant(c) => Term::Constant(c.clone()),
        Term::Function(f, args) => {
            Term::Function(
                f.clone(),
                args.iter().map(|arg| rename_variables(arg, suffix)).collect(),
            )
        }
    }
}

/// Get all variables in a term
pub fn variables_in_term(term: &Term) -> HashSet<Variable> {
    match term {
        Term::Variable(v) => {
            let mut set = HashSet::new();
            set.insert(v.clone());
            set
        }
        Term::Constant(_) => HashSet::new(),
        Term::Function(_, args) => {
            args.iter()
                .flat_map(|arg| variables_in_term(arg))
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Constant, FunctionSymbol};
    
    #[test]
    fn test_unify_variables() {
        let x = Term::Variable(Variable { name: "X".to_string() });
        let y = Term::Variable(Variable { name: "Y".to_string() });
        
        let result = unify(&x, &y).unwrap();
        assert_eq!(result.map.len(), 1);
    }
    
    #[test]
    fn test_unify_constant_variable() {
        let x = Term::Variable(Variable { name: "X".to_string() });
        let a = Term::Constant(Constant { name: "a".to_string() });
        
        let result = unify(&x, &a).unwrap();
        assert_eq!(result.map.len(), 1);
        
        let x_var = Variable { name: "X".to_string() };
        assert_eq!(result.map.get(&x_var), Some(&a));
    }
    
    #[test]
    fn test_unify_functions() {
        let f = FunctionSymbol { name: "f".to_string(), arity: 2 };
        let x = Term::Variable(Variable { name: "X".to_string() });
        let y = Term::Variable(Variable { name: "Y".to_string() });
        let a = Term::Constant(Constant { name: "a".to_string() });
        
        let t1 = Term::Function(f.clone(), vec![x.clone(), y.clone()]);
        let t2 = Term::Function(f.clone(), vec![a.clone(), a.clone()]);
        
        let result = unify(&t1, &t2).unwrap();
        assert_eq!(result.map.len(), 2);
    }
    
    #[test]
    fn test_occurs_check() {
        let f = FunctionSymbol { name: "f".to_string(), arity: 1 };
        let x = Variable { name: "X".to_string() };
        let x_term = Term::Variable(x.clone());
        let fx = Term::Function(f, vec![x_term.clone()]);
        
        let result = unify(&x_term, &fx);
        assert!(matches!(result, Err(UnificationError::OccursCheck(_, _))));
    }
}