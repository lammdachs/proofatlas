//! Variable substitutions

use super::term::{Term, Variable};
use super::literal::{Atom, Literal};
use super::clause::Clause;
use std::collections::HashMap;

/// A substitution mapping variables to terms
#[derive(Debug, Clone, Default)]
pub struct Substitution {
    pub map: HashMap<Variable, Term>,
}

impl Substitution {
    /// Create a new empty substitution
    pub fn new() -> Self {
        Substitution { map: HashMap::new() }
    }
    
    /// Add a variable -> term mapping
    pub fn insert(&mut self, var: Variable, term: Term) {
        self.map.insert(var, term);
    }
    
    /// Compose two substitutions
    pub fn compose(&self, other: &Substitution) -> Substitution {
        let mut result = Substitution::new();
        
        // Apply other to all terms in self
        for (var, term) in &self.map {
            result.insert(var.clone(), term.apply_substitution(other));
        }
        
        // Add mappings from other that aren't in self
        for (var, term) in &other.map {
            if !self.map.contains_key(var) {
                result.insert(var.clone(), term.clone());
            }
        }
        
        result
    }
}

impl Term {
    /// Apply a substitution to this term
    pub fn apply_substitution(&self, subst: &Substitution) -> Term {
        match self {
            Term::Variable(v) => {
                subst.map.get(v).cloned().unwrap_or_else(|| self.clone())
            }
            Term::Constant(_) => self.clone(),
            Term::Function(f, args) => {
                let new_args = args.iter()
                    .map(|arg| arg.apply_substitution(subst))
                    .collect();
                Term::Function(f.clone(), new_args)
            }
        }
    }
}

impl Atom {
    /// Apply a substitution to this atom
    pub fn apply_substitution(&self, subst: &Substitution) -> Atom {
        Atom {
            predicate: self.predicate.clone(),
            args: self.args.iter()
                .map(|arg| arg.apply_substitution(subst))
                .collect(),
        }
    }
}

impl Literal {
    /// Apply a substitution to this literal
    pub fn apply_substitution(&self, subst: &Substitution) -> Literal {
        Literal {
            atom: self.atom.apply_substitution(subst),
            polarity: self.polarity,
        }
    }
}

impl Clause {
    /// Apply a substitution to this clause
    pub fn apply_substitution(&self, subst: &Substitution) -> Clause {
        Clause {
            literals: self.literals.iter()
                .map(|lit| lit.apply_substitution(subst))
                .collect(),
            id: None, // New clause gets no ID
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Constant, PredicateSymbol, FunctionSymbol};
    
    #[test]
    fn test_term_substitution() {
        let x = Variable { name: "X".to_string() };
        let a = Constant { name: "a".to_string() };
        let term_x = Term::Variable(x.clone());
        let term_a = Term::Constant(a.clone());
        
        let mut subst = Substitution::new();
        subst.insert(x.clone(), term_a.clone());
        
        let result = term_x.apply_substitution(&subst);
        assert_eq!(result, term_a);
    }
}