//! Variable substitutions

use super::clause::Clause;
use super::interner::VariableId;
use super::literal::{Atom, Literal};
use super::term::{Term, Variable};
use std::collections::HashMap;

/// A substitution mapping variable IDs to terms
#[derive(Debug, Clone, Default)]
pub struct Substitution {
    pub map: HashMap<VariableId, Term>,
}

impl Substitution {
    /// Create a new empty substitution
    pub fn new() -> Self {
        Substitution {
            map: HashMap::new(),
        }
    }

    /// Add a variable -> term mapping
    pub fn insert(&mut self, var: Variable, term: Term) {
        self.map.insert(var.id, term);
    }

    /// Add a variable ID -> term mapping
    pub fn insert_id(&mut self, var_id: VariableId, term: Term) {
        self.map.insert(var_id, term);
    }

    /// Add a variable -> term mapping with eager substitution propagation
    /// This ensures all variables in the substitution are fully substituted
    pub fn insert_normalized(&mut self, var: Variable, term: Term) {
        let var_id = var.id;

        // First, apply existing substitutions to the new term
        let normalized_term = term.apply_substitution(self);

        // Insert the new mapping
        self.map.insert(var_id, normalized_term);

        // Now apply the new substitution to all existing mappings
        let mut updated_map = HashMap::new();
        for (&existing_var_id, existing_term) in self.map.iter() {
            if existing_var_id != var_id {
                let single_subst = Substitution {
                    map: [(var_id, self.map[&var_id].clone())]
                        .iter()
                        .cloned()
                        .collect(),
                };
                updated_map.insert(existing_var_id, existing_term.apply_substitution(&single_subst));
            } else {
                updated_map.insert(existing_var_id, existing_term.clone());
            }
        }
        self.map = updated_map;
    }

    /// Compose two substitutions
    pub fn compose(&self, other: &Substitution) -> Substitution {
        let mut result = Substitution::new();

        // Apply other to all terms in self
        for (&var_id, term) in &self.map {
            result.insert_id(var_id, term.apply_substitution(other));
        }

        // Add mappings from other that aren't in self
        for (&var_id, term) in &other.map {
            if !self.map.contains_key(&var_id) {
                result.insert_id(var_id, term.clone());
            }
        }

        result
    }

    /// Get the term for a variable ID, if bound
    pub fn get(&self, var_id: VariableId) -> Option<&Term> {
        self.map.get(&var_id)
    }

    /// Check if a variable ID is bound
    pub fn contains(&self, var_id: VariableId) -> bool {
        self.map.contains_key(&var_id)
    }
}

impl Term {
    /// Apply a substitution to this term
    pub fn apply_substitution(&self, subst: &Substitution) -> Term {
        match self {
            Term::Variable(v) => subst.map.get(&v.id).cloned().unwrap_or_else(|| self.clone()),
            Term::Constant(_) => self.clone(),
            Term::Function(f, args) => {
                let new_args = args
                    .iter()
                    .map(|arg| arg.apply_substitution(subst))
                    .collect();
                Term::Function(*f, new_args)
            }
        }
    }
}

impl Atom {
    /// Apply a substitution to this atom
    pub fn apply_substitution(&self, subst: &Substitution) -> Atom {
        Atom {
            predicate: self.predicate,
            args: self
                .args
                .iter()
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
            literals: self
                .literals
                .iter()
                .map(|lit| lit.apply_substitution(subst))
                .collect(),
            id: None, // New clause gets no ID
            role: self.role,
            age: self.age,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fol::{Constant, Interner};

    #[test]
    fn test_term_substitution() {
        let mut interner = Interner::new();
        let x_id = interner.intern_variable("X");
        let a_id = interner.intern_constant("a");

        let x = Variable::new(x_id);
        let a = Constant::new(a_id);
        let term_x = Term::Variable(x);
        let term_a = Term::Constant(a);

        let mut subst = Substitution::new();
        subst.insert(x, term_a.clone());

        let result = term_x.apply_substitution(&subst);
        assert_eq!(result, term_a);
    }

    #[test]
    fn test_substitution_lookup() {
        let mut interner = Interner::new();
        let x_id = interner.intern_variable("X");
        let y_id = interner.intern_variable("Y");
        let a_id = interner.intern_constant("a");

        let x = Variable::new(x_id);
        let a = Constant::new(a_id);
        let term_a = Term::Constant(a);

        let mut subst = Substitution::new();
        subst.insert(x, term_a.clone());

        assert!(subst.contains(x_id));
        assert!(!subst.contains(y_id));
        assert_eq!(subst.get(x_id), Some(&term_a));
        assert_eq!(subst.get(y_id), None);
    }
}
