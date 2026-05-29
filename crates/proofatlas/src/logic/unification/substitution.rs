//! Variable substitutions

use crate::logic::core::clause::Clause;
use crate::logic::interner::VariableId;
use crate::logic::core::literal::Literal;
use crate::logic::core::term::{Term, Variable};
use std::collections::HashMap;

/// A substitution mapping variable IDs to terms.
///
/// Supports optional trail-based backtracking for efficient subsumption checking.
/// The trail is only used when `mark()`/`backtrack()` or `bind()` are called.
#[derive(Debug, Clone, Default)]
pub struct Substitution {
    pub map: HashMap<VariableId, Term>,
    trail: Vec<VariableId>, // Empty when not using backtracking
}

impl Substitution {
    /// Create a new empty substitution
    pub fn new() -> Self {
        Substitution {
            map: HashMap::new(),
            trail: Vec::new(),
        }
    }

    /// Create a new substitution with pre-allocated capacity for backtracking
    pub fn with_capacity(var_count: usize) -> Self {
        Substitution {
            map: HashMap::with_capacity(var_count),
            trail: Vec::with_capacity(var_count * 2), // Extra space for backtracking
        }
    }

    /// Add a variable -> term mapping (without recording on trail)
    pub fn insert(&mut self, var: Variable, term: Term) {
        self.map.insert(var.id, term);
    }

    /// Add a variable ID -> term mapping (without recording on trail)
    pub fn insert_id(&mut self, var_id: VariableId, term: Term) {
        self.map.insert(var_id, term);
    }

    /// Bind variable, recording on trail for backtracking
    #[inline]
    pub fn bind(&mut self, var: Variable, term: Term) {
        self.trail.push(var.id);
        self.map.insert(var.id, term);
    }

    /// Save current position for later backtrack
    #[inline]
    pub fn mark(&self) -> usize {
        self.trail.len()
    }

    /// Undo bindings back to saved position
    #[inline]
    pub fn backtrack(&mut self, mark: usize) {
        while self.trail.len() > mark {
            if let Some(var_id) = self.trail.pop() {
                self.map.remove(&var_id);
            }
        }
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
                    trail: Vec::new(),
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

impl Literal {
    /// Apply a substitution to this literal
    pub fn apply_substitution(&self, subst: &Substitution) -> Literal {
        Literal {
            predicate: self.predicate,
            args: self
                .args
                .iter()
                .map(|arg| arg.apply_substitution(subst))
                .collect(),
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
            derivation_rule: self.derivation_rule,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::{Constant, FunctionSymbol, Interner};

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

    #[test]
    fn test_compose_applies_in_order() {
        // compose(self, other) must behave as "apply self, then other".
        let mut interner = Interner::new();
        let x = Variable::new(interner.intern_variable("X"));
        let y = Variable::new(interner.intern_variable("Y"));
        let g = interner.intern_function("g");
        let b = Term::Constant(Constant::new(interner.intern_constant("b")));
        let gx = Term::Function(FunctionSymbol::new(g, 1), vec![Term::Variable(x)]);
        let gb = Term::Function(FunctionSymbol::new(g, 1), vec![b.clone()]);

        let mut sigma = Substitution::new(); // {X -> Y}
        sigma.insert(x, Term::Variable(y));
        let mut tau = Substitution::new(); // {Y -> b}
        tau.insert(y, b.clone());

        let composed = sigma.compose(&tau);
        assert_eq!(gx.apply_substitution(&composed), gb);
        // Equivalent to applying sigma then tau by hand.
        assert_eq!(
            gx.apply_substitution(&sigma).apply_substitution(&tau),
            gb
        );
    }

    #[test]
    fn test_insert_normalized_propagates() {
        // insert_normalized must propagate a later binding into earlier ones:
        // X -> f(Y), then Y -> a, leaves X -> f(a).
        let mut interner = Interner::new();
        let x = Variable::new(interner.intern_variable("X"));
        let y = Variable::new(interner.intern_variable("Y"));
        let f = interner.intern_function("f");
        let a = Term::Constant(Constant::new(interner.intern_constant("a")));
        let fy = Term::Function(FunctionSymbol::new(f, 1), vec![Term::Variable(y)]);
        let fa = Term::Function(FunctionSymbol::new(f, 1), vec![a.clone()]);

        let mut subst = Substitution::new();
        subst.insert_normalized(x, fy);
        subst.insert_normalized(y, a.clone());
        assert_eq!(subst.get(x.id), Some(&fa), "X must be fully propagated to f(a)");
        assert_eq!(subst.get(y.id), Some(&a));
    }

    #[test]
    fn test_bind_mark_backtrack_trail() {
        // The bind/mark/backtrack trail underlies subsumption: backtrack must
        // restore exactly the bindings made after the mark, and no others.
        let mut interner = Interner::new();
        let x = Variable::new(interner.intern_variable("X"));
        let y = Variable::new(interner.intern_variable("Y"));
        let a = Term::Constant(Constant::new(interner.intern_constant("a")));
        let b = Term::Constant(Constant::new(interner.intern_constant("b")));

        let mut subst = Substitution::new();
        subst.bind(x, a.clone());
        let mark = subst.mark();
        subst.bind(y, b.clone());
        assert_eq!(subst.get(y.id), Some(&b));

        subst.backtrack(mark);
        assert_eq!(subst.get(y.id), None, "binding after the mark is undone");
        assert_eq!(subst.get(x.id), Some(&a), "binding before the mark survives");
    }
}
