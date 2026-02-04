//! Atoms and literals in first-order logic

use super::term::Term;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A predicate symbol with arity
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PredicateSymbol {
    pub name: String,
    pub arity: usize,
}

/// An atomic formula (predicate applied to terms)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Atom {
    pub predicate: PredicateSymbol,
    pub args: Vec<Term>,
}

/// A literal (positive or negative atom)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Literal {
    pub atom: Atom,
    pub polarity: bool, // true = positive, false = negative
}

impl Atom {
    /// Check if this is an equality atom
    pub fn is_equality(&self) -> bool {
        self.predicate.name == "=" && self.predicate.arity == 2
    }

    /// Estimate the heap memory usage of this atom in bytes
    pub fn memory_bytes(&self) -> usize {
        // Predicate name
        let pred_bytes = self.predicate.name.capacity();
        // Vec overhead: capacity * size_of::<Term>
        let vec_bytes = self.args.capacity() * std::mem::size_of::<super::Term>();
        // Term memory
        let args_bytes: usize = self.args.iter().map(|t| t.memory_bytes()).sum();
        pred_bytes + vec_bytes + args_bytes
    }
}

impl Literal {
    /// Create a new positive literal
    pub fn positive(atom: Atom) -> Self {
        Literal {
            atom,
            polarity: true,
        }
    }

    /// Create a new negative literal
    pub fn negative(atom: Atom) -> Self {
        Literal {
            atom,
            polarity: false,
        }
    }

    /// Get the complement of this literal
    pub fn complement(&self) -> Literal {
        Literal {
            atom: self.atom.clone(),
            polarity: !self.polarity,
        }
    }

    /// Collect all variables in this literal
    pub fn collect_variables(&self, vars: &mut std::collections::HashSet<super::Variable>) {
        for term in &self.atom.args {
            term.collect_variables(vars);
        }
    }

    /// Estimate the heap memory usage of this literal in bytes
    pub fn memory_bytes(&self) -> usize {
        self.atom.memory_bytes()
    }
}

// Display implementations

impl fmt::Display for Atom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_equality() && self.args.len() == 2 {
            write!(f, "{} = {}", self.args[0], self.args[1])
        } else {
            write!(f, "{}(", self.predicate.name)?;
            for (i, arg) in self.args.iter().enumerate() {
                if i > 0 {
                    write!(f, ",")?;
                }
                write!(f, "{}", arg)?;
            }
            write!(f, ")")
        }
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.polarity {
            write!(f, "~")?;
        }
        write!(f, "{}", self.atom)
    }
}
