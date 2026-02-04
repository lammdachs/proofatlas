//! Atoms and literals in first-order logic

use super::interner::{Interner, PredicateId};
use super::term::Term;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A predicate symbol with arity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PredicateSymbol {
    pub id: PredicateId,
    pub arity: u8,
}

impl PredicateSymbol {
    /// Create a new predicate symbol from an ID and arity
    pub fn new(id: PredicateId, arity: u8) -> Self {
        PredicateSymbol { id, arity }
    }

    /// Get the name of this predicate symbol from the interner
    pub fn name<'a>(&self, interner: &'a Interner) -> &'a str {
        interner.resolve_predicate(self.id)
    }
}

/// An atomic formula (predicate applied to terms)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Atom {
    pub predicate: PredicateSymbol,
    pub args: Vec<Term>,
}

impl Atom {
    /// Check if this is an equality atom
    pub fn is_equality(&self, interner: &Interner) -> bool {
        interner.resolve_predicate(self.predicate.id) == "=" && self.predicate.arity == 2
    }

    /// Estimate the heap memory usage of this atom in bytes
    pub fn memory_bytes(&self) -> usize {
        // PredicateSymbol is now Copy, no heap allocation
        // Vec overhead: capacity * size_of::<Term>
        let vec_bytes = self.args.capacity() * std::mem::size_of::<super::Term>();
        // Term memory
        let args_bytes: usize = self.args.iter().map(|t| t.memory_bytes()).sum();
        vec_bytes + args_bytes
    }

    /// Format this atom with an interner for name resolution
    pub fn display<'a>(&'a self, interner: &'a Interner) -> AtomDisplay<'a> {
        AtomDisplay {
            atom: self,
            interner,
        }
    }
}

/// A literal (positive or negative atom)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Literal {
    pub atom: Atom,
    pub polarity: bool, // true = positive, false = negative
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

    /// Format this literal with an interner for name resolution
    pub fn display<'a>(&'a self, interner: &'a Interner) -> LiteralDisplay<'a> {
        LiteralDisplay {
            literal: self,
            interner,
        }
    }
}

// Display wrappers

/// Display wrapper for Atom that includes an interner for name resolution
pub struct AtomDisplay<'a> {
    atom: &'a Atom,
    interner: &'a Interner,
}

impl<'a> fmt::Display for AtomDisplay<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pred_name = self.interner.resolve_predicate(self.atom.predicate.id);
        if pred_name == "=" && self.atom.args.len() == 2 {
            write!(
                f,
                "{} = {}",
                self.atom.args[0].display(self.interner),
                self.atom.args[1].display(self.interner)
            )
        } else {
            write!(f, "{}(", pred_name)?;
            for (i, arg) in self.atom.args.iter().enumerate() {
                if i > 0 {
                    write!(f, ",")?;
                }
                write!(f, "{}", arg.display(self.interner))?;
            }
            write!(f, ")")
        }
    }
}

/// Display wrapper for Literal that includes an interner for name resolution
pub struct LiteralDisplay<'a> {
    literal: &'a Literal,
    interner: &'a Interner,
}

impl<'a> fmt::Display for LiteralDisplay<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.literal.polarity {
            write!(f, "~")?;
        }
        write!(f, "{}", self.literal.atom.display(self.interner))
    }
}

// Display implementations that show IDs (for debugging without interner)

impl fmt::Display for Atom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "P{}(", self.predicate.id.as_u32())?;
        for (i, arg) in self.args.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{}", arg)?;
        }
        write!(f, ")")
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
