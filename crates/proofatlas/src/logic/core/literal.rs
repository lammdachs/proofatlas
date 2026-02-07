//! Atoms and literals in first-order logic

use crate::logic::interner::{Interner, PredicateId};
use super::term::{Term, Variable};
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

/// An atomic formula (predicate applied to terms).
///
/// Used in FOF formula representation. For CNF/clause representation,
/// use `Literal` which includes polarity directly.
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

    /// Format this atom with an interner for name resolution
    pub fn display<'a>(&'a self, interner: &'a Interner) -> AtomDisplay<'a> {
        AtomDisplay {
            atom: self,
            interner,
        }
    }
}

/// A literal (positive or negative atomic formula)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Literal {
    pub predicate: PredicateSymbol,
    pub args: Vec<Term>,
    pub polarity: bool, // true = positive, false = negative
}

impl Literal {
    /// Create a new positive literal
    pub fn positive(predicate: PredicateSymbol, args: Vec<Term>) -> Self {
        Literal {
            predicate,
            args,
            polarity: true,
        }
    }

    /// Create a new negative literal
    pub fn negative(predicate: PredicateSymbol, args: Vec<Term>) -> Self {
        Literal {
            predicate,
            args,
            polarity: false,
        }
    }

    /// Create a literal from an Atom and polarity
    pub fn from_atom(atom: Atom, polarity: bool) -> Self {
        Literal {
            predicate: atom.predicate,
            args: atom.args,
            polarity,
        }
    }

    /// Check if this is an equality literal
    pub fn is_equality(&self, interner: &Interner) -> bool {
        interner.resolve_predicate(self.predicate.id) == "=" && self.predicate.arity == 2
    }

    /// Get the complement of this literal
    pub fn complement(&self) -> Literal {
        Literal {
            predicate: self.predicate,
            args: self.args.clone(),
            polarity: !self.polarity,
        }
    }

    /// Collect all variables in this literal
    pub fn collect_variables(&self, vars: &mut std::collections::HashSet<Variable>) {
        for term in &self.args {
            term.collect_variables(vars);
        }
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
        let pred_name = self.interner.resolve_predicate(self.literal.predicate.id);
        if pred_name == "=" && self.literal.args.len() == 2 {
            write!(
                f,
                "{} = {}",
                self.literal.args[0].display(self.interner),
                self.literal.args[1].display(self.interner)
            )
        } else {
            write!(f, "{}(", pred_name)?;
            for (i, arg) in self.literal.args.iter().enumerate() {
                if i > 0 {
                    write!(f, ",")?;
                }
                write!(f, "{}", arg.display(self.interner))?;
            }
            write!(f, ")")
        }
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
