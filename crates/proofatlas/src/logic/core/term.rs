//! Terms in first-order logic

use crate::logic::interner::{ConstantId, FunctionId, Interner, VariableId};
use serde::{Deserialize, Serialize};
use std::fmt;

/// A variable in first-order logic
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Variable {
    pub id: VariableId,
}

impl Variable {
    /// Create a new variable from an ID
    pub fn new(id: VariableId) -> Self {
        Variable { id }
    }

    /// Get the name of this variable from the interner
    pub fn name<'a>(&self, interner: &'a Interner) -> &'a str {
        interner.resolve_variable(self.id)
    }
}

/// A constant symbol
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Constant {
    pub id: ConstantId,
}

impl Constant {
    /// Create a new constant from an ID
    pub fn new(id: ConstantId) -> Self {
        Constant { id }
    }

    /// Get the name of this constant from the interner
    pub fn name<'a>(&self, interner: &'a Interner) -> &'a str {
        interner.resolve_constant(self.id)
    }
}

/// A function symbol with arity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FunctionSymbol {
    pub id: FunctionId,
    pub arity: u8,
}

impl FunctionSymbol {
    /// Create a new function symbol from an ID and arity
    pub fn new(id: FunctionId, arity: u8) -> Self {
        FunctionSymbol { id, arity }
    }

    /// Get the name of this function symbol from the interner
    pub fn name<'a>(&self, interner: &'a Interner) -> &'a str {
        interner.resolve_function(self.id)
    }
}

/// A term in first-order logic
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Term {
    Variable(Variable),
    Constant(Constant),
    Function(FunctionSymbol, Vec<Term>),
}

impl Term {
    /// Get all variable IDs in this term
    pub fn variable_ids(&self) -> Vec<VariableId> {
        match self {
            Term::Variable(v) => vec![v.id],
            Term::Constant(_) => vec![],
            Term::Function(_, args) => args.iter().flat_map(|arg| arg.variable_ids()).collect(),
        }
    }

    /// Get all variables in this term
    pub fn variables(&self) -> Vec<Variable> {
        match self {
            Term::Variable(v) => vec![*v],
            Term::Constant(_) => vec![],
            Term::Function(_, args) => args.iter().flat_map(|arg| arg.variables()).collect(),
        }
    }

    /// Collect all variable IDs in this term
    pub fn collect_variable_ids(&self, vars: &mut std::collections::HashSet<VariableId>) {
        match self {
            Term::Variable(v) => {
                vars.insert(v.id);
            }
            Term::Constant(_) => {}
            Term::Function(_, args) => {
                for arg in args {
                    arg.collect_variable_ids(vars);
                }
            }
        }
    }

    /// Format this term with an interner for name resolution
    pub fn display<'a>(&'a self, interner: &'a Interner) -> TermDisplay<'a> {
        TermDisplay {
            term: self,
            interner,
        }
    }
}

/// Display wrapper for Term that includes an interner for name resolution
pub struct TermDisplay<'a> {
    term: &'a Term,
    interner: &'a Interner,
}

impl<'a> fmt::Display for TermDisplay<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.term {
            Term::Variable(v) => write!(f, "{}", self.interner.resolve_variable(v.id)),
            Term::Constant(c) => write!(f, "{}", self.interner.resolve_constant(c.id)),
            Term::Function(func, args) => {
                write!(f, "{}", self.interner.resolve_function(func.id))?;
                if !args.is_empty() {
                    write!(f, "(")?;
                    for (i, arg) in args.iter().enumerate() {
                        if i > 0 {
                            write!(f, ",")?;
                        }
                        write!(f, "{}", arg.display(self.interner))?;
                    }
                    write!(f, ")")?;
                }
                Ok(())
            }
        }
    }
}

// Display implementations that show IDs (for debugging without interner)

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "V{}", self.id.as_u32())
    }
}

impl fmt::Display for Constant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "C{}", self.id.as_u32())
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::Variable(v) => write!(f, "{}", v),
            Term::Constant(c) => write!(f, "{}", c),
            Term::Function(func, args) => {
                write!(f, "F{}(", func.id.as_u32())?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
        }
    }
}
