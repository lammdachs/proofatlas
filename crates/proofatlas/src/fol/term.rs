//! Terms in first-order logic

use serde::{Deserialize, Serialize};
use std::fmt;

/// A variable in first-order logic
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Variable {
    pub name: String,
}

/// A constant symbol
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Constant {
    pub name: String,
}

/// A function symbol with arity
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FunctionSymbol {
    pub name: String,
    pub arity: usize,
}

/// A term in first-order logic
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Term {
    Variable(Variable),
    Constant(Constant),
    Function(FunctionSymbol, Vec<Term>),
}

impl Term {
    /// Get all variables in this term
    pub fn variables(&self) -> Vec<Variable> {
        match self {
            Term::Variable(v) => vec![v.clone()],
            Term::Constant(_) => vec![],
            Term::Function(_, args) => args.iter().flat_map(|arg| arg.variables()).collect(),
        }
    }

    /// Collect all variables in this term
    pub fn collect_variables(&self, vars: &mut std::collections::HashSet<Variable>) {
        match self {
            Term::Variable(v) => {
                vars.insert(v.clone());
            }
            Term::Constant(_) => {}
            Term::Function(_, args) => {
                for arg in args {
                    arg.collect_variables(vars);
                }
            }
        }
    }
}

// Display implementations for pretty printing

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl fmt::Display for Constant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl Term {
    /// Estimate the heap memory usage of this term in bytes
    pub fn memory_bytes(&self) -> usize {
        match self {
            Term::Variable(v) => v.name.capacity(),
            Term::Constant(c) => c.name.capacity(),
            Term::Function(func, args) => {
                // Function symbol name
                let name_bytes = func.name.capacity();
                // Vec overhead: capacity * size_of::<Term>
                let vec_bytes = args.capacity() * std::mem::size_of::<Term>();
                // Recursive term memory
                let args_bytes: usize = args.iter().map(|t| t.memory_bytes()).sum();
                name_bytes + vec_bytes + args_bytes
            }
        }
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::Variable(v) => write!(f, "{}", v),
            Term::Constant(c) => write!(f, "{}", c),
            Term::Function(func, args) => {
                write!(f, "{}(", func.name)?;
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
