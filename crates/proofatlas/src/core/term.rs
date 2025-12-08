//! Terms in first-order logic

use std::fmt;

/// A variable in first-order logic
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Variable {
    pub name: String,
}

/// A constant symbol
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Constant {
    pub name: String,
}

/// A function symbol with arity
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionSymbol {
    pub name: String,
    pub arity: usize,
}

/// A term in first-order logic
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
