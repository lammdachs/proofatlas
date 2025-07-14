//! Clauses and CNF formulas

use super::literal::Literal;
use std::fmt;

/// A clause (disjunction of literals)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Clause {
    pub literals: Vec<Literal>,
    pub id: Option<usize>, // Optional ID for tracking
}

/// A CNF formula (conjunction of clauses)
#[derive(Debug, Clone)]
pub struct CNFFormula {
    pub clauses: Vec<Clause>,
}

impl Clause {
    /// Create a new clause from literals
    pub fn new(literals: Vec<Literal>) -> Self {
        Clause { literals, id: None }
    }
    
    /// Check if this clause is empty (contradiction)
    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }
    
    /// Check if this clause is a tautology
    pub fn is_tautology(&self) -> bool {
        // Check for complementary literals
        for i in 0..self.literals.len() {
            for j in (i + 1)..self.literals.len() {
                if self.literals[i].atom == self.literals[j].atom &&
                   self.literals[i].polarity != self.literals[j].polarity {
                    return true;
                }
            }
        }
        
        // Check for reflexive equality
        for lit in &self.literals {
            if lit.polarity && lit.atom.is_equality() {
                if let [ref t1, ref t2] = lit.atom.args.as_slice() {
                    if t1 == t2 {
                        return true;
                    }
                }
            }
        }
        
        false
    }
    
    /// Count the total number of symbols in this clause
    pub fn symbol_count(&self) -> usize {
        self.literals.iter().map(|lit| {
            // Count predicate symbol
            1 + lit.atom.args.iter().map(|t| Self::term_symbol_count(t)).sum::<usize>()
        }).sum()
    }
    
    fn term_symbol_count(term: &super::Term) -> usize {
        match term {
            super::Term::Variable(_) => 1,
            super::Term::Constant(_) => 1,
            super::Term::Function(_, args) => {
                1 + args.iter().map(|t| Self::term_symbol_count(t)).sum::<usize>()
            }
        }
    }
}

impl fmt::Display for Clause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            write!(f, "⊥")
        } else {
            for (i, lit) in self.literals.iter().enumerate() {
                if i > 0 { write!(f, " ∨ ")?; }
                write!(f, "{}", lit)?;
            }
            Ok(())
        }
    }
}