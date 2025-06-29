//! Core first-order logic types
//! 
//! This module corresponds to Python's proofatlas.core.logic

use serde::{Serialize, Deserialize};
use std::fmt;
use std::collections::HashSet;
use indexmap::IndexSet;

/// Represents a term in first-order logic
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Term {
    Variable(String),
    Constant(String),
    Function {
        name: String,
        args: Vec<Term>,
    },
}

impl Term {
    pub fn is_variable(&self) -> bool {
        matches!(self, Term::Variable(_))
    }
    
    pub fn is_constant(&self) -> bool {
        matches!(self, Term::Constant(_))
    }
    
    pub fn is_function(&self) -> bool {
        matches!(self, Term::Function { .. })
    }
    
    /// Get all variables in this term
    pub fn variables(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        self.collect_variables(&mut vars);
        vars
    }
    
    fn collect_variables(&self, vars: &mut HashSet<String>) {
        match self {
            Term::Variable(name) => {
                vars.insert(name.clone());
            }
            Term::Function { args, .. } => {
                for arg in args {
                    arg.collect_variables(vars);
                }
            }
            _ => {}
        }
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::Variable(name) => write!(f, "{}", name),
            Term::Constant(name) => write!(f, "{}", name),
            Term::Function { name, args } => {
                write!(f, "{}", name)?;
                if !args.is_empty() {
                    write!(f, "(")?;
                    for (i, arg) in args.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", arg)?;
                    }
                    write!(f, ")")?;
                }
                Ok(())
            }
        }
    }
}

/// Represents a predicate application
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Predicate {
    pub name: String,
    pub args: Vec<Term>,
}

impl Predicate {
    pub fn new(name: String, args: Vec<Term>) -> Self {
        Predicate { name, args }
    }
    
    pub fn arity(&self) -> usize {
        self.args.len()
    }
}

impl fmt::Display for Predicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Special case for equality
        if self.name == "=" && self.args.len() == 2 {
            write!(f, "{} = {}", self.args[0], self.args[1])
        } else if self.args.is_empty() {
            write!(f, "{}", self.name)
        } else {
            write!(f, "{}(", self.name)?;
            for (i, arg) in self.args.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", arg)?;
            }
            write!(f, ")")
        }
    }
}

/// Represents a literal (positive or negative predicate)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Literal {
    pub polarity: bool,
    pub predicate: Predicate,
}

impl Literal {
    pub fn new(polarity: bool, predicate: Predicate) -> Self {
        Literal { polarity, predicate }
    }
    
    pub fn positive(predicate: Predicate) -> Self {
        Literal::new(true, predicate)
    }
    
    pub fn negative(predicate: Predicate) -> Self {
        Literal::new(false, predicate)
    }
    
    pub fn negate(&self) -> Self {
        Literal {
            polarity: !self.polarity,
            predicate: self.predicate.clone(),
        }
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.polarity {
            write!(f, "~")?;
        }
        write!(f, "{}", self.predicate)
    }
}

/// Represents a clause (disjunction of literals)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Clause {
    pub literals: Vec<Literal>,
}

impl Clause {
    pub fn new(literals: Vec<Literal>) -> Self {
        Clause { literals }
    }
    
    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }
    
    pub fn is_unit(&self) -> bool {
        self.literals.len() == 1
    }
    
    pub fn is_tautology(&self) -> bool {
        // Check if clause contains P and ~P
        for (i, lit1) in self.literals.iter().enumerate() {
            for lit2 in self.literals.iter().skip(i + 1) {
                if lit1.polarity != lit2.polarity && lit1.predicate == lit2.predicate {
                    return true;
                }
            }
        }
        false
    }
}

impl fmt::Display for Clause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            write!(f, "⊥")  // Empty clause (false)
        } else {
            for (i, lit) in self.literals.iter().enumerate() {
                if i > 0 {
                    write!(f, " ∨ ")?;
                }
                write!(f, "{}", lit)?;
            }
            Ok(())
        }
    }
}

/// Represents a problem (set of clauses)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Problem {
    pub clauses: Vec<Clause>,
    pub conjecture_indices: IndexSet<usize>,
}

impl Problem {
    pub fn new(clauses: Vec<Clause>) -> Self {
        Problem {
            clauses,
            conjecture_indices: IndexSet::new(),
        }
    }
    
    pub fn with_conjectures(clauses: Vec<Clause>, conjecture_indices: Vec<usize>) -> Self {
        Problem {
            clauses,
            conjecture_indices: conjecture_indices.into_iter().collect(),
        }
    }
    
    pub fn count_literals(&self) -> usize {
        self.clauses.iter().map(|c| c.literals.len()).sum()
    }
    
    pub fn is_conjecture_clause(&self, index: usize) -> bool {
        self.conjecture_indices.contains(&index)
    }
    
    pub fn get_conjecture_clauses(&self) -> Vec<(usize, &Clause)> {
        self.conjecture_indices
            .iter()
            .filter_map(|&idx| {
                self.clauses.get(idx).map(|clause| (idx, clause))
            })
            .collect()
    }
    
    /// Serialize the problem to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
    
    /// Deserialize a problem from JSON string
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl fmt::Display for Problem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, clause) in self.clauses.iter().enumerate() {
            if self.is_conjecture_clause(i) {
                write!(f, "[CONJ] ")?;
            }
            writeln!(f, "{}: {}", i + 1, clause)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_term_creation() {
        let var = Term::Variable("X".to_string());
        assert!(var.is_variable());
        
        let const_term = Term::Constant("a".to_string());
        assert!(const_term.is_constant());
        
        let func = Term::Function {
            name: "f".to_string(),
            args: vec![var.clone(), const_term.clone()],
        };
        assert!(func.is_function());
    }
    
    #[test]
    fn test_literal_negation() {
        let pred = Predicate::new("P".to_string(), vec![]);
        let lit = Literal::positive(pred);
        assert_eq!(lit.polarity, true);
        
        let neg_lit = lit.negate();
        assert_eq!(neg_lit.polarity, false);
    }
    
    #[test]
    fn test_clause_tautology() {
        let pred = Predicate::new("P".to_string(), vec![]);
        let pos_lit = Literal::positive(pred.clone());
        let neg_lit = Literal::negative(pred);
        
        let taut_clause = Clause::new(vec![pos_lit, neg_lit]);
        assert!(taut_clause.is_tautology());
        
        let normal_clause = Clause::new(vec![
            Literal::positive(Predicate::new("P".to_string(), vec![])),
            Literal::positive(Predicate::new("Q".to_string(), vec![])),
        ]);
        assert!(!normal_clause.is_tautology());
    }
}