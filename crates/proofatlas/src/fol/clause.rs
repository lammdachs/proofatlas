//! Clauses and CNF formulas

use super::interner::Interner;
use super::literal::Literal;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Role of a clause in the proof (from TPTP or derived)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ClauseRole {
    /// Axiom from the problem
    #[default]
    Axiom,
    /// Hypothesis
    Hypothesis,
    /// Definition
    Definition,
    /// Negated conjecture (goal)
    NegatedConjecture,
    /// Derived clause (from inference)
    Derived,
}

impl ClauseRole {
    /// Convert to a numeric value for ML features
    pub fn to_feature_value(&self) -> f32 {
        match self {
            ClauseRole::Axiom => 0.0,
            ClauseRole::Hypothesis => 1.0,
            ClauseRole::Definition => 2.0,
            ClauseRole::NegatedConjecture => 3.0,
            ClauseRole::Derived => 4.0,
        }
    }

    /// Check if this is a goal clause (negated conjecture)
    pub fn is_goal(&self) -> bool {
        matches!(self, ClauseRole::NegatedConjecture)
    }

    /// Convert from a TPTP role string to ClauseRole
    pub fn from_tptp_role(role: &str) -> Self {
        match role {
            "axiom" | "lemma" | "theorem" | "corollary" | "assumption" => ClauseRole::Axiom,
            "hypothesis" => ClauseRole::Hypothesis,
            "definition" => ClauseRole::Definition,
            "negated_conjecture" => ClauseRole::NegatedConjecture,
            "conjecture" => ClauseRole::NegatedConjecture, // Will be negated in processing
            _ => ClauseRole::Axiom,
        }
    }
}

/// A clause (disjunction of literals)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Clause {
    pub literals: Vec<Literal>,
    pub id: Option<usize>,
    /// Role of the clause (axiom, hypothesis, negated conjecture, derived)
    pub role: ClauseRole,
    /// Age of the clause (derivation step when it was created, 0 for input clauses)
    pub age: usize,
}

/// A CNF formula (conjunction of clauses)
#[derive(Debug, Clone)]
pub struct CNFFormula {
    pub clauses: Vec<Clause>,
}

impl Clause {
    /// Create a new clause from literals
    pub fn new(literals: Vec<Literal>) -> Self {
        Clause {
            literals,
            id: None,
            role: ClauseRole::default(),
            age: 0,
        }
    }

    /// Create a new clause with a specific role
    pub fn with_role(literals: Vec<Literal>, role: ClauseRole) -> Self {
        Clause {
            literals,
            id: None,
            role,
            age: 0,
        }
    }

    /// Create a derived clause with age
    pub fn derived(literals: Vec<Literal>, age: usize) -> Self {
        Clause {
            literals,
            id: None,
            role: ClauseRole::Derived,
            age,
        }
    }

    /// Check if this clause is empty (contradiction)
    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }

    /// Check if this clause is a tautology (needs interner for equality check)
    pub fn is_tautology(&self, interner: &Interner) -> bool {
        // Check for complementary literals
        for i in 0..self.literals.len() {
            for j in (i + 1)..self.literals.len() {
                if self.literals[i].atom == self.literals[j].atom
                    && self.literals[i].polarity != self.literals[j].polarity
                {
                    return true;
                }
            }
        }

        // Check for reflexive equality
        for lit in &self.literals {
            if lit.polarity && lit.atom.is_equality(interner) {
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
        self.literals
            .iter()
            .map(|lit| {
                // Count predicate symbol
                1 + lit
                    .atom
                    .args
                    .iter()
                    .map(Self::term_symbol_count)
                    .sum::<usize>()
            })
            .sum()
    }

    fn term_symbol_count(term: &super::Term) -> usize {
        match term {
            super::Term::Variable(_) => 1,
            super::Term::Constant(_) => 1,
            super::Term::Function(_, args) => {
                1 + args
                    .iter()
                    .map(Self::term_symbol_count)
                    .sum::<usize>()
            }
        }
    }

    /// Estimate the heap memory usage of this clause in bytes
    pub fn memory_bytes(&self) -> usize {
        // Vec overhead: capacity * size_of::<Literal>
        let vec_bytes = self.literals.capacity() * std::mem::size_of::<super::Literal>();
        // Literal memory
        let lits_bytes: usize = self.literals.iter().map(|l| l.memory_bytes()).sum();
        vec_bytes + lits_bytes
    }
}

impl Clause {
    /// Format this clause with an interner for name resolution
    pub fn display<'a>(&'a self, interner: &'a Interner) -> ClauseDisplay<'a> {
        ClauseDisplay {
            clause: self,
            interner,
        }
    }
}

/// Display wrapper for Clause that includes an interner for name resolution
pub struct ClauseDisplay<'a> {
    clause: &'a Clause,
    interner: &'a Interner,
}

impl<'a> fmt::Display for ClauseDisplay<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.clause.is_empty() {
            write!(f, "⊥")
        } else {
            for (i, lit) in self.clause.literals.iter().enumerate() {
                if i > 0 {
                    write!(f, " ∨ ")?;
                }
                write!(f, "{}", lit.display(self.interner))?;
            }
            Ok(())
        }
    }
}

// Display implementation that shows IDs (for debugging without interner)
impl fmt::Display for Clause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            write!(f, "⊥")
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
