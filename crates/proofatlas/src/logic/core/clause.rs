//! Clauses and CNF formulas

use crate::logic::interner::Interner;
use super::literal::Literal;
use super::term::Term;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::Hash;

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
    /// Derivation rule ID (0=input, 1=resolution, 2=factoring, 3=superposition,
    /// 4=equality_resolution, 5=equality_factoring, 6=demodulation)
    pub derivation_rule: u8,
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
            derivation_rule: 0,
        }
    }

    /// Create a new clause with a specific role
    pub fn with_role(literals: Vec<Literal>, role: ClauseRole) -> Self {
        Clause {
            literals,
            id: None,
            role,
            age: 0,
            derivation_rule: 0,
        }
    }

    /// Create a derived clause with age
    pub fn derived(literals: Vec<Literal>, age: usize) -> Self {
        Clause {
            literals,
            id: None,
            role: ClauseRole::Derived,
            age,
            derivation_rule: 0,
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
                if self.literals[i].predicate == self.literals[j].predicate
                    && self.literals[i].args == self.literals[j].args
                    && self.literals[i].polarity != self.literals[j].polarity
                {
                    return true;
                }
            }
        }

        // Check for reflexive equality
        for lit in &self.literals {
            if lit.polarity && lit.is_equality(interner) {
                if let [ref t1, ref t2] = lit.args.as_slice() {
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
                    .args
                    .iter()
                    .map(Self::term_symbol_count)
                    .sum::<usize>()
            })
            .sum()
    }

    fn term_symbol_count(term: &Term) -> usize {
        match term {
            Term::Variable(_) => 1,
            Term::Constant(_) => 1,
            Term::Function(_, args) => {
                1 + args
                    .iter()
                    .map(Self::term_symbol_count)
                    .sum::<usize>()
            }
        }
    }

    /// Maximum term nesting depth across all literals
    pub fn max_depth(&self) -> usize {
        self.literals
            .iter()
            .flat_map(|lit| lit.args.iter())
            .map(Self::term_depth)
            .max()
            .unwrap_or(0)
    }

    fn term_depth(term: &Term) -> usize {
        match term {
            Term::Variable(_) | Term::Constant(_) => 0,
            Term::Function(_, args) => {
                1 + args.iter().map(Self::term_depth).max().unwrap_or(0)
            }
        }
    }

    /// Count of distinct function, constant, and predicate symbols
    pub fn distinct_symbol_count(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        for lit in &self.literals {
            // Predicate symbol (tag with high bit to distinguish from function/constant IDs)
            seen.insert(lit.predicate.id.as_u32() as u64 | (1u64 << 32));
            for arg in &lit.args {
                Self::collect_distinct_symbols(arg, &mut seen);
            }
        }
        seen.len()
    }

    fn collect_distinct_symbols(term: &Term, seen: &mut std::collections::HashSet<u64>) {
        match term {
            Term::Variable(_) => {}
            Term::Constant(c) => { seen.insert(c.id.as_u32() as u64); }
            Term::Function(f, args) => {
                seen.insert(f.id.as_u32() as u64);
                for arg in args {
                    Self::collect_distinct_symbols(arg, seen);
                }
            }
        }
    }

    /// Total variable occurrences across all literals
    pub fn variable_count(&self) -> usize {
        self.literals
            .iter()
            .flat_map(|lit| lit.args.iter())
            .map(Self::term_variable_count)
            .sum()
    }

    fn term_variable_count(term: &Term) -> usize {
        match term {
            Term::Variable(_) => 1,
            Term::Constant(_) => 0,
            Term::Function(_, args) => {
                args.iter().map(Self::term_variable_count).sum()
            }
        }
    }

    /// Count of distinct variables
    pub fn distinct_variable_count(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        for lit in &self.literals {
            for arg in &lit.args {
                Self::collect_distinct_variables(arg, &mut seen);
            }
        }
        seen.len()
    }

    fn collect_distinct_variables(term: &Term, seen: &mut std::collections::HashSet<u32>) {
        match term {
            Term::Variable(v) => { seen.insert(v.id.as_u32()); }
            Term::Constant(_) => {}
            Term::Function(_, args) => {
                for arg in args {
                    Self::collect_distinct_variables(arg, seen);
                }
            }
        }
    }

    /// Map a rule name string to its numeric ID for ML features.
    ///
    /// Must match the mapping in `python_bindings.rs::compute_clause_features_flat`.
    pub fn rule_name_to_id(rule_name: &str) -> u8 {
        match rule_name {
            "Input" => 0,
            "Resolution" => 1,
            "Factoring" => 2,
            "Superposition" => 3,
            "EqualityResolution" => 4,
            "EqualityFactoring" => 5,
            "Demodulation" => 6,
            _ => 0, // Unknown rules default to input
        }
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

// =============================================================================
// ClauseKey - Structural hash key for clause deduplication
// =============================================================================

/// A sortable representation of a literal for use in ClauseKey.
/// Sorting order: polarity (negative first), then predicate ID, then args.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct LiteralKey {
    /// Polarity: false (negative) sorts before true (positive)
    polarity: bool,
    /// Predicate ID
    predicate_id: u32,
    /// Predicate arity
    predicate_arity: u8,
    /// Serialized arguments (for consistent ordering)
    args: Vec<TermKey>,
}

/// A sortable representation of a term for use in LiteralKey.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum TermKey {
    /// Variable with ID
    Variable(u32),
    /// Constant with ID
    Constant(u32),
    /// Function with ID, arity, and args
    Function(u32, u8, Vec<TermKey>),
}

impl TermKey {
    fn from_term(term: &Term) -> Self {
        match term {
            Term::Variable(v) => TermKey::Variable(v.id.as_u32()),
            Term::Constant(c) => TermKey::Constant(c.id.as_u32()),
            Term::Function(f, args) => {
                TermKey::Function(
                    f.id.as_u32(),
                    f.arity,
                    args.iter().map(TermKey::from_term).collect(),
                )
            }
        }
    }
}

impl LiteralKey {
    fn from_literal(lit: &Literal) -> Self {
        LiteralKey {
            polarity: lit.polarity,
            predicate_id: lit.predicate.id.as_u32(),
            predicate_arity: lit.predicate.arity,
            args: lit.args.iter().map(TermKey::from_term).collect(),
        }
    }
}

/// A structural key for a clause that enables O(1) duplicate detection.
///
/// The key sorts literals in a canonical order to ensure that logically equivalent
/// clauses (differing only in literal order) produce the same key. This replaces
/// string-based hashing (`format!("{}", clause)`) which was a major bottleneck.
///
/// Sorting order for literals: polarity (negative first), then predicate ID, then args.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ClauseKey {
    /// Sorted literal keys
    literals: Vec<LiteralKey>,
}

impl ClauseKey {
    /// Create a ClauseKey from a clause.
    ///
    /// The literals are sorted to produce a canonical representation.
    pub fn from_clause(clause: &Clause) -> Self {
        let mut literals: Vec<LiteralKey> = clause
            .literals
            .iter()
            .map(LiteralKey::from_literal)
            .collect();
        literals.sort();
        ClauseKey { literals }
    }
}
