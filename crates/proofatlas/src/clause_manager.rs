//! Centralized clause management: interner, literal selection, and term ordering.
//!
//! The `ClauseManager` provides a unified interface for operations that require
//! coordination between the symbol interner, literal selector, and term ordering:
//! variable renaming, equality orientation, and clause normalization.

use crate::fol::{Clause, Interner, KBOConfig, Literal, Term, TermOrdering, Variable, KBO};
use crate::selection::LiteralSelector;

/// Centralized clause management combining the symbol interner, literal
/// selection strategy, and term ordering.
pub struct ClauseManager {
    /// Symbol interner for resolving and creating symbol names
    pub interner: Interner,
    /// Literal selection strategy for inference rules
    pub literal_selector: Box<dyn LiteralSelector>,
    /// Term ordering (KBO) for equality orientation and ordering constraints
    pub term_ordering: KBO,
}

impl ClauseManager {
    /// Create a new ClauseManager with the given interner and literal selector.
    /// Uses default KBO configuration for term ordering.
    pub fn new(interner: Interner, literal_selector: Box<dyn LiteralSelector>) -> Self {
        ClauseManager {
            interner,
            literal_selector,
            term_ordering: KBO::new(KBOConfig::default()),
        }
    }

    /// Rename all variables in a clause to avoid capture.
    ///
    /// Appends `_suffix` to each variable name and interns the new name.
    /// Used before combining clauses in inference rules.
    pub fn rename_variables(&mut self, clause: &Clause, suffix: &str) -> Clause {
        Clause {
            literals: clause
                .literals
                .iter()
                .map(|lit| Literal {
                    predicate: lit.predicate,
                    args: lit
                        .args
                        .iter()
                        .map(|arg| self.rename_term_variables(arg, suffix))
                        .collect(),
                    polarity: lit.polarity,
                })
                .collect(),
            id: clause.id,
            role: clause.role,
            age: clause.age,
        }
    }

    /// Rename variables in a single term (recursive helper).
    fn rename_term_variables(&mut self, term: &Term, suffix: &str) -> Term {
        match term {
            Term::Variable(v) => {
                let old_name = self.interner.resolve_variable(v.id);
                let new_name = format!("{}_{}", old_name, suffix);
                let new_id = self.interner.intern_variable(&new_name);
                Term::Variable(Variable::new(new_id))
            }
            Term::Constant(c) => Term::Constant(*c),
            Term::Function(f, args) => Term::Function(
                *f,
                args.iter()
                    .map(|arg| self.rename_term_variables(arg, suffix))
                    .collect(),
            ),
        }
    }

    /// Orient equality literals so the larger term (by KBO) is on the left.
    ///
    /// For each equality literal `s = t`, if `t ≻ s` by the term ordering,
    /// swap arguments so the larger term comes first. This improves superposition
    /// performance by ensuring rewrites go in the right direction.
    pub fn orient_equalities(&self, clause: &mut Clause) {
        for literal in &mut clause.literals {
            if literal.is_equality(&self.interner) && literal.args.len() == 2 {
                let left = &literal.args[0];
                let right = &literal.args[1];
                match self.term_ordering.compare(left, right) {
                    TermOrdering::Less => {
                        literal.args.swap(0, 1);
                    }
                    _ => {}
                }
            }
        }
    }

    /// Normalize a clause: orient equalities and apply any future normalizations.
    pub fn normalize_clause(&self, clause: &mut Clause) {
        self.orient_equalities(clause);
    }
}
