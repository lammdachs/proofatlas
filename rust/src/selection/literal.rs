//! Literal selection strategies
//!
//! These strategies determine which literals in a clause are eligible
//! for inference rules like resolution and superposition.
//!
//! In resolution-based theorem proving, not all literals need to participate
//! in every inference. By restricting which literals can be used, we can
//! dramatically reduce the number of generated clauses while still finding proofs.

use crate::core::Clause;
use std::collections::HashSet;

/// Trait for literal selection strategies
pub trait LiteralSelector: Send + Sync {
    /// Select eligible literals from a clause
    /// Returns indices of selected literals
    fn select(&self, clause: &Clause) -> HashSet<usize>;

    /// Get the name of this selection strategy
    fn name(&self) -> &str;
}

/// Select all literals - all literals are eligible for inference
///
/// With this strategy, any literal in a clause can be resolved upon.
/// For example, in the clause `P(x) ∨ Q(y) ∨ R(z)`, all three literals
/// can participate in resolution with complementary literals from other clauses.
///
/// This preserves completeness but generates many inferences.
pub struct SelectAll;

impl LiteralSelector for SelectAll {
    fn select(&self, clause: &Clause) -> HashSet<usize> {
        (0..clause.literals.len()).collect()
    }

    fn name(&self) -> &str {
        "SelectAll"
    }
}

use crate::core::{Literal, Term};

/// Select literals with maximum weight (symbol count)
///
/// Only the "largest" literals in a clause can be used for inference.
/// For example, in `P(x) ∨ Q(f(g(a)))`, only `Q(f(g(a)))` would be selected
/// because it contains more symbols.
///
/// The intuition: complex literals often contain the "meat" of the problem,
/// while simpler literals may be auxiliary. This dramatically reduces the
/// search space but is incomplete - some theorems may require resolving
/// on the smaller literals.
pub struct SelectMaxWeight;

impl SelectMaxWeight {
    pub fn new() -> Self {
        SelectMaxWeight
    }

    /// Calculate the weight of a literal (predicate + argument symbols)
    fn literal_weight(&self, literal: &Literal) -> usize {
        // Count predicate symbol + all symbols in arguments
        1 + literal
            .atom
            .args
            .iter()
            .map(|term| Self::term_symbol_count(term))
            .sum::<usize>()
    }

    /// Count the number of symbols in a term
    fn term_symbol_count(term: &Term) -> usize {
        match term {
            Term::Variable(_) => 1,
            Term::Constant(_) => 1,
            Term::Function(_, args) => {
                1 + args
                    .iter()
                    .map(|t| Self::term_symbol_count(t))
                    .sum::<usize>()
            }
        }
    }
}

impl LiteralSelector for SelectMaxWeight {
    fn select(&self, clause: &Clause) -> HashSet<usize> {
        if clause.literals.is_empty() {
            return HashSet::new();
        }

        // Calculate weight of each literal
        let weights: Vec<usize> = clause
            .literals
            .iter()
            .map(|lit| self.literal_weight(lit))
            .collect();

        // Find the maximum weight
        let max_weight = *weights.iter().max().unwrap();

        // Select all literals with maximum weight
        weights
            .iter()
            .enumerate()
            .filter(|(_, &weight)| weight == max_weight)
            .map(|(idx, _)| idx)
            .collect()
    }

    fn name(&self) -> &str {
        "SelectMaxWeight"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{
        Atom, Clause, Constant, FunctionSymbol, Literal, PredicateSymbol, Term, Variable,
    };

    #[test]
    fn test_select_max_weight() {
        let p = PredicateSymbol {
            name: "P".to_string(),
            arity: 2,
        };
        let q = PredicateSymbol {
            name: "Q".to_string(),
            arity: 1,
        };

        let x = Term::Variable(Variable {
            name: "X".to_string(),
        });
        let a = Term::Constant(Constant {
            name: "a".to_string(),
        });
        let f = FunctionSymbol {
            name: "f".to_string(),
            arity: 1,
        };
        let fa = Term::Function(f, vec![a.clone()]);

        // P(X, a) - weight 2 (X=1, a=1)
        // Q(f(a)) - weight 2 (f=1, a=1)
        let clause = Clause::new(vec![
            Literal::positive(Atom {
                predicate: p.clone(),
                args: vec![x.clone(), a.clone()],
            }),
            Literal::positive(Atom {
                predicate: q.clone(),
                args: vec![fa.clone()],
            }),
        ]);

        let selector = SelectMaxWeight::new();
        let selected = selector.select(&clause);

        // Both literals have weight 2, so both should be selected
        assert_eq!(selected.len(), 2);
        assert!(selected.contains(&0));
        assert!(selected.contains(&1));
    }
}
