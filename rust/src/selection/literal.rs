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

/// Select the largest negative literal (by weight)
///
/// This strategy selects the negative literal with the maximum symbol count.
/// If there are no negative literals (i.e., the clause is purely positive),
/// all literals are selected to maintain completeness.
///
/// This is a common and effective strategy because:
/// - It restricts the search space by selecting fewer literals
/// - Negative literals are often good candidates for resolution
/// - Larger literals tend to contain more information about the problem
/// - Maintains completeness through the "all positive" fallback rule
pub struct SelectLargestNegative;

impl SelectLargestNegative {
    pub fn new() -> Self {
        SelectLargestNegative
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

impl LiteralSelector for SelectLargestNegative {
    fn select(&self, clause: &Clause) -> HashSet<usize> {
        if clause.literals.is_empty() {
            return HashSet::new();
        }

        // Find all negative literals
        let negative_literals: Vec<(usize, usize)> = clause
            .literals
            .iter()
            .enumerate()
            .filter(|(_, lit)| !lit.polarity)
            .map(|(idx, lit)| (idx, self.literal_weight(lit)))
            .collect();

        // If there are no negative literals, select all (for completeness)
        if negative_literals.is_empty() {
            return (0..clause.literals.len()).collect();
        }

        // Find the maximum weight among negative literals
        let max_weight = negative_literals
            .iter()
            .map(|(_, weight)| weight)
            .max()
            .unwrap();

        // Select all negative literals with maximum weight
        negative_literals
            .iter()
            .filter(|(_, weight)| weight == max_weight)
            .map(|(idx, _)| *idx)
            .collect()
    }

    fn name(&self) -> &str {
        "SelectLargestNegative"
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

    #[test]
    fn test_select_largest_negative() {
        let p = PredicateSymbol {
            name: "P".to_string(),
            arity: 1,
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

        // ~P(X) ∨ ~Q(f(a)) ∨ P(a)
        // Should select ~Q(f(a)) because it's the largest negative literal
        let clause = Clause::new(vec![
            Literal::negative(Atom {
                predicate: p.clone(),
                args: vec![x.clone()],
            }),
            Literal::negative(Atom {
                predicate: q.clone(),
                args: vec![fa.clone()],
            }),
            Literal::positive(Atom {
                predicate: p.clone(),
                args: vec![a.clone()],
            }),
        ]);

        let selector = SelectLargestNegative::new();
        let selected = selector.select(&clause);

        // Should select only ~Q(f(a)) (index 1) as it's the largest negative
        assert_eq!(selected.len(), 1);
        assert!(selected.contains(&1));
    }

    #[test]
    fn test_select_largest_negative_all_positive() {
        let p = PredicateSymbol {
            name: "P".to_string(),
            arity: 1,
        };

        let x = Term::Variable(Variable {
            name: "X".to_string(),
        });
        let a = Term::Constant(Constant {
            name: "a".to_string(),
        });

        // P(X) ∨ P(a) - all positive, should select all for completeness
        let clause = Clause::new(vec![
            Literal::positive(Atom {
                predicate: p.clone(),
                args: vec![x.clone()],
            }),
            Literal::positive(Atom {
                predicate: p.clone(),
                args: vec![a.clone()],
            }),
        ]);

        let selector = SelectLargestNegative::new();
        let selected = selector.select(&clause);

        // Should select all literals (completeness for all-positive clauses)
        assert_eq!(selected.len(), 2);
        assert!(selected.contains(&0));
        assert!(selected.contains(&1));
    }

    #[test]
    fn test_select_largest_negative_equal_weight_negatives() {
        let p = PredicateSymbol {
            name: "P".to_string(),
            arity: 1,
        };
        let q = PredicateSymbol {
            name: "Q".to_string(),
            arity: 1,
        };

        let x = Term::Variable(Variable { name: "X".to_string() });
        let y = Term::Variable(Variable { name: "Y".to_string() });
        let a = Term::Constant(Constant { name: "a".to_string() });

        // ~P(X) ∨ ~Q(Y) ∨ R(a) - two negatives with equal weight
        let clause = Clause::new(vec![
            Literal::negative(Atom {
                predicate: p.clone(),
                args: vec![x.clone()],
            }),
            Literal::negative(Atom {
                predicate: q.clone(),
                args: vec![y.clone()],
            }),
            Literal::positive(Atom {
                predicate: p.clone(),
                args: vec![a.clone()],
            }),
        ]);

        let selector = SelectLargestNegative::new();
        let selected = selector.select(&clause);

        // Should select both negative literals (same weight)
        assert_eq!(selected.len(), 2);
        assert!(selected.contains(&0));
        assert!(selected.contains(&1));
        assert!(!selected.contains(&2)); // Should NOT select positive literal
    }

    #[test]
    fn test_select_largest_negative_subset_of_select_all() {
        // Verify that SelectLargestNegative selects a subset of SelectAll
        let p = PredicateSymbol {
            name: "P".to_string(),
            arity: 1,
        };

        let x = Term::Variable(Variable { name: "X".to_string() });
        let a = Term::Constant(Constant { name: "a".to_string() });

        // ~P(X) ∨ P(a)
        let clause = Clause::new(vec![
            Literal::negative(Atom {
                predicate: p.clone(),
                args: vec![x.clone()],
            }),
            Literal::positive(Atom {
                predicate: p.clone(),
                args: vec![a.clone()],
            }),
        ]);

        let select_all = SelectAll;
        let select_largest_neg = SelectLargestNegative::new();

        let all_selected = select_all.select(&clause);
        let neg_selected = select_largest_neg.select(&clause);

        // SelectLargestNegative should be a subset of SelectAll
        for &idx in &neg_selected {
            assert!(all_selected.contains(&idx));
        }

        // In this case, should select only the negative literal
        assert_eq!(neg_selected.len(), 1);
        assert!(neg_selected.contains(&0));
    }

    #[test]
    fn test_select_largest_negative_unit_clause() {
        let p = PredicateSymbol {
            name: "P".to_string(),
            arity: 1,
        };
        let a = Term::Constant(Constant { name: "a".to_string() });

        // Unit positive clause: P(a)
        let clause_pos = Clause::new(vec![
            Literal::positive(Atom {
                predicate: p.clone(),
                args: vec![a.clone()],
            }),
        ]);

        let selector = SelectLargestNegative::new();
        let selected = selector.select(&clause_pos);

        // Should select the only literal (all-positive rule)
        assert_eq!(selected.len(), 1);
        assert!(selected.contains(&0));

        // Unit negative clause: ~P(a)
        let clause_neg = Clause::new(vec![
            Literal::negative(Atom {
                predicate: p,
                args: vec![a],
            }),
        ]);

        let selected = selector.select(&clause_neg);

        // Should select the negative literal
        assert_eq!(selected.len(), 1);
        assert!(selected.contains(&0));
    }
}
