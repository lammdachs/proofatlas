//! First-Order Form (FOF) formula representation
//! 
//! This module provides structures for representing full first-order logic
//! formulas before conversion to CNF.

use crate::core::{Variable, Atom};
use std::collections::HashSet;

/// Quantifier type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Quantifier {
    Forall,
    Exists,
}

/// First-order formula
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FOFFormula {
    /// Atomic formula
    Atom(Atom),
    /// Negation
    Not(Box<FOFFormula>),
    /// Conjunction
    And(Box<FOFFormula>, Box<FOFFormula>),
    /// Disjunction
    Or(Box<FOFFormula>, Box<FOFFormula>),
    /// Implication
    Implies(Box<FOFFormula>, Box<FOFFormula>),
    /// Biconditional
    Iff(Box<FOFFormula>, Box<FOFFormula>),
    /// Quantified formula
    Quantified(Quantifier, Variable, Box<FOFFormula>),
}

impl FOFFormula {
    /// Get all free variables in the formula
    pub fn free_variables(&self) -> HashSet<Variable> {
        match self {
            FOFFormula::Atom(atom) => {
                atom.args.iter()
                    .flat_map(|t| t.variables())
                    .collect()
            }
            FOFFormula::Not(f) => f.free_variables(),
            FOFFormula::And(f1, f2) | FOFFormula::Or(f1, f2) | 
            FOFFormula::Implies(f1, f2) | FOFFormula::Iff(f1, f2) => {
                let mut vars = f1.free_variables();
                vars.extend(f2.free_variables());
                vars
            }
            FOFFormula::Quantified(_, var, f) => {
                let mut vars = f.free_variables();
                vars.remove(var);
                vars
            }
        }
    }
    
    /// Check if the formula is closed (no free variables)
    pub fn is_closed(&self) -> bool {
        self.free_variables().is_empty()
    }
    
    /// Convert to negation normal form (NNF)
    pub fn to_nnf(self) -> FOFFormula {
        self.to_nnf_impl(false)
    }
    
    fn to_nnf_impl(self, negate: bool) -> FOFFormula {
        match (self, negate) {
            // Atom
            (FOFFormula::Atom(a), false) => FOFFormula::Atom(a),
            (FOFFormula::Atom(a), true) => FOFFormula::Not(Box::new(FOFFormula::Atom(a))),
            
            // Double negation
            (FOFFormula::Not(f), false) => f.to_nnf_impl(true),
            (FOFFormula::Not(f), true) => f.to_nnf_impl(false),
            
            // Conjunction
            (FOFFormula::And(f1, f2), false) => {
                FOFFormula::And(
                    Box::new(f1.to_nnf_impl(false)),
                    Box::new(f2.to_nnf_impl(false))
                )
            }
            (FOFFormula::And(f1, f2), true) => {
                // De Morgan: ~(A & B) = ~A | ~B
                FOFFormula::Or(
                    Box::new(f1.to_nnf_impl(true)),
                    Box::new(f2.to_nnf_impl(true))
                )
            }
            
            // Disjunction
            (FOFFormula::Or(f1, f2), false) => {
                FOFFormula::Or(
                    Box::new(f1.to_nnf_impl(false)),
                    Box::new(f2.to_nnf_impl(false))
                )
            }
            (FOFFormula::Or(f1, f2), true) => {
                // De Morgan: ~(A | B) = ~A & ~B
                FOFFormula::And(
                    Box::new(f1.to_nnf_impl(true)),
                    Box::new(f2.to_nnf_impl(true))
                )
            }
            
            // Implication
            (FOFFormula::Implies(f1, f2), false) => {
                // A => B = ~A | B
                FOFFormula::Or(
                    Box::new(f1.to_nnf_impl(true)),
                    Box::new(f2.to_nnf_impl(false))
                )
            }
            (FOFFormula::Implies(f1, f2), true) => {
                // ~(A => B) = A & ~B
                FOFFormula::And(
                    Box::new(f1.to_nnf_impl(false)),
                    Box::new(f2.to_nnf_impl(true))
                )
            }
            
            // Biconditional
            (FOFFormula::Iff(f1, f2), false) => {
                // A <=> B = (A => B) & (B => A) = (~A | B) & (~B | A)
                let f1_clone = f1.clone();
                let f2_clone = f2.clone();
                FOFFormula::And(
                    Box::new(FOFFormula::Or(
                        Box::new(f1.to_nnf_impl(true)),
                        Box::new(f2.to_nnf_impl(false))
                    )),
                    Box::new(FOFFormula::Or(
                        Box::new(f2_clone.to_nnf_impl(true)),
                        Box::new(f1_clone.to_nnf_impl(false))
                    ))
                )
            }
            (FOFFormula::Iff(f1, f2), true) => {
                // ~(A <=> B) = (A & ~B) | (~A & B)
                let f1_clone = f1.clone();
                let f2_clone = f2.clone();
                FOFFormula::Or(
                    Box::new(FOFFormula::And(
                        Box::new(f1.to_nnf_impl(false)),
                        Box::new(f2.to_nnf_impl(true))
                    )),
                    Box::new(FOFFormula::And(
                        Box::new(f1_clone.to_nnf_impl(true)),
                        Box::new(f2_clone.to_nnf_impl(false))
                    ))
                )
            }
            
            // Quantifiers
            (FOFFormula::Quantified(Quantifier::Forall, var, f), false) => {
                FOFFormula::Quantified(Quantifier::Forall, var, Box::new(f.to_nnf_impl(false)))
            }
            (FOFFormula::Quantified(Quantifier::Forall, var, f), true) => {
                // ~(∀x.P) = ∃x.~P
                FOFFormula::Quantified(Quantifier::Exists, var, Box::new(f.to_nnf_impl(true)))
            }
            (FOFFormula::Quantified(Quantifier::Exists, var, f), false) => {
                FOFFormula::Quantified(Quantifier::Exists, var, Box::new(f.to_nnf_impl(false)))
            }
            (FOFFormula::Quantified(Quantifier::Exists, var, f), true) => {
                // ~(∃x.P) = ∀x.~P
                FOFFormula::Quantified(Quantifier::Forall, var, Box::new(f.to_nnf_impl(true)))
            }
        }
    }
}

/// TPTP formula role
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormulaRole {
    Axiom,
    Hypothesis,
    Definition,
    Assumption,
    Lemma,
    Theorem,
    Conjecture,
    NegatedConjecture,
    Plain,
    Type,
    Unknown,
}

/// A named FOF formula as it appears in TPTP
#[derive(Debug, Clone)]
pub struct NamedFormula {
    pub name: String,
    pub role: FormulaRole,
    pub formula: FOFFormula,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{PredicateSymbol};
    
    #[test]
    fn test_nnf_conversion() {
        // Test: ~(P & Q) -> ~P | ~Q
        let p = FOFFormula::Atom(Atom {
            predicate: PredicateSymbol { name: "P".to_string(), arity: 0 },
            args: vec![],
        });
        let q = FOFFormula::Atom(Atom {
            predicate: PredicateSymbol { name: "Q".to_string(), arity: 0 },
            args: vec![],
        });
        
        let formula = FOFFormula::Not(Box::new(FOFFormula::And(
            Box::new(p.clone()),
            Box::new(q.clone())
        )));
        
        let nnf = formula.to_nnf();
        
        // Should be: ~P | ~Q
        match nnf {
            FOFFormula::Or(f1, f2) => {
                assert!(matches!(*f1, FOFFormula::Not(_)));
                assert!(matches!(*f2, FOFFormula::Not(_)));
            }
            _ => panic!("Expected Or formula"),
        }
    }
}