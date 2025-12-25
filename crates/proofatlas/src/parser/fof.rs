//! First-Order Form (FOF) formula representation
//!
//! This module provides structures for representing full first-order logic
//! formulas before conversion to CNF.

use crate::core::{Atom, Term, Variable};
use std::collections::{HashMap, HashSet};

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
    /// XOR (exclusive or)
    Xor(Box<FOFFormula>, Box<FOFFormula>),
    /// NAND (not and)
    Nand(Box<FOFFormula>, Box<FOFFormula>),
    /// NOR (not or)
    Nor(Box<FOFFormula>, Box<FOFFormula>),
    /// Quantified formula
    Quantified(Quantifier, Variable, Box<FOFFormula>),
}

impl FOFFormula {
    /// Get all free variables in the formula
    pub fn free_variables(&self) -> HashSet<Variable> {
        match self {
            FOFFormula::Atom(atom) => atom.args.iter().flat_map(|t| t.variables()).collect(),
            FOFFormula::Not(f) => f.free_variables(),
            FOFFormula::And(f1, f2)
            | FOFFormula::Or(f1, f2)
            | FOFFormula::Implies(f1, f2)
            | FOFFormula::Iff(f1, f2)
            | FOFFormula::Xor(f1, f2)
            | FOFFormula::Nand(f1, f2)
            | FOFFormula::Nor(f1, f2) => {
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

    /// Convert to negation normal form (NNF) using iterative approach to avoid stack overflow
    pub fn to_nnf(self) -> FOFFormula {
        // Pure stack-based algorithm - result stack holds intermediate FOFFormulas
        enum WorkItem {
            Process(FOFFormula, bool), // (formula, negate)
            CombineAnd,
            CombineOr,
            CombineQuantified(Quantifier, Variable),
        }

        let mut stack: Vec<WorkItem> = vec![WorkItem::Process(self, false)];
        let mut results: Vec<FOFFormula> = Vec::new();

        while let Some(item) = stack.pop() {
            match item {
                WorkItem::Process(formula, negate) => {
                    match (formula, negate) {
                        // Atom - base case
                        (FOFFormula::Atom(a), false) => {
                            results.push(FOFFormula::Atom(a));
                        }
                        (FOFFormula::Atom(a), true) => {
                            results.push(FOFFormula::Not(Box::new(FOFFormula::Atom(a))));
                        }

                        // Double negation - just flip and continue
                        (FOFFormula::Not(f), neg) => {
                            stack.push(WorkItem::Process(*f, !neg));
                        }

                        // Conjunction
                        (FOFFormula::And(f1, f2), false) => {
                            stack.push(WorkItem::CombineAnd);
                            stack.push(WorkItem::Process(*f2, false));
                            stack.push(WorkItem::Process(*f1, false));
                        }
                        (FOFFormula::And(f1, f2), true) => {
                            // De Morgan: ~(A & B) = ~A | ~B
                            stack.push(WorkItem::CombineOr);
                            stack.push(WorkItem::Process(*f2, true));
                            stack.push(WorkItem::Process(*f1, true));
                        }

                        // Disjunction
                        (FOFFormula::Or(f1, f2), false) => {
                            stack.push(WorkItem::CombineOr);
                            stack.push(WorkItem::Process(*f2, false));
                            stack.push(WorkItem::Process(*f1, false));
                        }
                        (FOFFormula::Or(f1, f2), true) => {
                            // De Morgan: ~(A | B) = ~A & ~B
                            stack.push(WorkItem::CombineAnd);
                            stack.push(WorkItem::Process(*f2, true));
                            stack.push(WorkItem::Process(*f1, true));
                        }

                        // Implication: A => B = ~A | B
                        (FOFFormula::Implies(f1, f2), false) => {
                            stack.push(WorkItem::CombineOr);
                            stack.push(WorkItem::Process(*f2, false));
                            stack.push(WorkItem::Process(*f1, true));
                        }
                        (FOFFormula::Implies(f1, f2), true) => {
                            // ~(A => B) = A & ~B
                            stack.push(WorkItem::CombineAnd);
                            stack.push(WorkItem::Process(*f2, true));
                            stack.push(WorkItem::Process(*f1, false));
                        }

                        // Biconditional: A <=> B = (~A | B) & (~B | A) = (A & B) | (~A & ~B)
                        // Using: (~A | B) & (A | ~B)
                        (FOFFormula::Iff(f1, f2), false) => {
                            let f1_clone = (*f1).clone();
                            let f2_clone = (*f2).clone();

                            // (~A | B) & (A | ~B)
                            // Structure: And(Or(~A, B), Or(A, ~B))
                            stack.push(WorkItem::CombineAnd);
                            // Second Or: (A | ~B)
                            stack.push(WorkItem::CombineOr);
                            stack.push(WorkItem::Process(f2_clone, true));
                            stack.push(WorkItem::Process(f1_clone, false));
                            // First Or: (~A | B)
                            stack.push(WorkItem::CombineOr);
                            stack.push(WorkItem::Process(*f2, false));
                            stack.push(WorkItem::Process(*f1, true));
                        }
                        (FOFFormula::Iff(f1, f2), true) => {
                            // ~(A <=> B) = (A & ~B) | (~A & B)
                            let f1_clone = (*f1).clone();
                            let f2_clone = (*f2).clone();

                            // (A & ~B) | (~A & B)
                            // Structure: Or(And(A, ~B), And(~A, B))
                            stack.push(WorkItem::CombineOr);
                            // Second And: (~A & B)
                            stack.push(WorkItem::CombineAnd);
                            stack.push(WorkItem::Process(f2_clone, false));
                            stack.push(WorkItem::Process(f1_clone, true));
                            // First And: (A & ~B)
                            stack.push(WorkItem::CombineAnd);
                            stack.push(WorkItem::Process(*f2, true));
                            stack.push(WorkItem::Process(*f1, false));
                        }

                        // XOR: A <~> B = (A & ~B) | (~A & B)
                        (FOFFormula::Xor(f1, f2), false) => {
                            let f1_clone = (*f1).clone();
                            let f2_clone = (*f2).clone();

                            // (A & ~B) | (~A & B)
                            stack.push(WorkItem::CombineOr);
                            // Second And: (~A & B)
                            stack.push(WorkItem::CombineAnd);
                            stack.push(WorkItem::Process(f2_clone, false));
                            stack.push(WorkItem::Process(f1_clone, true));
                            // First And: (A & ~B)
                            stack.push(WorkItem::CombineAnd);
                            stack.push(WorkItem::Process(*f2, true));
                            stack.push(WorkItem::Process(*f1, false));
                        }
                        (FOFFormula::Xor(f1, f2), true) => {
                            // ~(A <~> B) = (A <=> B) = (~A | B) & (A | ~B)
                            let f1_clone = (*f1).clone();
                            let f2_clone = (*f2).clone();

                            stack.push(WorkItem::CombineAnd);
                            // Second Or: (A | ~B)
                            stack.push(WorkItem::CombineOr);
                            stack.push(WorkItem::Process(f2_clone, true));
                            stack.push(WorkItem::Process(f1_clone, false));
                            // First Or: (~A | B)
                            stack.push(WorkItem::CombineOr);
                            stack.push(WorkItem::Process(*f2, false));
                            stack.push(WorkItem::Process(*f1, true));
                        }

                        // NAND: A ~& B = ~(A & B) = ~A | ~B
                        (FOFFormula::Nand(f1, f2), false) => {
                            stack.push(WorkItem::CombineOr);
                            stack.push(WorkItem::Process(*f2, true));
                            stack.push(WorkItem::Process(*f1, true));
                        }
                        (FOFFormula::Nand(f1, f2), true) => {
                            // ~(A ~& B) = A & B
                            stack.push(WorkItem::CombineAnd);
                            stack.push(WorkItem::Process(*f2, false));
                            stack.push(WorkItem::Process(*f1, false));
                        }

                        // NOR: A ~| B = ~(A | B) = ~A & ~B
                        (FOFFormula::Nor(f1, f2), false) => {
                            stack.push(WorkItem::CombineAnd);
                            stack.push(WorkItem::Process(*f2, true));
                            stack.push(WorkItem::Process(*f1, true));
                        }
                        (FOFFormula::Nor(f1, f2), true) => {
                            // ~(A ~| B) = A | B
                            stack.push(WorkItem::CombineOr);
                            stack.push(WorkItem::Process(*f2, false));
                            stack.push(WorkItem::Process(*f1, false));
                        }

                        // Quantifiers
                        (FOFFormula::Quantified(Quantifier::Forall, var, f), false) => {
                            stack.push(WorkItem::CombineQuantified(Quantifier::Forall, var));
                            stack.push(WorkItem::Process(*f, false));
                        }
                        (FOFFormula::Quantified(Quantifier::Forall, var, f), true) => {
                            // ~(∀x.P) = ∃x.~P
                            stack.push(WorkItem::CombineQuantified(Quantifier::Exists, var));
                            stack.push(WorkItem::Process(*f, true));
                        }
                        (FOFFormula::Quantified(Quantifier::Exists, var, f), false) => {
                            stack.push(WorkItem::CombineQuantified(Quantifier::Exists, var));
                            stack.push(WorkItem::Process(*f, false));
                        }
                        (FOFFormula::Quantified(Quantifier::Exists, var, f), true) => {
                            // ~(∃x.P) = ∀x.~P
                            stack.push(WorkItem::CombineQuantified(Quantifier::Forall, var));
                            stack.push(WorkItem::Process(*f, true));
                        }
                    }
                }

                WorkItem::CombineAnd => {
                    let child2 = results.pop().unwrap();
                    let child1 = results.pop().unwrap();
                    results.push(FOFFormula::And(Box::new(child1), Box::new(child2)));
                }

                WorkItem::CombineOr => {
                    let child2 = results.pop().unwrap();
                    let child1 = results.pop().unwrap();
                    results.push(FOFFormula::Or(Box::new(child1), Box::new(child2)));
                }

                WorkItem::CombineQuantified(q, var) => {
                    let child = results.pop().unwrap();
                    results.push(FOFFormula::Quantified(q, var, Box::new(child)));
                }
            }
        }

        results.pop().unwrap()
    }

    /// Standardize apart: rename all bound variables to unique names
    ///
    /// This ensures that each quantifier binds a unique variable, avoiding
    /// issues with variable capture during CNF conversion.
    pub fn standardize_apart(self) -> FOFFormula {
        // Collect all variable names used in the formula to avoid clashes
        let mut used_names = HashSet::new();
        self.collect_variable_names(&mut used_names);

        // Find a starting counter that won't clash
        let mut counter = 0;
        while used_names.contains(&format!("V{}", counter)) {
            counter += 1;
        }

        self.standardize_apart_with_counter(&mut counter, &HashMap::new(), &used_names)
    }

    fn collect_variable_names(&self, names: &mut HashSet<String>) {
        match self {
            FOFFormula::Atom(atom) => {
                for arg in &atom.args {
                    Self::collect_term_variable_names(arg, names);
                }
            }
            FOFFormula::Not(f) => f.collect_variable_names(names),
            FOFFormula::And(f1, f2)
            | FOFFormula::Or(f1, f2)
            | FOFFormula::Implies(f1, f2)
            | FOFFormula::Iff(f1, f2)
            | FOFFormula::Xor(f1, f2)
            | FOFFormula::Nand(f1, f2)
            | FOFFormula::Nor(f1, f2) => {
                f1.collect_variable_names(names);
                f2.collect_variable_names(names);
            }
            FOFFormula::Quantified(_, var, f) => {
                names.insert(var.name.clone());
                f.collect_variable_names(names);
            }
        }
    }

    fn collect_term_variable_names(term: &Term, names: &mut HashSet<String>) {
        match term {
            Term::Variable(v) => {
                names.insert(v.name.clone());
            }
            Term::Constant(_) => {}
            Term::Function(_, args) => {
                for arg in args {
                    Self::collect_term_variable_names(arg, names);
                }
            }
        }
    }

    fn standardize_apart_with_counter(
        self,
        counter: &mut usize,
        renaming: &HashMap<String, String>,
        used_names: &HashSet<String>,
    ) -> FOFFormula {
        match self {
            FOFFormula::Atom(atom) => {
                // Rename variables in the atom according to the current renaming
                let new_args: Vec<Term> = atom
                    .args
                    .into_iter()
                    .map(|t| Self::rename_term_vars(t, renaming))
                    .collect();
                FOFFormula::Atom(Atom {
                    predicate: atom.predicate,
                    args: new_args,
                })
            }

            FOFFormula::Not(f) => {
                FOFFormula::Not(Box::new(f.standardize_apart_with_counter(counter, renaming, used_names)))
            }

            FOFFormula::And(f1, f2) => FOFFormula::And(
                Box::new(f1.standardize_apart_with_counter(counter, renaming, used_names)),
                Box::new(f2.standardize_apart_with_counter(counter, renaming, used_names)),
            ),

            FOFFormula::Or(f1, f2) => FOFFormula::Or(
                Box::new(f1.standardize_apart_with_counter(counter, renaming, used_names)),
                Box::new(f2.standardize_apart_with_counter(counter, renaming, used_names)),
            ),

            FOFFormula::Implies(f1, f2) => FOFFormula::Implies(
                Box::new(f1.standardize_apart_with_counter(counter, renaming, used_names)),
                Box::new(f2.standardize_apart_with_counter(counter, renaming, used_names)),
            ),

            FOFFormula::Iff(f1, f2) => FOFFormula::Iff(
                Box::new(f1.standardize_apart_with_counter(counter, renaming, used_names)),
                Box::new(f2.standardize_apart_with_counter(counter, renaming, used_names)),
            ),

            FOFFormula::Xor(f1, f2) => FOFFormula::Xor(
                Box::new(f1.standardize_apart_with_counter(counter, renaming, used_names)),
                Box::new(f2.standardize_apart_with_counter(counter, renaming, used_names)),
            ),

            FOFFormula::Nand(f1, f2) => FOFFormula::Nand(
                Box::new(f1.standardize_apart_with_counter(counter, renaming, used_names)),
                Box::new(f2.standardize_apart_with_counter(counter, renaming, used_names)),
            ),

            FOFFormula::Nor(f1, f2) => FOFFormula::Nor(
                Box::new(f1.standardize_apart_with_counter(counter, renaming, used_names)),
                Box::new(f2.standardize_apart_with_counter(counter, renaming, used_names)),
            ),

            FOFFormula::Quantified(quant, var, f) => {
                // Generate a fresh variable name that doesn't clash
                let mut new_name = format!("V{}", *counter);
                *counter += 1;
                while used_names.contains(&new_name) {
                    new_name = format!("V{}", *counter);
                    *counter += 1;
                }

                // Create new renaming that maps old name to new name
                let mut new_renaming = renaming.clone();
                new_renaming.insert(var.name.clone(), new_name.clone());

                let new_var = Variable { name: new_name };
                FOFFormula::Quantified(
                    quant,
                    new_var,
                    Box::new(f.standardize_apart_with_counter(counter, &new_renaming, used_names)),
                )
            }
        }
    }

    fn rename_term_vars(term: Term, renaming: &HashMap<String, String>) -> Term {
        match term {
            Term::Variable(v) => {
                if let Some(new_name) = renaming.get(&v.name) {
                    Term::Variable(Variable {
                        name: new_name.clone(),
                    })
                } else {
                    Term::Variable(v)
                }
            }
            Term::Constant(c) => Term::Constant(c),
            Term::Function(f, args) => Term::Function(
                f,
                args.into_iter()
                    .map(|a| Self::rename_term_vars(a, renaming))
                    .collect(),
            ),
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
    Corollary,
    Conjecture,
    NegatedConjecture,
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
    use crate::core::PredicateSymbol;

    #[test]
    fn test_nnf_conversion() {
        // Test: ~(P & Q) -> ~P | ~Q
        let p = FOFFormula::Atom(Atom {
            predicate: PredicateSymbol {
                name: "P".to_string(),
                arity: 0,
            },
            args: vec![],
        });
        let q = FOFFormula::Atom(Atom {
            predicate: PredicateSymbol {
                name: "Q".to_string(),
                arity: 0,
            },
            args: vec![],
        });

        let formula = FOFFormula::Not(Box::new(FOFFormula::And(
            Box::new(p.clone()),
            Box::new(q.clone()),
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
