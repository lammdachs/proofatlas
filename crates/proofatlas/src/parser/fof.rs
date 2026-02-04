//! First-Order Form (FOF) formula representation
//!
//! This module provides structures for representing full first-order logic
//! formulas before conversion to CNF.

use crate::fol::{Atom, Interner, Term, Variable, VariableId};
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

    /// Get all free variable IDs in the formula
    pub fn free_variable_ids(&self) -> HashSet<VariableId> {
        match self {
            FOFFormula::Atom(atom) => {
                let mut ids = HashSet::new();
                for arg in &atom.args {
                    arg.collect_variable_ids(&mut ids);
                }
                ids
            }
            FOFFormula::Not(f) => f.free_variable_ids(),
            FOFFormula::And(f1, f2)
            | FOFFormula::Or(f1, f2)
            | FOFFormula::Implies(f1, f2)
            | FOFFormula::Iff(f1, f2)
            | FOFFormula::Xor(f1, f2)
            | FOFFormula::Nand(f1, f2)
            | FOFFormula::Nor(f1, f2) => {
                let mut ids = f1.free_variable_ids();
                ids.extend(f2.free_variable_ids());
                ids
            }
            FOFFormula::Quantified(_, var, f) => {
                let mut ids = f.free_variable_ids();
                ids.remove(&var.id);
                ids
            }
        }
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
    ///
    /// Requires an interner to create new variable IDs.
    pub fn standardize_apart(self, interner: &mut Interner) -> FOFFormula {
        // Collect all variable IDs used in the formula to track existing names
        let mut used_ids = HashSet::new();
        self.collect_variable_ids(&mut used_ids);

        // Collect used names to avoid clashes when generating new names
        let mut used_names = HashSet::new();
        for id in &used_ids {
            used_names.insert(interner.resolve_variable(*id).to_string());
        }

        // Find a starting counter that won't clash
        let mut counter = 0;
        while used_names.contains(&format!("V{}", counter)) {
            counter += 1;
        }

        self.standardize_apart_with_counter(&mut counter, &HashMap::new(), &used_names, interner)
    }

    fn collect_variable_ids(&self, ids: &mut HashSet<VariableId>) {
        match self {
            FOFFormula::Atom(atom) => {
                for arg in &atom.args {
                    arg.collect_variable_ids(ids);
                }
            }
            FOFFormula::Not(f) => f.collect_variable_ids(ids),
            FOFFormula::And(f1, f2)
            | FOFFormula::Or(f1, f2)
            | FOFFormula::Implies(f1, f2)
            | FOFFormula::Iff(f1, f2)
            | FOFFormula::Xor(f1, f2)
            | FOFFormula::Nand(f1, f2)
            | FOFFormula::Nor(f1, f2) => {
                f1.collect_variable_ids(ids);
                f2.collect_variable_ids(ids);
            }
            FOFFormula::Quantified(_, var, f) => {
                ids.insert(var.id);
                f.collect_variable_ids(ids);
            }
        }
    }

    fn standardize_apart_with_counter(
        self,
        counter: &mut usize,
        renaming: &HashMap<VariableId, VariableId>,
        used_names: &HashSet<String>,
        interner: &mut Interner,
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
                FOFFormula::Not(Box::new(f.standardize_apart_with_counter(counter, renaming, used_names, interner)))
            }

            FOFFormula::And(f1, f2) => FOFFormula::And(
                Box::new(f1.standardize_apart_with_counter(counter, renaming, used_names, interner)),
                Box::new(f2.standardize_apart_with_counter(counter, renaming, used_names, interner)),
            ),

            FOFFormula::Or(f1, f2) => FOFFormula::Or(
                Box::new(f1.standardize_apart_with_counter(counter, renaming, used_names, interner)),
                Box::new(f2.standardize_apart_with_counter(counter, renaming, used_names, interner)),
            ),

            FOFFormula::Implies(f1, f2) => FOFFormula::Implies(
                Box::new(f1.standardize_apart_with_counter(counter, renaming, used_names, interner)),
                Box::new(f2.standardize_apart_with_counter(counter, renaming, used_names, interner)),
            ),

            FOFFormula::Iff(f1, f2) => FOFFormula::Iff(
                Box::new(f1.standardize_apart_with_counter(counter, renaming, used_names, interner)),
                Box::new(f2.standardize_apart_with_counter(counter, renaming, used_names, interner)),
            ),

            FOFFormula::Xor(f1, f2) => FOFFormula::Xor(
                Box::new(f1.standardize_apart_with_counter(counter, renaming, used_names, interner)),
                Box::new(f2.standardize_apart_with_counter(counter, renaming, used_names, interner)),
            ),

            FOFFormula::Nand(f1, f2) => FOFFormula::Nand(
                Box::new(f1.standardize_apart_with_counter(counter, renaming, used_names, interner)),
                Box::new(f2.standardize_apart_with_counter(counter, renaming, used_names, interner)),
            ),

            FOFFormula::Nor(f1, f2) => FOFFormula::Nor(
                Box::new(f1.standardize_apart_with_counter(counter, renaming, used_names, interner)),
                Box::new(f2.standardize_apart_with_counter(counter, renaming, used_names, interner)),
            ),

            FOFFormula::Quantified(quant, var, f) => {
                // Generate a fresh variable name that doesn't clash
                let mut new_name = format!("V{}", *counter);
                *counter += 1;
                while used_names.contains(&new_name) {
                    new_name = format!("V{}", *counter);
                    *counter += 1;
                }

                // Create new variable via interner
                let new_var = Variable::new(interner.intern_variable(&new_name));

                // Create new renaming that maps old ID to new ID
                let mut new_renaming = renaming.clone();
                new_renaming.insert(var.id, new_var.id);

                FOFFormula::Quantified(
                    quant,
                    new_var,
                    Box::new(f.standardize_apart_with_counter(counter, &new_renaming, used_names, interner)),
                )
            }
        }
    }

    fn rename_term_vars(term: Term, renaming: &HashMap<VariableId, VariableId>) -> Term {
        match term {
            Term::Variable(v) => {
                if let Some(&new_id) = renaming.get(&v.id) {
                    Term::Variable(Variable::new(new_id))
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
    use crate::fol::{Interner, PredicateSymbol};

    #[test]
    fn test_nnf_conversion() {
        let mut interner = Interner::new();

        // Test: ~(P & Q) -> ~P | ~Q
        let p_pred = PredicateSymbol::new(interner.intern_predicate("P"), 0);
        let q_pred = PredicateSymbol::new(interner.intern_predicate("Q"), 0);

        let p = FOFFormula::Atom(Atom {
            predicate: p_pred,
            args: vec![],
        });
        let q = FOFFormula::Atom(Atom {
            predicate: q_pred,
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
