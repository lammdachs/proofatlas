//! Conversion from FOF to CNF
//!
//! This module implements the standard algorithm for converting
//! first-order formulas to Conjunctive Normal Form (CNF).

use super::fof::{FOFFormula, Quantifier};
use crate::core::{Atom, CNFFormula, Clause, Constant, FunctionSymbol, Literal, Term, Variable};

/// Convert a FOF formula to CNF
pub fn fof_to_cnf(formula: FOFFormula) -> CNFFormula {
    let mut converter = CNFConverter::new();
    converter.convert(formula)
}

struct CNFConverter {
    skolem_counter: usize,
    universal_vars: Vec<Variable>,
}

impl CNFConverter {
    fn new() -> Self {
        CNFConverter {
            skolem_counter: 0,
            universal_vars: Vec::new(),
        }
    }

    fn convert(&mut self, formula: FOFFormula) -> CNFFormula {
        // Step 1: Convert to NNF
        let nnf = formula.to_nnf();

        // Step 2: Skolemize (remove existential quantifiers)
        let skolemized = self.skolemize(nnf);

        // Step 3: Remove universal quantifiers (they're implicit in CNF)
        let matrix = self.remove_universal_quantifiers(skolemized);

        // Step 4: Convert to CNF using distribution
        let clauses = self.distribute_to_cnf(matrix);

        CNFFormula { clauses }
    }

    fn skolemize(&mut self, formula: FOFFormula) -> FOFFormula {
        match formula {
            FOFFormula::Atom(_) | FOFFormula::Not(_) => formula,

            FOFFormula::And(f1, f2) => {
                FOFFormula::And(Box::new(self.skolemize(*f1)), Box::new(self.skolemize(*f2)))
            }

            FOFFormula::Or(f1, f2) => {
                FOFFormula::Or(Box::new(self.skolemize(*f1)), Box::new(self.skolemize(*f2)))
            }

            FOFFormula::Quantified(Quantifier::Forall, var, f) => {
                self.universal_vars.push(var.clone());
                let result =
                    FOFFormula::Quantified(Quantifier::Forall, var, Box::new(self.skolemize(*f)));
                self.universal_vars.pop();
                result
            }

            FOFFormula::Quantified(Quantifier::Exists, var, f) => {
                // Create Skolem function/constant
                let skolem_term = if self.universal_vars.is_empty() {
                    // No universal variables - create a Skolem constant
                    Term::Constant(Constant {
                        name: format!("sk{}", self.skolem_counter),
                    })
                } else {
                    // Create Skolem function with universal variables as arguments
                    Term::Function(
                        FunctionSymbol {
                            name: format!("sk{}", self.skolem_counter),
                            arity: self.universal_vars.len(),
                        },
                        self.universal_vars
                            .iter()
                            .map(|v| Term::Variable(v.clone()))
                            .collect(),
                    )
                };

                self.skolem_counter += 1;

                // Replace the existential variable with the Skolem term
                let substituted = self.substitute_in_formula(*f, &var, &skolem_term);
                self.skolemize(substituted)
            }

            // These shouldn't appear after NNF conversion
            FOFFormula::Implies(_, _)
            | FOFFormula::Iff(_, _)
            | FOFFormula::Xor(_, _)
            | FOFFormula::Nand(_, _)
            | FOFFormula::Nor(_, _) => {
                panic!("Complex connectives should be eliminated by NNF conversion")
            }
        }
    }

    fn substitute_in_formula(
        &self,
        formula: FOFFormula,
        var: &Variable,
        term: &Term,
    ) -> FOFFormula {
        match formula {
            FOFFormula::Atom(atom) => {
                let new_args = atom
                    .args
                    .iter()
                    .map(|t| self.substitute_in_term(t, var, term))
                    .collect();
                FOFFormula::Atom(Atom {
                    predicate: atom.predicate,
                    args: new_args,
                })
            }

            FOFFormula::Not(f) => {
                FOFFormula::Not(Box::new(self.substitute_in_formula(*f, var, term)))
            }

            FOFFormula::And(f1, f2) => FOFFormula::And(
                Box::new(self.substitute_in_formula(*f1, var, term)),
                Box::new(self.substitute_in_formula(*f2, var, term)),
            ),

            FOFFormula::Or(f1, f2) => FOFFormula::Or(
                Box::new(self.substitute_in_formula(*f1, var, term)),
                Box::new(self.substitute_in_formula(*f2, var, term)),
            ),

            FOFFormula::Quantified(q, v, f) => {
                if &v == var {
                    // Variable is bound here, don't substitute
                    FOFFormula::Quantified(q, v, f)
                } else {
                    FOFFormula::Quantified(
                        q,
                        v,
                        Box::new(self.substitute_in_formula(*f, var, term)),
                    )
                }
            }

            FOFFormula::Implies(_, _)
            | FOFFormula::Iff(_, _)
            | FOFFormula::Xor(_, _)
            | FOFFormula::Nand(_, _)
            | FOFFormula::Nor(_, _) => {
                panic!("Complex connectives should be eliminated")
            }
        }
    }

    fn substitute_in_term(&self, t: &Term, var: &Variable, replacement: &Term) -> Term {
        match t {
            Term::Variable(v) => {
                if v == var {
                    replacement.clone()
                } else {
                    t.clone()
                }
            }
            Term::Constant(_) => t.clone(),
            Term::Function(f, args) => {
                let new_args = args
                    .iter()
                    .map(|arg| self.substitute_in_term(arg, var, replacement))
                    .collect();
                Term::Function(f.clone(), new_args)
            }
        }
    }

    fn remove_universal_quantifiers(&self, formula: FOFFormula) -> FOFFormula {
        match formula {
            FOFFormula::Quantified(Quantifier::Forall, _, f) => {
                self.remove_universal_quantifiers(*f)
            }
            FOFFormula::And(f1, f2) => FOFFormula::And(
                Box::new(self.remove_universal_quantifiers(*f1)),
                Box::new(self.remove_universal_quantifiers(*f2)),
            ),
            FOFFormula::Or(f1, f2) => FOFFormula::Or(
                Box::new(self.remove_universal_quantifiers(*f1)),
                Box::new(self.remove_universal_quantifiers(*f2)),
            ),
            FOFFormula::Not(f) => FOFFormula::Not(Box::new(self.remove_universal_quantifiers(*f))),
            _ => formula,
        }
    }

    fn distribute_to_cnf(&self, formula: FOFFormula) -> Vec<Clause> {
        match formula {
            FOFFormula::And(f1, f2) => {
                let mut clauses = self.distribute_to_cnf(*f1);
                clauses.extend(self.distribute_to_cnf(*f2));
                clauses
            }

            FOFFormula::Or(f1, f2) => {
                // Check if we need to distribute OR over AND
                match (*f1, *f2) {
                    (FOFFormula::And(a1, a2), f2) => {
                        // (A & B) | C => (A | C) & (B | C)
                        let c1 = FOFFormula::Or(a1, Box::new(f2.clone()));
                        let c2 = FOFFormula::Or(a2, Box::new(f2));
                        let mut clauses = self.distribute_to_cnf(c1);
                        clauses.extend(self.distribute_to_cnf(c2));
                        clauses
                    }
                    (f1, FOFFormula::And(a1, a2)) => {
                        // C | (A & B) => (C | A) & (C | B)
                        let c1 = FOFFormula::Or(Box::new(f1.clone()), a1);
                        let c2 = FOFFormula::Or(Box::new(f1), a2);
                        let mut clauses = self.distribute_to_cnf(c1);
                        clauses.extend(self.distribute_to_cnf(c2));
                        clauses
                    }
                    (f1, f2) => {
                        // Normal disjunction - convert to clause
                        vec![self.formula_to_clause(FOFFormula::Or(Box::new(f1), Box::new(f2)))]
                    }
                }
            }

            FOFFormula::Atom(_) | FOFFormula::Not(_) => {
                // Single literal
                vec![self.formula_to_clause(formula)]
            }

            _ => panic!("Unexpected formula type in CNF conversion: {:?}", formula),
        }
    }

    fn formula_to_clause(&self, formula: FOFFormula) -> Clause {
        let literals = self.collect_literals(formula);
        Clause::new(literals)
    }

    fn collect_literals(&self, formula: FOFFormula) -> Vec<Literal> {
        match formula {
            FOFFormula::Or(f1, f2) => {
                let mut lits = self.collect_literals(*f1);
                lits.extend(self.collect_literals(*f2));
                lits
            }

            FOFFormula::Atom(atom) => vec![Literal::positive(atom)],

            FOFFormula::Not(f) => match *f {
                FOFFormula::Atom(atom) => vec![Literal::negative(atom)],
                _ => panic!("Negation of non-atom in CNF: {:?}", f),
            },

            _ => panic!("Non-disjunctive formula in clause: {:?}", formula),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::PredicateSymbol;

    #[test]
    fn test_simple_cnf_conversion() {
        // Test: P & Q -> two unit clauses
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

        let formula = FOFFormula::And(Box::new(p), Box::new(q));
        let cnf = fof_to_cnf(formula);

        assert_eq!(cnf.clauses.len(), 2);
        assert_eq!(cnf.clauses[0].literals.len(), 1);
        assert_eq!(cnf.clauses[1].literals.len(), 1);
    }

    #[test]
    fn test_skolemization() {
        // Test: ∃x.P(x) -> P(sk0)
        let x = Variable {
            name: "X".to_string(),
        };
        let p_x = FOFFormula::Atom(Atom {
            predicate: PredicateSymbol {
                name: "P".to_string(),
                arity: 1,
            },
            args: vec![Term::Variable(x.clone())],
        });

        let formula = FOFFormula::Quantified(Quantifier::Exists, x, Box::new(p_x));
        let cnf = fof_to_cnf(formula);

        assert_eq!(cnf.clauses.len(), 1);
        assert_eq!(cnf.clauses[0].literals.len(), 1);

        // Check that the variable was replaced with a Skolem constant
        match &cnf.clauses[0].literals[0].atom.args[0] {
            Term::Constant(c) => assert!(c.name.starts_with("sk")),
            _ => panic!("Expected Skolem constant"),
        }
    }
}
