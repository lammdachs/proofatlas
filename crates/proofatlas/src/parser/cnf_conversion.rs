//! Conversion from FOF to CNF
//!
//! This module implements the standard algorithm for converting
//! first-order formulas to Conjunctive Normal Form (CNF).

use std::time::Instant;

use super::fof::{FOFFormula, Quantifier};
use crate::core::{
    Atom, CNFFormula, Clause, ClauseRole, Constant, FunctionSymbol, Literal, PredicateSymbol, Term,
    Variable,
};

/// Error during CNF conversion
#[derive(Debug, Clone)]
pub enum CNFConversionError {
    Timeout,
}

impl std::fmt::Display for CNFConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CNFConversionError::Timeout => write!(f, "CNF conversion timed out"),
        }
    }
}

/// Convert a FOF formula to CNF
pub fn fof_to_cnf(formula: FOFFormula) -> Result<CNFFormula, CNFConversionError> {
    fof_to_cnf_with_role(formula, ClauseRole::Axiom, None)
}

/// Convert a FOF formula to CNF with a specific role and optional timeout
pub fn fof_to_cnf_with_role(
    formula: FOFFormula,
    role: ClauseRole,
    timeout: Option<Instant>,
) -> Result<CNFFormula, CNFConversionError> {
    let mut converter = CNFConverter::new(role, timeout);
    converter.convert(formula)
}

struct CNFConverter {
    skolem_counter: usize,
    def_counter: usize,
    universal_vars: Vec<Variable>,
    role: ClauseRole,
    timeout: Option<Instant>,
}

impl CNFConverter {
    fn new(role: ClauseRole, timeout: Option<Instant>) -> Self {
        CNFConverter {
            skolem_counter: 0,
            def_counter: 0,
            universal_vars: Vec::new(),
            role,
            timeout,
        }
    }

    fn check_timeout(&self) -> Result<(), CNFConversionError> {
        if let Some(timeout) = self.timeout {
            if Instant::now() >= timeout {
                return Err(CNFConversionError::Timeout);
            }
        }
        Ok(())
    }

    fn convert(&mut self, formula: FOFFormula) -> Result<CNFFormula, CNFConversionError> {
        // Step 0: Apply definitional CNF to biconditionals with quantifiers
        // When A <=> B contains quantified subformulas, NNF expansion would duplicate them.
        // Instead, we replace the biconditional with a definition predicate and add
        // polarity-appropriate definition clauses.
        let mut definitions = Vec::new();
        let transformed = self.definitional_transform(formula, true, &mut definitions);

        // Combine with definitions
        let combined = definitions.into_iter().fold(transformed, |acc, def| {
            FOFFormula::And(Box::new(acc), Box::new(def))
        });

        // Step 1: Convert to NNF
        let nnf = combined.to_nnf();

        // Step 2: Skolemize (remove existential quantifiers)
        let skolemized = self.skolemize(nnf);

        // Step 3: Remove universal quantifiers (they're implicit in CNF)
        let matrix = self.remove_universal_quantifiers(skolemized);

        // Step 4: Convert to CNF using distribution
        let clauses = self.distribute_to_cnf(matrix)?;

        Ok(CNFFormula { clauses })
    }

    /// Apply definitional transformation to biconditionals containing quantifiers.
    ///
    /// Tracks polarity to generate the correct one-sided definitions:
    /// - Positive occurrence: D => (A <=> B), i.e., (D & A) => B and (D & B) => A
    /// - Negative occurrence: (A <=> B) => D, i.e., (A & B) => D and (~A & ~B) => D
    fn definitional_transform(
        &mut self,
        formula: FOFFormula,
        positive: bool, // true = positive polarity, false = negative
        definitions: &mut Vec<FOFFormula>,
    ) -> FOFFormula {
        match formula {
            FOFFormula::Atom(_) => formula,

            FOFFormula::Not(f) => {
                // Negation flips polarity
                FOFFormula::Not(Box::new(self.definitional_transform(*f, !positive, definitions)))
            }

            FOFFormula::And(f1, f2) => FOFFormula::And(
                Box::new(self.definitional_transform(*f1, positive, definitions)),
                Box::new(self.definitional_transform(*f2, positive, definitions)),
            ),

            FOFFormula::Or(f1, f2) => FOFFormula::Or(
                Box::new(self.definitional_transform(*f1, positive, definitions)),
                Box::new(self.definitional_transform(*f2, positive, definitions)),
            ),

            FOFFormula::Implies(f1, f2) => {
                // A => B: A has flipped polarity, B keeps polarity
                FOFFormula::Implies(
                    Box::new(self.definitional_transform(*f1, !positive, definitions)),
                    Box::new(self.definitional_transform(*f2, positive, definitions)),
                )
            }

            FOFFormula::Iff(f1, f2) => {
                // First, recursively transform subformulas (both polarities for biconditional)
                let f1 = self.definitional_transform(*f1, positive, definitions);
                let f2 = self.definitional_transform(*f2, positive, definitions);

                // Check if either side contains quantifiers
                if self.contains_quantifier(&f1) || self.contains_quantifier(&f2) {
                    self.create_iff_definition(f1, f2, positive, definitions)
                } else {
                    FOFFormula::Iff(Box::new(f1), Box::new(f2))
                }
            }

            FOFFormula::Xor(f1, f2) => {
                // XOR is ~(A <=> B), so flip polarity for the inner biconditional
                let f1 = self.definitional_transform(*f1, positive, definitions);
                let f2 = self.definitional_transform(*f2, positive, definitions);

                if self.contains_quantifier(&f1) || self.contains_quantifier(&f2) {
                    // Create definition for (A <=> B) with flipped polarity, then negate
                    let def_atom = self.create_iff_definition(f1, f2, !positive, definitions);
                    FOFFormula::Not(Box::new(def_atom))
                } else {
                    FOFFormula::Xor(Box::new(f1), Box::new(f2))
                }
            }

            FOFFormula::Nand(f1, f2) => FOFFormula::Nand(
                Box::new(self.definitional_transform(*f1, !positive, definitions)),
                Box::new(self.definitional_transform(*f2, !positive, definitions)),
            ),

            FOFFormula::Nor(f1, f2) => FOFFormula::Nor(
                Box::new(self.definitional_transform(*f1, !positive, definitions)),
                Box::new(self.definitional_transform(*f2, !positive, definitions)),
            ),

            FOFFormula::Quantified(q, var, f) => FOFFormula::Quantified(
                q,
                var,
                Box::new(self.definitional_transform(*f, positive, definitions)),
            ),
        }
    }

    /// Check if a formula contains any quantifiers
    fn contains_quantifier(&self, formula: &FOFFormula) -> bool {
        match formula {
            FOFFormula::Atom(_) => false,
            FOFFormula::Not(f) => self.contains_quantifier(f),
            FOFFormula::And(f1, f2)
            | FOFFormula::Or(f1, f2)
            | FOFFormula::Implies(f1, f2)
            | FOFFormula::Iff(f1, f2)
            | FOFFormula::Xor(f1, f2)
            | FOFFormula::Nand(f1, f2)
            | FOFFormula::Nor(f1, f2) => {
                self.contains_quantifier(f1) || self.contains_quantifier(f2)
            }
            FOFFormula::Quantified(_, _, _) => true,
        }
    }

    /// Create a definition for A <=> B based on polarity.
    ///
    /// - Positive: D => (A <=> B), i.e., (D & A) => B and (D & B) => A
    /// - Negative: (A <=> B) => D, i.e., (A & B) => D and (~A & ~B) => D
    fn create_iff_definition(
        &mut self,
        a: FOFFormula,
        b: FOFFormula,
        positive: bool,
        definitions: &mut Vec<FOFFormula>,
    ) -> FOFFormula {
        // Collect free variables from both sides
        let mut free_vars: Vec<Variable> = a.free_variables().into_iter().collect();
        for v in b.free_variables() {
            if !free_vars.contains(&v) {
                free_vars.push(v);
            }
        }
        free_vars.sort_by(|x, y| x.name.cmp(&y.name));

        // Create definition predicate D
        let def_name = format!("def{}", self.def_counter);
        self.def_counter += 1;

        let def_pred = PredicateSymbol {
            name: def_name,
            arity: free_vars.len(),
        };

        let def_args: Vec<Term> = free_vars.iter().map(|v| Term::Variable(v.clone())).collect();
        let def_atom = FOFFormula::Atom(Atom {
            predicate: def_pred,
            args: def_args,
        });

        if positive {
            // D => (A <=> B): (D & A) => B and (D & B) => A
            // As implications: D => (A => B) and D => (B => A)
            let def1 = FOFFormula::Implies(
                Box::new(def_atom.clone()),
                Box::new(FOFFormula::Implies(Box::new(a.clone()), Box::new(b.clone()))),
            );
            let def2 = FOFFormula::Implies(
                Box::new(def_atom.clone()),
                Box::new(FOFFormula::Implies(Box::new(b), Box::new(a))),
            );
            definitions.push(self.wrap_with_forall(def1, &free_vars));
            definitions.push(self.wrap_with_forall(def2, &free_vars));
        } else {
            // (A <=> B) => D: (A & B) => D and (~A & ~B) => D
            let def1 = FOFFormula::Implies(
                Box::new(FOFFormula::And(Box::new(a.clone()), Box::new(b.clone()))),
                Box::new(def_atom.clone()),
            );
            let def2 = FOFFormula::Implies(
                Box::new(FOFFormula::And(
                    Box::new(FOFFormula::Not(Box::new(a))),
                    Box::new(FOFFormula::Not(Box::new(b))),
                )),
                Box::new(def_atom.clone()),
            );
            definitions.push(self.wrap_with_forall(def1, &free_vars));
            definitions.push(self.wrap_with_forall(def2, &free_vars));
        }

        def_atom
    }

    /// Wrap a formula with universal quantifiers for the given variables
    fn wrap_with_forall(&self, formula: FOFFormula, vars: &[Variable]) -> FOFFormula {
        vars.iter().rev().fold(formula, |f, v| {
            FOFFormula::Quantified(Quantifier::Forall, v.clone(), Box::new(f))
        })
    }

    fn skolemize(&mut self, formula: FOFFormula) -> FOFFormula {
        // Iterative skolemization using explicit stack
        // The tricky part is managing universal_vars scope correctly
        enum WorkItem {
            Process(FOFFormula),
            CombineAnd,
            CombineOr,
            CombineForall(Variable), // After processing body, pop var and wrap result
        }

        let mut stack: Vec<WorkItem> = vec![WorkItem::Process(formula)];
        let mut results: Vec<FOFFormula> = Vec::new();

        while let Some(item) = stack.pop() {
            match item {
                WorkItem::Process(f) => match f {
                    FOFFormula::Atom(_) | FOFFormula::Not(_) => {
                        results.push(f);
                    }

                    FOFFormula::And(f1, f2) => {
                        stack.push(WorkItem::CombineAnd);
                        stack.push(WorkItem::Process(*f2));
                        stack.push(WorkItem::Process(*f1));
                    }

                    FOFFormula::Or(f1, f2) => {
                        stack.push(WorkItem::CombineOr);
                        stack.push(WorkItem::Process(*f2));
                        stack.push(WorkItem::Process(*f1));
                    }

                    FOFFormula::Quantified(Quantifier::Forall, var, f) => {
                        // Push var to universal_vars scope
                        self.universal_vars.push(var.clone());
                        // When body is done, wrap it in Forall and pop the var
                        stack.push(WorkItem::CombineForall(var));
                        stack.push(WorkItem::Process(*f));
                    }

                    FOFFormula::Quantified(Quantifier::Exists, var, f) => {
                        // Create Skolem function/constant
                        let skolem_term = if self.universal_vars.is_empty() {
                            Term::Constant(Constant {
                                name: format!("sk{}", self.skolem_counter),
                            })
                        } else {
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
                        // Continue processing the substituted formula (existential is eliminated)
                        stack.push(WorkItem::Process(substituted));
                    }

                    FOFFormula::Implies(_, _)
                    | FOFFormula::Iff(_, _)
                    | FOFFormula::Xor(_, _)
                    | FOFFormula::Nand(_, _)
                    | FOFFormula::Nor(_, _) => {
                        panic!("Complex connectives should be eliminated by NNF conversion")
                    }
                },

                WorkItem::CombineAnd => {
                    let right = results.pop().unwrap();
                    let left = results.pop().unwrap();
                    results.push(FOFFormula::And(Box::new(left), Box::new(right)));
                }

                WorkItem::CombineOr => {
                    let right = results.pop().unwrap();
                    let left = results.pop().unwrap();
                    results.push(FOFFormula::Or(Box::new(left), Box::new(right)));
                }

                WorkItem::CombineForall(var) => {
                    self.universal_vars.pop();
                    let body = results.pop().unwrap();
                    results.push(FOFFormula::Quantified(
                        Quantifier::Forall,
                        var,
                        Box::new(body),
                    ));
                }
            }
        }

        results.pop().unwrap()
    }

    fn substitute_in_formula(
        &self,
        formula: FOFFormula,
        var: &Variable,
        term: &Term,
    ) -> FOFFormula {
        // Iterative substitution using explicit stack
        enum WorkItem {
            Process(FOFFormula),
            CombineNot,
            CombineAnd,
            CombineOr,
            CombineImplies,
            CombineIff,
            CombineXor,
            CombineNand,
            CombineNor,
            CombineQuantified(Quantifier, Variable),
        }

        let mut stack: Vec<WorkItem> = vec![WorkItem::Process(formula)];
        let mut results: Vec<FOFFormula> = Vec::new();

        while let Some(item) = stack.pop() {
            match item {
                WorkItem::Process(f) => match f {
                    FOFFormula::Atom(atom) => {
                        let new_args = atom
                            .args
                            .iter()
                            .map(|t| self.substitute_in_term(t, var, term))
                            .collect();
                        results.push(FOFFormula::Atom(Atom {
                            predicate: atom.predicate,
                            args: new_args,
                        }));
                    }

                    FOFFormula::Not(f) => {
                        stack.push(WorkItem::CombineNot);
                        stack.push(WorkItem::Process(*f));
                    }

                    FOFFormula::And(f1, f2) => {
                        stack.push(WorkItem::CombineAnd);
                        stack.push(WorkItem::Process(*f2));
                        stack.push(WorkItem::Process(*f1));
                    }

                    FOFFormula::Or(f1, f2) => {
                        stack.push(WorkItem::CombineOr);
                        stack.push(WorkItem::Process(*f2));
                        stack.push(WorkItem::Process(*f1));
                    }

                    FOFFormula::Quantified(q, v, f) => {
                        if &v == var {
                            // Variable is bound here, don't substitute - keep formula as-is
                            results.push(FOFFormula::Quantified(q, v, f));
                        } else {
                            stack.push(WorkItem::CombineQuantified(q, v));
                            stack.push(WorkItem::Process(*f));
                        }
                    }

                    FOFFormula::Implies(f1, f2) => {
                        stack.push(WorkItem::CombineImplies);
                        stack.push(WorkItem::Process(*f2));
                        stack.push(WorkItem::Process(*f1));
                    }

                    FOFFormula::Iff(f1, f2) => {
                        stack.push(WorkItem::CombineIff);
                        stack.push(WorkItem::Process(*f2));
                        stack.push(WorkItem::Process(*f1));
                    }

                    FOFFormula::Xor(f1, f2) => {
                        stack.push(WorkItem::CombineXor);
                        stack.push(WorkItem::Process(*f2));
                        stack.push(WorkItem::Process(*f1));
                    }

                    FOFFormula::Nand(f1, f2) => {
                        stack.push(WorkItem::CombineNand);
                        stack.push(WorkItem::Process(*f2));
                        stack.push(WorkItem::Process(*f1));
                    }

                    FOFFormula::Nor(f1, f2) => {
                        stack.push(WorkItem::CombineNor);
                        stack.push(WorkItem::Process(*f2));
                        stack.push(WorkItem::Process(*f1));
                    }
                },

                WorkItem::CombineNot => {
                    let inner = results.pop().unwrap();
                    results.push(FOFFormula::Not(Box::new(inner)));
                }

                WorkItem::CombineAnd => {
                    let right = results.pop().unwrap();
                    let left = results.pop().unwrap();
                    results.push(FOFFormula::And(Box::new(left), Box::new(right)));
                }

                WorkItem::CombineOr => {
                    let right = results.pop().unwrap();
                    let left = results.pop().unwrap();
                    results.push(FOFFormula::Or(Box::new(left), Box::new(right)));
                }

                WorkItem::CombineQuantified(q, v) => {
                    let inner = results.pop().unwrap();
                    results.push(FOFFormula::Quantified(q, v, Box::new(inner)));
                }

                WorkItem::CombineImplies => {
                    let right = results.pop().unwrap();
                    let left = results.pop().unwrap();
                    results.push(FOFFormula::Implies(Box::new(left), Box::new(right)));
                }

                WorkItem::CombineIff => {
                    let right = results.pop().unwrap();
                    let left = results.pop().unwrap();
                    results.push(FOFFormula::Iff(Box::new(left), Box::new(right)));
                }

                WorkItem::CombineXor => {
                    let right = results.pop().unwrap();
                    let left = results.pop().unwrap();
                    results.push(FOFFormula::Xor(Box::new(left), Box::new(right)));
                }

                WorkItem::CombineNand => {
                    let right = results.pop().unwrap();
                    let left = results.pop().unwrap();
                    results.push(FOFFormula::Nand(Box::new(left), Box::new(right)));
                }

                WorkItem::CombineNor => {
                    let right = results.pop().unwrap();
                    let left = results.pop().unwrap();
                    results.push(FOFFormula::Nor(Box::new(left), Box::new(right)));
                }
            }
        }

        results.pop().unwrap()
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
        // Iterative removal of universal quantifiers
        enum WorkItem {
            Process(FOFFormula),
            CombineAnd,
            CombineOr,
            CombineNot,
        }

        let mut stack: Vec<WorkItem> = vec![WorkItem::Process(formula)];
        let mut results: Vec<FOFFormula> = Vec::new();

        while let Some(item) = stack.pop() {
            match item {
                WorkItem::Process(f) => match f {
                    FOFFormula::Quantified(Quantifier::Forall, _, inner) => {
                        // Skip the quantifier, process the body
                        stack.push(WorkItem::Process(*inner));
                    }
                    FOFFormula::And(f1, f2) => {
                        stack.push(WorkItem::CombineAnd);
                        stack.push(WorkItem::Process(*f2));
                        stack.push(WorkItem::Process(*f1));
                    }
                    FOFFormula::Or(f1, f2) => {
                        stack.push(WorkItem::CombineOr);
                        stack.push(WorkItem::Process(*f2));
                        stack.push(WorkItem::Process(*f1));
                    }
                    FOFFormula::Not(inner) => {
                        stack.push(WorkItem::CombineNot);
                        stack.push(WorkItem::Process(*inner));
                    }
                    _ => {
                        results.push(f);
                    }
                },

                WorkItem::CombineAnd => {
                    let right = results.pop().unwrap();
                    let left = results.pop().unwrap();
                    results.push(FOFFormula::And(Box::new(left), Box::new(right)));
                }

                WorkItem::CombineOr => {
                    let right = results.pop().unwrap();
                    let left = results.pop().unwrap();
                    results.push(FOFFormula::Or(Box::new(left), Box::new(right)));
                }

                WorkItem::CombineNot => {
                    let inner = results.pop().unwrap();
                    results.push(FOFFormula::Not(Box::new(inner)));
                }
            }
        }

        results.pop().unwrap()
    }

    fn distribute_to_cnf(&self, formula: FOFFormula) -> Result<Vec<Clause>, CNFConversionError> {
        // Iterative CNF distribution - matches original recursive structure
        // Key: work with Vec<Clause> directly, distribute OR over AND inline
        enum WorkItem {
            Process(FOFFormula),
            CombineAnd,       // Concatenate clause lists
            CombineOrCross,   // Cross-product of clause lists
        }

        let mut stack: Vec<WorkItem> = vec![WorkItem::Process(formula)];
        let mut results: Vec<Vec<Clause>> = Vec::new();

        while let Some(item) = stack.pop() {
            self.check_timeout()?;

            match item {
                WorkItem::Process(f) => {
                    match f {
                        FOFFormula::And(f1, f2) => {
                            stack.push(WorkItem::CombineAnd);
                            stack.push(WorkItem::Process(*f2));
                            stack.push(WorkItem::Process(*f1));
                        }

                        FOFFormula::Or(f1, f2) => {
                            // Check if we need to distribute OR over AND
                            match (*f1, *f2) {
                                (FOFFormula::And(a1, a2), f2) => {
                                    // (A & B) | C => (A | C) & (B | C)
                                    let c1 = FOFFormula::Or(a1, Box::new(f2.clone()));
                                    let c2 = FOFFormula::Or(a2, Box::new(f2));
                                    stack.push(WorkItem::CombineAnd);
                                    stack.push(WorkItem::Process(c2));
                                    stack.push(WorkItem::Process(c1));
                                }
                                (f1, FOFFormula::And(a1, a2)) => {
                                    // C | (A & B) => (C | A) & (C | B)
                                    let c1 = FOFFormula::Or(Box::new(f1.clone()), a1);
                                    let c2 = FOFFormula::Or(Box::new(f1), a2);
                                    stack.push(WorkItem::CombineAnd);
                                    stack.push(WorkItem::Process(c2));
                                    stack.push(WorkItem::Process(c1));
                                }
                                (f1, f2) => {
                                    // No And at top level of either child
                                    // Recurse and take cross product
                                    stack.push(WorkItem::CombineOrCross);
                                    stack.push(WorkItem::Process(f2));
                                    stack.push(WorkItem::Process(f1));
                                }
                            }
                        }

                        FOFFormula::Atom(_) | FOFFormula::Not(_) => {
                            results.push(vec![self.formula_to_clause(f)]);
                        }

                        _ => panic!("Unexpected formula type in CNF conversion: {:?}", f),
                    }
                }

                WorkItem::CombineAnd => {
                    let right = results.pop().unwrap();
                    let mut left = results.pop().unwrap();
                    left.extend(right);
                    results.push(left);
                }

                WorkItem::CombineOrCross => {
                    let clauses2 = results.pop().unwrap();
                    let clauses1 = results.pop().unwrap();

                    let mut result = Vec::new();
                    for c1 in &clauses1 {
                        for c2 in &clauses2 {
                            let mut combined = c1.literals.clone();
                            combined.extend(c2.literals.clone());
                            result.push(Clause::with_role(combined, self.role));
                        }
                    }
                    results.push(result);
                }
            }
        }

        Ok(results.pop().unwrap_or_default())
    }

    fn formula_to_clause(&self, formula: FOFFormula) -> Clause {
        let literals = self.collect_literals(formula);
        Clause::with_role(literals, self.role)
    }

    fn collect_literals(&self, formula: FOFFormula) -> Vec<Literal> {
        // Iterative literal collection
        let mut stack: Vec<FOFFormula> = vec![formula];
        let mut literals: Vec<Literal> = Vec::new();

        while let Some(f) = stack.pop() {
            match f {
                FOFFormula::Or(f1, f2) => {
                    stack.push(*f2);
                    stack.push(*f1);
                }

                FOFFormula::Atom(atom) => {
                    literals.push(Literal::positive(atom));
                }

                FOFFormula::Not(inner) => match *inner {
                    FOFFormula::Atom(atom) => {
                        literals.push(Literal::negative(atom));
                    }
                    _ => panic!("Negation of non-atom in CNF: {:?}", inner),
                },

                _ => panic!("Non-disjunctive formula in clause: {:?}", f),
            }
        }

        literals
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
        let cnf = fof_to_cnf(formula).unwrap();

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
        let cnf = fof_to_cnf(formula).unwrap();

        assert_eq!(cnf.clauses.len(), 1);
        assert_eq!(cnf.clauses[0].literals.len(), 1);

        // Check that the variable was replaced with a Skolem constant
        match &cnf.clauses[0].literals[0].atom.args[0] {
            Term::Constant(c) => assert!(c.name.starts_with("sk")),
            _ => panic!("Expected Skolem constant"),
        }
    }

    /// Test CNF conversion of biconditional with quantifiers on both sides
    ///
    /// Formula: (∃X.p(X) <=> ∀Y.p(Y))
    ///
    /// With definitional CNF, this becomes:
    ///   def0                           (the biconditional is asserted true)
    ///   ~def0 | ~p(X) | p(Y)           (if def0 and p(X), then p(Y))
    ///   ~def0 | ~p(sk0) | p(sk1)       (if def0 and ~p(sk0), then p(sk1))
    ///
    /// The definition predicate `def0` connects the two directions without
    /// duplicating the quantified subformulas.
    #[test]
    fn test_biconditional_with_quantifiers() {
        // Build: (∃X.p(X) <=> ∀Y.p(Y))
        let x = Variable {
            name: "X".to_string(),
        };
        let y = Variable {
            name: "Y".to_string(),
        };
        let p = PredicateSymbol {
            name: "p".to_string(),
            arity: 1,
        };

        // p(X)
        let p_x = FOFFormula::Atom(Atom {
            predicate: p.clone(),
            args: vec![Term::Variable(x.clone())],
        });

        // p(Y)
        let p_y = FOFFormula::Atom(Atom {
            predicate: p.clone(),
            args: vec![Term::Variable(y.clone())],
        });

        // ∃X.p(X)
        let exists_x_px = FOFFormula::Quantified(Quantifier::Exists, x, Box::new(p_x));

        // ∀Y.p(Y)
        let forall_y_py = FOFFormula::Quantified(Quantifier::Forall, y, Box::new(p_y));

        // (∃X.p(X) <=> ∀Y.p(Y))
        let formula = FOFFormula::Iff(Box::new(exists_x_px), Box::new(forall_y_py));

        let cnf = fof_to_cnf(formula).unwrap();

        // With definitional CNF, we should have:
        // 1. A definition predicate (def0 or similar)
        // 2. The formula should NOT have duplicated quantified subformulas
        let has_definition_predicate = cnf
            .clauses
            .iter()
            .any(|c| c.literals.iter().any(|l| l.atom.predicate.name.starts_with("def")));

        assert!(
            has_definition_predicate,
            "Definitional CNF should introduce a definition predicate. Clauses: {:?}",
            cnf.clauses.iter().map(|c| c.to_string()).collect::<Vec<_>>()
        );

        // Count how many clauses contain Skolem constants - with definitional CNF,
        // the quantified subformulas are not duplicated, so there should be fewer
        // clauses with Skolem constants than without the fix
        let clauses_with_skolem = cnf
            .clauses
            .iter()
            .filter(|c| {
                c.literals.iter().any(|l| {
                    l.atom.args.iter().any(|arg| {
                        matches!(arg, Term::Constant(c) if c.name.starts_with("sk"))
                    })
                })
            })
            .count();

        // Should have at most 1-2 clauses with Skolem constants (from the definitions)
        // rather than having Skolems spread across many duplicated clauses
        assert!(
            clauses_with_skolem <= 2,
            "Definitional CNF should minimize Skolem constant spread. \
             Found {} clauses with Skolem constants. Clauses: {:?}",
            clauses_with_skolem,
            cnf.clauses.iter().map(|c| c.to_string()).collect::<Vec<_>>()
        );
    }
}
