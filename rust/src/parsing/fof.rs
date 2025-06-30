//! FOF (First Order Form) formula representation and CNF conversion
//!
//! This module provides:
//! - AST types for representing FOF formulas
//! - FOF to CNF conversion using standard transformations

use crate::parsing::parse_types::{ParseClause, ParseLiteral, ParsePredicate, ParseTerm};
use std::collections::HashMap;

/// First-order formula representation
#[derive(Debug, Clone, PartialEq)]
pub enum FofFormula {
    /// Atomic formula (predicate)
    Atom(ParsePredicate),
    
    /// Negation
    Not(Box<FofFormula>),
    
    /// Conjunction (and)
    And(Vec<FofFormula>),
    
    /// Disjunction (or)
    Or(Vec<FofFormula>),
    
    /// Implication (=>)
    Implies(Box<FofFormula>, Box<FofFormula>),
    
    /// Biconditional (<=>)
    Iff(Box<FofFormula>, Box<FofFormula>),
    
    /// Universal quantification
    Forall(Vec<String>, Box<FofFormula>),
    
    /// Existential quantification
    Exists(Vec<String>, Box<FofFormula>),
}

impl FofFormula {
    /// Convert FOF formula to CNF (Conjunctive Normal Form)
    /// Returns a vector of clauses representing the CNF
    pub fn to_cnf(&self) -> Vec<ParseClause> {
        // Step 1: Eliminate biconditionals and implications
        let formula = self.eliminate_iff_implies();
        
        // Step 2: Move negations inward (NNF)
        let formula = formula.to_nnf();
        
        // Step 3: Skolemize (eliminate existential quantifiers)
        let mut skolem_counter = 0;
        let formula = formula.skolemize(&mut skolem_counter, &Vec::new());
        
        // Step 4: Drop universal quantifiers
        let formula = formula.drop_universals();
        
        // Step 5: Split top-level conjunctions to avoid unnecessary distribution
        let formulas = formula.split_top_level_conjunctions();
        
        // Step 5b: Flatten any nested ORs in the formulas
        let formulas: Vec<FofFormula> = formulas.into_iter()
            .map(|f| f.flatten_nested_ors())
            .collect();
        
        // Step 6: Convert each formula to clauses
        let mut all_clauses = Vec::new();
        for f in formulas {
            if f.is_clause() {
                // Already a clause - convert directly without distribution
                all_clauses.push(f.to_clause());
            } else {
                // Distribute OR over AND to get CNF
                let cnf = f.distribute_or();
                all_clauses.extend(cnf.extract_clauses());
            }
        }
        
        all_clauses
    }
    
    /// Calculate the size of a formula (approximate node count)
    #[allow(dead_code)]
    fn size(&self) -> usize {
        match self {
            FofFormula::Atom(_) => 1,
            FofFormula::Not(f) => 1 + f.size(),
            FofFormula::And(fs) | FofFormula::Or(fs) => {
                1 + fs.iter().map(|f| f.size()).sum::<usize>()
            }
            FofFormula::Implies(f1, f2) | FofFormula::Iff(f1, f2) => {
                1 + f1.size() + f2.size()
            }
            FofFormula::Forall(vars, f) | FofFormula::Exists(vars, f) => {
                1 + vars.len() + f.size()
            }
        }
    }
    
    /// Eliminate biconditionals and implications
    fn eliminate_iff_implies(&self) -> FofFormula {
        match self {
            FofFormula::Atom(p) => FofFormula::Atom(p.clone()),
            FofFormula::Not(f) => FofFormula::Not(Box::new(f.eliminate_iff_implies())),
            FofFormula::And(fs) => FofFormula::And(
                fs.iter().map(|f| f.eliminate_iff_implies()).collect()
            ),
            FofFormula::Or(fs) => FofFormula::Or(
                fs.iter().map(|f| f.eliminate_iff_implies()).collect()
            ),
            FofFormula::Implies(a, b) => {
                // A => B becomes ~A | B
                FofFormula::Or(vec![
                    FofFormula::Not(Box::new(a.eliminate_iff_implies())),
                    b.eliminate_iff_implies()
                ])
            },
            FofFormula::Iff(a, b) => {
                // A <=> B becomes (A => B) & (B => A)
                // Which becomes (~A | B) & (~B | A)
                let a_elim = a.eliminate_iff_implies();
                let b_elim = b.eliminate_iff_implies();
                FofFormula::And(vec![
                    FofFormula::Or(vec![
                        FofFormula::Not(Box::new(a_elim.clone())),
                        b_elim.clone()
                    ]),
                    FofFormula::Or(vec![
                        FofFormula::Not(Box::new(b_elim)),
                        a_elim
                    ])
                ])
            },
            FofFormula::Forall(vars, f) => {
                FofFormula::Forall(vars.clone(), Box::new(f.eliminate_iff_implies()))
            },
            FofFormula::Exists(vars, f) => {
                FofFormula::Exists(vars.clone(), Box::new(f.eliminate_iff_implies()))
            }
        }
    }
    
    /// Convert to Negation Normal Form (NNF)
    fn to_nnf(&self) -> FofFormula {
        self.to_nnf_impl(false)
    }
    
    fn to_nnf_impl(&self, negate: bool) -> FofFormula {
        match (self, negate) {
            (FofFormula::Atom(p), false) => FofFormula::Atom(p.clone()),
            (FofFormula::Atom(p), true) => FofFormula::Not(Box::new(FofFormula::Atom(p.clone()))),
            
            (FofFormula::Not(f), negate) => f.to_nnf_impl(!negate),
            
            (FofFormula::And(fs), false) => FofFormula::And(
                fs.iter().map(|f| f.to_nnf_impl(false)).collect()
            ),
            (FofFormula::And(fs), true) => {
                // ~(A & B) becomes ~A | ~B
                FofFormula::Or(fs.iter().map(|f| f.to_nnf_impl(true)).collect())
            },
            
            (FofFormula::Or(fs), false) => FofFormula::Or(
                fs.iter().map(|f| f.to_nnf_impl(false)).collect()
            ),
            (FofFormula::Or(fs), true) => {
                // ~(A | B) becomes ~A & ~B
                FofFormula::And(fs.iter().map(|f| f.to_nnf_impl(true)).collect())
            },
            
            (FofFormula::Forall(vars, f), false) => {
                FofFormula::Forall(vars.clone(), Box::new(f.to_nnf_impl(false)))
            },
            (FofFormula::Forall(vars, f), true) => {
                // ~(forall X. P) becomes exists X. ~P
                FofFormula::Exists(vars.clone(), Box::new(f.to_nnf_impl(true)))
            },
            
            (FofFormula::Exists(vars, f), false) => {
                FofFormula::Exists(vars.clone(), Box::new(f.to_nnf_impl(false)))
            },
            (FofFormula::Exists(vars, f), true) => {
                // ~(exists X. P) becomes forall X. ~P
                FofFormula::Forall(vars.clone(), Box::new(f.to_nnf_impl(true)))
            },
            
            (FofFormula::Implies(_, _), _) | (FofFormula::Iff(_, _), _) => {
                panic!("Implications and biconditionals should be eliminated before NNF")
            }
        }
    }
    
    /// Skolemize: eliminate existential quantifiers
    fn skolemize(&self, counter: &mut usize, universal_vars: &Vec<String>) -> FofFormula {
        match self {
            FofFormula::Atom(p) => FofFormula::Atom(p.clone()),
            FofFormula::Not(f) => FofFormula::Not(Box::new(f.skolemize(counter, universal_vars))),
            FofFormula::And(fs) => FofFormula::And(
                fs.iter().map(|f| f.skolemize(counter, universal_vars)).collect()
            ),
            FofFormula::Or(fs) => FofFormula::Or(
                fs.iter().map(|f| f.skolemize(counter, universal_vars)).collect()
            ),
            FofFormula::Forall(vars, f) => {
                let mut new_universal_vars = universal_vars.clone();
                new_universal_vars.extend(vars.clone());
                FofFormula::Forall(vars.clone(), Box::new(f.skolemize(counter, &new_universal_vars)))
            },
            FofFormula::Exists(vars, f) => {
                // Replace existentially quantified variables with Skolem functions
                let mut substitution = HashMap::new();
                
                for var in vars {
                    *counter += 1;
                    let skolem_name = format!("sk{}", counter);
                    
                    let skolem_term = if universal_vars.is_empty() {
                        // Skolem constant
                        ParseTerm::Constant(skolem_name)
                    } else {
                        // Skolem function depending on universal variables
                        ParseTerm::Function {
                            name: skolem_name,
                            args: universal_vars.iter()
                                .map(|v| ParseTerm::Variable(v.clone()))
                                .collect()
                        }
                    };
                    
                    substitution.insert(var.clone(), skolem_term);
                }
                
                // Apply substitution and continue skolemization
                let substituted = f.substitute(&substitution);
                substituted.skolemize(counter, universal_vars)
            },
            FofFormula::Implies(_, _) | FofFormula::Iff(_, _) => {
                panic!("Implications and biconditionals should be eliminated before Skolemization")
            }
        }
    }
    
    /// Substitute terms for variables in the formula
    fn substitute(&self, subst: &HashMap<String, ParseTerm>) -> FofFormula {
        match self {
            FofFormula::Atom(p) => {
                let new_args = p.args.iter().map(|term| {
                    substitute_term(term, subst)
                }).collect();
                FofFormula::Atom(ParsePredicate { name: p.name.clone(), args: new_args })
            },
            FofFormula::Not(f) => FofFormula::Not(Box::new(f.substitute(subst))),
            FofFormula::And(fs) => FofFormula::And(
                fs.iter().map(|f| f.substitute(subst)).collect()
            ),
            FofFormula::Or(fs) => FofFormula::Or(
                fs.iter().map(|f| f.substitute(subst)).collect()
            ),
            FofFormula::Forall(vars, f) => {
                // Remove bound variables from substitution
                let mut new_subst = subst.clone();
                for var in vars {
                    new_subst.remove(var);
                }
                FofFormula::Forall(vars.clone(), Box::new(f.substitute(&new_subst)))
            },
            FofFormula::Exists(vars, f) => {
                // Remove bound variables from substitution
                let mut new_subst = subst.clone();
                for var in vars {
                    new_subst.remove(var);
                }
                FofFormula::Exists(vars.clone(), Box::new(f.substitute(&new_subst)))
            },
            FofFormula::Implies(a, b) => FofFormula::Implies(
                Box::new(a.substitute(subst)),
                Box::new(b.substitute(subst))
            ),
            FofFormula::Iff(a, b) => FofFormula::Iff(
                Box::new(a.substitute(subst)),
                Box::new(b.substitute(subst))
            ),
        }
    }
    
    /// Drop universal quantifiers (after skolemization)
    fn drop_universals(&self) -> FofFormula {
        match self {
            FofFormula::Forall(_, f) => f.drop_universals(),
            FofFormula::Not(f) => FofFormula::Not(Box::new(f.drop_universals())),
            FofFormula::And(fs) => FofFormula::And(
                fs.iter().map(|f| f.drop_universals()).collect()
            ),
            FofFormula::Or(fs) => FofFormula::Or(
                fs.iter().map(|f| f.drop_universals()).collect()
            ),
            FofFormula::Exists(_, _) => {
                panic!("Existential quantifiers should be eliminated before dropping universals")
            },
            _ => self.clone()
        }
    }
    
    /// Distribute OR over AND to get CNF
    fn distribute_or(&self) -> FofFormula {
        match self {
            FofFormula::Or(fs) => {
                let distributed: Vec<FofFormula> = fs.iter()
                    .map(|f| f.distribute_or())
                    .collect();
                
                // Check if any operand is an AND
                if let Some(and_idx) = distributed.iter().position(|f| matches!(f, FofFormula::And(_))) {
                    // Distribute OR over the AND
                    if let FofFormula::And(and_clauses) = &distributed[and_idx] {
                        let mut result_clauses = Vec::new();
                        
                        for clause in and_clauses {
                            let mut or_args = Vec::new();
                            
                            // Add all non-AND operands
                            for (i, f) in distributed.iter().enumerate() {
                                if i != and_idx {
                                    or_args.push(f.clone());
                                }
                            }
                            
                            // Add the current AND clause
                            or_args.push(clause.clone());
                            
                            result_clauses.push(FofFormula::Or(or_args).distribute_or());
                        }
                        
                        FofFormula::And(result_clauses)
                    } else {
                        unreachable!()
                    }
                } else {
                    FofFormula::Or(distributed)
                }
            },
            FofFormula::And(fs) => FofFormula::And(
                fs.iter().map(|f| f.distribute_or()).collect()
            ),
            FofFormula::Not(f) => FofFormula::Not(Box::new(f.distribute_or())),
            _ => self.clone()
        }
    }
    
    /// Extract clauses from CNF formula
    fn extract_clauses(&self) -> Vec<ParseClause> {
        match self {
            FofFormula::And(fs) => {
                fs.iter().flat_map(|f| f.extract_clauses()).collect()
            },
            FofFormula::Or(fs) => {
                // Handle empty disjunction (should not happen in normal CNF)
                if fs.is_empty() {
                    return vec![ParseClause { literals: vec![] }];
                }
                
                // Flatten nested ORs and collect literals
                let mut literals = Vec::new();
                self.collect_literals_from_or(&mut literals);
                vec![ParseClause { literals }]
            },
            FofFormula::Not(f) => {
                if let FofFormula::Atom(p) = f.as_ref() {
                    vec![ParseClause { literals: vec![ParseLiteral { predicate: p.clone(), polarity: false }] }]
                } else {
                    panic!("Not should only be applied to atoms in CNF")
                }
            },
            FofFormula::Atom(p) => {
                vec![ParseClause { literals: vec![ParseLiteral { predicate: p.clone(), polarity: true }] }]
            },
            _ => panic!("Formula should be in CNF form: {:?}", self)
        }
    }
    
    /// Helper to flatten OR structures and collect literals
    fn collect_literals_from_or(&self, literals: &mut Vec<ParseLiteral>) {
        match self {
            FofFormula::Or(fs) => {
                // Recursively collect from nested ORs
                for f in fs {
                    f.collect_literals_from_or(literals);
                }
            },
            FofFormula::Atom(p) => {
                literals.push(ParseLiteral { predicate: p.clone(), polarity: true });
            },
            FofFormula::Not(f) => {
                if let FofFormula::Atom(p) = f.as_ref() {
                    literals.push(ParseLiteral { predicate: p.clone(), polarity: false });
                } else {
                    panic!("Not should only be applied to atoms in CNF");
                }
            },
            _ => panic!("Unexpected formula in OR: {:?}", self),
        }
    }
    
    /// Split top-level conjunctions into separate formulas
    fn split_top_level_conjunctions(&self) -> Vec<FofFormula> {
        match self {
            FofFormula::And(conjuncts) => {
                // Recursively split nested conjunctions at top level
                let mut result = Vec::new();
                for conjunct in conjuncts {
                    result.extend(conjunct.split_top_level_conjunctions());
                }
                result
            }
            _ => vec![self.clone()]
        }
    }
    
    /// Flatten nested ORs into a single OR
    fn flatten_nested_ors(&self) -> FofFormula {
        match self {
            FofFormula::Or(disjuncts) => {
                let mut flattened = Vec::new();
                for d in disjuncts {
                    match d {
                        FofFormula::Or(_inner) => {
                            // Recursively flatten nested ORs
                            let inner_flat = d.flatten_nested_ors();
                            if let FofFormula::Or(inner_disj) = inner_flat {
                                flattened.extend(inner_disj);
                            } else {
                                flattened.push(inner_flat);
                            }
                        }
                        _ => flattened.push(d.clone()),
                    }
                }
                FofFormula::Or(flattened)
            }
            FofFormula::And(conjuncts) => {
                // Recursively flatten in AND formulas too
                FofFormula::And(conjuncts.iter().map(|c| c.flatten_nested_ors()).collect())
            }
            FofFormula::Not(f) => FofFormula::Not(Box::new(f.flatten_nested_ors())),
            _ => self.clone()
        }
    }
    
    /// Check if this formula is already a clause (disjunction of literals)
    fn is_clause(&self) -> bool {
        match self {
            FofFormula::Atom(_) => true,
            FofFormula::Not(f) => matches!(f.as_ref(), FofFormula::Atom(_)),
            FofFormula::Or(disjuncts) => {
                // Check if all disjuncts are literals or can be flattened to literals
                disjuncts.iter().all(|d| d.is_literal() || d.is_flat_disjunction())
            }
            _ => false
        }
    }
    
    /// Check if this is a disjunction that only contains literals (possibly nested)
    fn is_flat_disjunction(&self) -> bool {
        match self {
            FofFormula::Or(disjuncts) => {
                disjuncts.iter().all(|d| d.is_literal() || d.is_flat_disjunction())
            }
            _ => false
        }
    }
    
    /// Check if this formula is a literal (atom or negated atom)
    fn is_literal(&self) -> bool {
        match self {
            FofFormula::Atom(_) => true,
            FofFormula::Not(f) => matches!(f.as_ref(), FofFormula::Atom(_)),
            _ => false
        }
    }
    
    /// Convert a clause-form formula directly to a ParseClause
    fn to_clause(&self) -> ParseClause {
        match self {
            FofFormula::Atom(p) => ParseClause { literals: vec![ParseLiteral { predicate: p.clone(), polarity: true }] },
            FofFormula::Not(f) => {
                if let FofFormula::Atom(p) = f.as_ref() {
                    ParseClause { literals: vec![ParseLiteral { predicate: p.clone(), polarity: false }] }
                } else {
                    panic!("Not should only be applied to atoms in clause")
                }
            }
            FofFormula::Or(_disjuncts) => {
                // Flatten nested ORs and collect all literals
                let mut literals = Vec::new();
                self.collect_literals_from_flat_or(&mut literals);
                ParseClause { literals }
            }
            _ => panic!("Formula is not in clause form: {:?}", self)
        }
    }
    
    /// Collect literals from a flat disjunction (possibly with nested ORs)
    fn collect_literals_from_flat_or(&self, literals: &mut Vec<ParseLiteral>) {
        match self {
            FofFormula::Atom(p) => literals.push(ParseLiteral { predicate: p.clone(), polarity: true }),
            FofFormula::Not(f) => {
                if let FofFormula::Atom(p) = f.as_ref() {
                    literals.push(ParseLiteral { predicate: p.clone(), polarity: false });
                }
            }
            FofFormula::Or(disjuncts) => {
                for d in disjuncts {
                    d.collect_literals_from_flat_or(literals);
                }
            }
            _ => {} // Skip non-literals in malformed formulas
        }
    }
    
    /// Convert formula to literal (for CNF extraction)
    #[allow(dead_code)]
    fn to_literal(&self) -> ParseLiteral {
        match self {
            FofFormula::Atom(p) => ParseLiteral { predicate: p.clone(), polarity: true },
            FofFormula::Not(f) => {
                if let FofFormula::Atom(p) = f.as_ref() {
                    ParseLiteral { predicate: p.clone(), polarity: false }
                } else {
                    panic!("Not should only be applied to atoms in CNF")
                }
            },
            _ => panic!("Only atoms and negated atoms can be converted to literals")
        }
    }
}

/// Substitute terms in a term
fn substitute_term(term: &ParseTerm, subst: &HashMap<String, ParseTerm>) -> ParseTerm {
    match term {
        ParseTerm::Variable(v) => {
            subst.get(v).cloned().unwrap_or_else(|| term.clone())
        },
        ParseTerm::Constant(_) => term.clone(),
        ParseTerm::Function { name, args } => {
            ParseTerm::Function {
                name: name.clone(),
                args: args.iter().map(|t| substitute_term(t, subst)).collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simple_cnf_conversion() {
        // P & Q
        let formula = FofFormula::And(vec![
            FofFormula::Atom(ParsePredicate { name: "P".to_string(), args: vec![] }),
            FofFormula::Atom(ParsePredicate { name: "Q".to_string(), args: vec![] }),
        ]);
        
        let clauses = formula.to_cnf();
        assert_eq!(clauses.len(), 2);
    }
    
    #[test]
    fn test_implication_elimination() {
        // P => Q becomes ~P | Q
        let formula = FofFormula::Implies(
            Box::new(FofFormula::Atom(ParsePredicate { name: "P".to_string(), args: vec![] })),
            Box::new(FofFormula::Atom(ParsePredicate { name: "Q".to_string(), args: vec![] }))
        );
        
        let clauses = formula.to_cnf();
        assert_eq!(clauses.len(), 1);
        assert_eq!(clauses[0].literals.len(), 2);
    }
    
    #[test]
    fn test_skolemization() {
        // ?[X]: p(X) becomes p(sk1)
        let formula = FofFormula::Exists(
            vec!["X".to_string()],
            Box::new(FofFormula::Atom(ParsePredicate { name: "p".to_string(), args: vec![ParseTerm::Variable("X".to_string())] }))
        );
        
        let clauses = formula.to_cnf();
        assert_eq!(clauses.len(), 1);
        assert_eq!(clauses[0].literals.len(), 1);
        
        // Should have replaced X with a Skolem constant
        let pred = &clauses[0].literals[0].predicate;
        assert_eq!(pred.name, "p");
        assert_eq!(pred.args.len(), 1);
        match &pred.args[0] {
            ParseTerm::Constant(name) => assert!(name.starts_with("sk")),
            _ => panic!("Expected Skolem constant"),
        }
    }
    
    #[test]
    fn test_complex_fof_conversion() {
        // ![X,Y]: ((p(X) & q(Y)) => (r(X) | s(Y)))
        // Should become: ~p(X) | ~q(Y) | r(X) | s(Y)
        let formula = FofFormula::Forall(
            vec!["X".to_string(), "Y".to_string()],
            Box::new(FofFormula::Implies(
                Box::new(FofFormula::And(vec![
                    FofFormula::Atom(ParsePredicate { name: "p".to_string(), args: vec![ParseTerm::Variable("X".to_string())] }),
                    FofFormula::Atom(ParsePredicate { name: "q".to_string(), args: vec![ParseTerm::Variable("Y".to_string())] }),
                ])),
                Box::new(FofFormula::Or(vec![
                    FofFormula::Atom(ParsePredicate { name: "r".to_string(), args: vec![ParseTerm::Variable("X".to_string())] }),
                    FofFormula::Atom(ParsePredicate { name: "s".to_string(), args: vec![ParseTerm::Variable("Y".to_string())] }),
                ]))
            ))
        );
        
        let clauses = formula.to_cnf();
        assert_eq!(clauses.len(), 1);
        assert_eq!(clauses[0].literals.len(), 4);
    }
}