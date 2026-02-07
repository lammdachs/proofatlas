//! Conversion from FOF to CNF
//!
//! This module implements the standard algorithm for converting
//! first-order formulas to Conjunctive Normal Form (CNF).

use std::time::Instant;

use super::fof::{FOFFormula, Quantifier};
use crate::logic::{
    Atom, CNFFormula, Clause, ClauseRole, Constant, FunctionSymbol, Interner, Literal,
    PredicateSymbol, Term, Variable,
};

/// Error during CNF conversion
#[derive(Debug, Clone)]
pub enum CNFConversionError {
    Timeout,
    MemoryLimit,
}

/// Polarity context for definitional CNF transformation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Polarity {
    /// Formula appears only in positive context
    Positive,
    /// Formula appears only in negative context
    Negative,
    /// Formula appears in both contexts (e.g., inside biconditional)
    Both,
}

impl Polarity {
    /// Flip polarity (for negation, implication antecedent)
    fn flip(self) -> Self {
        match self {
            Polarity::Positive => Polarity::Negative,
            Polarity::Negative => Polarity::Positive,
            Polarity::Both => Polarity::Both,
        }
    }

    /// Inside a biconditional, everything becomes Both
    fn in_biconditional(self) -> Self {
        Polarity::Both
    }
}

impl std::fmt::Display for CNFConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CNFConversionError::Timeout => write!(f, "CNF conversion timed out"),
            CNFConversionError::MemoryLimit => write!(f, "CNF conversion exceeded memory limit"),
        }
    }
}

/// Convert a FOF formula to CNF using the provided interner
pub fn fof_to_cnf(
    formula: FOFFormula,
    interner: &mut Interner,
) -> Result<CNFFormula, CNFConversionError> {
    fof_to_cnf_with_role(formula, ClauseRole::Axiom, None, None, interner)
}

/// Convert a FOF formula to CNF with a specific role and optional timeout/memory limit
pub fn fof_to_cnf_with_role(
    formula: FOFFormula,
    role: ClauseRole,
    timeout: Option<Instant>,
    memory_limit: Option<usize>,
    interner: &mut Interner,
) -> Result<CNFFormula, CNFConversionError> {
    let mut converter = CNFConverter::new(role, timeout, memory_limit, interner);
    converter.convert(formula)
}

struct CNFConverter<'a> {
    skolem_counter: usize,
    def_counter: usize,
    universal_vars: Vec<Variable>,
    role: ClauseRole,
    timeout: Option<Instant>,
    memory_limit: Option<usize>,
    interner: &'a mut Interner,
}

impl<'a> CNFConverter<'a> {
    fn new(role: ClauseRole, timeout: Option<Instant>, memory_limit: Option<usize>, interner: &'a mut Interner) -> Self {
        CNFConverter {
            skolem_counter: 0,
            def_counter: 0,
            universal_vars: Vec::new(),
            role,
            timeout,
            memory_limit,
            interner,
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

    fn check_memory(&self) -> Result<(), CNFConversionError> {
        if let Some(limit) = self.memory_limit {
            if let Some(rss) = crate::config::process_memory_mb() {
                if rss >= limit {
                    return Err(CNFConversionError::MemoryLimit);
                }
            }
        }
        Ok(())
    }

    fn convert(&mut self, formula: FOFFormula) -> Result<CNFFormula, CNFConversionError> {
        // Step 0a: Simplify $true and $false
        let simplified = self.simplify_truth_constants(formula);

        // Handle degenerate cases
        if self.is_true_constant(&simplified) {
            // Formula is trivially true - no clauses needed
            return Ok(CNFFormula { clauses: vec![] });
        }
        if self.is_false_constant(&simplified) {
            // Formula is trivially false - return empty clause
            let mut empty_clause = Clause::new(vec![]);
            empty_clause.role = self.role;
            return Ok(CNFFormula {
                clauses: vec![empty_clause]
            });
        }

        // Step 0b: Apply definitional CNF to biconditionals with quantifiers
        // When A <=> B contains quantified subformulas, NNF expansion would duplicate them.
        // Instead, we replace the biconditional with a definition predicate and add
        // polarity-appropriate definition clauses.
        let mut definitions = Vec::new();
        let transformed = self.definitional_transform(simplified, Polarity::Positive, &mut definitions);

        // Combine with definitions
        let combined = definitions.into_iter().fold(transformed, |acc, def| {
            FOFFormula::And(Box::new(acc), Box::new(def))
        });

        // Step 0c: Standardize apart - rename bound variables to be unique
        // This prevents variable capture when the same variable name is used
        // in different quantifier scopes (e.g., ∃Y.P(Y) => ∀Y.Q(Y))
        let standardized = combined.standardize_apart(self.interner);

        // Step 1: Convert to NNF
        let nnf = standardized.to_nnf();

        // Step 2: Skolemize (remove existential quantifiers)
        let skolemized = self.skolemize(nnf);

        // Step 3: Remove universal quantifiers (they're implicit in CNF)
        let matrix = self.remove_universal_quantifiers(skolemized);

        // Step 4: Convert to CNF using distribution
        let clauses = self.distribute_to_cnf(matrix)?;

        Ok(CNFFormula { clauses })
    }

    /// Check if a formula is the $true constant
    fn is_true_constant(&self, formula: &FOFFormula) -> bool {
        if let FOFFormula::Atom(atom) = formula {
            self.interner.resolve_predicate(atom.predicate.id) == "$true"
        } else {
            false
        }
    }

    /// Check if a formula is the $false constant
    fn is_false_constant(&self, formula: &FOFFormula) -> bool {
        if let FOFFormula::Atom(atom) = formula {
            self.interner.resolve_predicate(atom.predicate.id) == "$false"
        } else {
            false
        }
    }

    /// Get the name of a distinct object (without the '"' prefix)
    fn distinct_object_name(&self, term: &Term) -> Option<String> {
        match term {
            Term::Constant(c) => {
                let name = self.interner.resolve_constant(c.id);
                if name.starts_with('"') {
                    Some(name[1..].to_string())
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Create an atom for $true
    fn make_true_atom(&mut self) -> FOFFormula {
        let pred = PredicateSymbol::new(self.interner.intern_predicate("$true"), 0);
        FOFFormula::Atom(Atom {
            predicate: pred,
            args: vec![],
        })
    }

    /// Create an atom for $false
    fn make_false_atom(&mut self) -> FOFFormula {
        let pred = PredicateSymbol::new(self.interner.intern_predicate("$false"), 0);
        FOFFormula::Atom(Atom {
            predicate: pred,
            args: vec![],
        })
    }

    /// Simplify formulas containing $true, $false, and distinct object equalities
    fn simplify_truth_constants(&mut self, formula: FOFFormula) -> FOFFormula {
        match formula {
            FOFFormula::Atom(ref atom) => {
                // Check for equality between distinct objects
                if atom.is_equality(self.interner) && atom.args.len() == 2 {
                    if let (Some(left), Some(right)) = (
                        self.distinct_object_name(&atom.args[0]),
                        self.distinct_object_name(&atom.args[1]),
                    ) {
                        // "A" = "B" is false if different, true if same
                        if left == right {
                            return self.make_true_atom();
                        } else {
                            return self.make_false_atom();
                        }
                    }
                }
                formula
            }

            FOFFormula::Not(f) => {
                let f = self.simplify_truth_constants(*f);
                if self.is_true_constant(&f) {
                    // ~$true = $false
                    self.make_false_atom()
                } else if self.is_false_constant(&f) {
                    // ~$false = $true
                    self.make_true_atom()
                } else {
                    FOFFormula::Not(Box::new(f))
                }
            }

            FOFFormula::And(f1, f2) => {
                let f1 = self.simplify_truth_constants(*f1);
                let f2 = self.simplify_truth_constants(*f2);
                if self.is_false_constant(&f1) || self.is_false_constant(&f2) {
                    // $false & A = $false
                    self.make_false_atom()
                } else if self.is_true_constant(&f1) {
                    // $true & A = A
                    f2
                } else if self.is_true_constant(&f2) {
                    // A & $true = A
                    f1
                } else {
                    FOFFormula::And(Box::new(f1), Box::new(f2))
                }
            }

            FOFFormula::Or(f1, f2) => {
                let f1 = self.simplify_truth_constants(*f1);
                let f2 = self.simplify_truth_constants(*f2);
                if self.is_true_constant(&f1) || self.is_true_constant(&f2) {
                    // $true | A = $true
                    self.make_true_atom()
                } else if self.is_false_constant(&f1) {
                    // $false | A = A
                    f2
                } else if self.is_false_constant(&f2) {
                    // A | $false = A
                    f1
                } else {
                    FOFFormula::Or(Box::new(f1), Box::new(f2))
                }
            }

            FOFFormula::Implies(f1, f2) => {
                let f1 = self.simplify_truth_constants(*f1);
                let f2 = self.simplify_truth_constants(*f2);
                if self.is_false_constant(&f1) || self.is_true_constant(&f2) {
                    // $false => A = $true, A => $true = $true
                    self.make_true_atom()
                } else if self.is_true_constant(&f1) {
                    // $true => A = A
                    f2
                } else if self.is_false_constant(&f2) {
                    // A => $false = ~A
                    FOFFormula::Not(Box::new(f1))
                } else {
                    FOFFormula::Implies(Box::new(f1), Box::new(f2))
                }
            }

            FOFFormula::Iff(f1, f2) => {
                let f1 = self.simplify_truth_constants(*f1);
                let f2 = self.simplify_truth_constants(*f2);
                if self.is_true_constant(&f1) {
                    // $true <=> A = A
                    f2
                } else if self.is_true_constant(&f2) {
                    // A <=> $true = A
                    f1
                } else if self.is_false_constant(&f1) {
                    // $false <=> A = ~A
                    FOFFormula::Not(Box::new(f2))
                } else if self.is_false_constant(&f2) {
                    // A <=> $false = ~A
                    FOFFormula::Not(Box::new(f1))
                } else {
                    FOFFormula::Iff(Box::new(f1), Box::new(f2))
                }
            }

            FOFFormula::Xor(f1, f2) => {
                let f1 = self.simplify_truth_constants(*f1);
                let f2 = self.simplify_truth_constants(*f2);
                if self.is_false_constant(&f1) {
                    // $false XOR A = A
                    f2
                } else if self.is_false_constant(&f2) {
                    // A XOR $false = A
                    f1
                } else if self.is_true_constant(&f1) {
                    // $true XOR A = ~A
                    FOFFormula::Not(Box::new(f2))
                } else if self.is_true_constant(&f2) {
                    // A XOR $true = ~A
                    FOFFormula::Not(Box::new(f1))
                } else {
                    FOFFormula::Xor(Box::new(f1), Box::new(f2))
                }
            }

            FOFFormula::Nand(f1, f2) => {
                let f1 = self.simplify_truth_constants(*f1);
                let f2 = self.simplify_truth_constants(*f2);
                // A NAND B = ~(A & B)
                if self.is_false_constant(&f1) || self.is_false_constant(&f2) {
                    self.make_true_atom()
                } else if self.is_true_constant(&f1) && self.is_true_constant(&f2) {
                    self.make_false_atom()
                } else if self.is_true_constant(&f1) {
                    FOFFormula::Not(Box::new(f2))
                } else if self.is_true_constant(&f2) {
                    FOFFormula::Not(Box::new(f1))
                } else {
                    FOFFormula::Nand(Box::new(f1), Box::new(f2))
                }
            }

            FOFFormula::Nor(f1, f2) => {
                let f1 = self.simplify_truth_constants(*f1);
                let f2 = self.simplify_truth_constants(*f2);
                // A NOR B = ~(A | B)
                if self.is_true_constant(&f1) || self.is_true_constant(&f2) {
                    self.make_false_atom()
                } else if self.is_false_constant(&f1) && self.is_false_constant(&f2) {
                    self.make_true_atom()
                } else if self.is_false_constant(&f1) {
                    FOFFormula::Not(Box::new(f2))
                } else if self.is_false_constant(&f2) {
                    FOFFormula::Not(Box::new(f1))
                } else {
                    FOFFormula::Nor(Box::new(f1), Box::new(f2))
                }
            }

            FOFFormula::Quantified(q, v, f) => {
                let f = self.simplify_truth_constants(*f);
                // Quantification over truth constants: ∀x.$true = $true, etc.
                if self.is_true_constant(&f) || self.is_false_constant(&f) {
                    f
                } else {
                    FOFFormula::Quantified(q, v, Box::new(f))
                }
            }
        }
    }

    /// Apply definitional transformation to biconditionals containing quantifiers.
    ///
    /// Tracks polarity to generate the correct definitions:
    /// - Positive: D => (A <=> B)
    /// - Negative: (A <=> B) => D
    /// - Both: full equivalence D <=> (A <=> B)
    fn definitional_transform(
        &mut self,
        formula: FOFFormula,
        polarity: Polarity,
        definitions: &mut Vec<FOFFormula>,
    ) -> FOFFormula {
        match formula {
            FOFFormula::Atom(_) => formula,

            FOFFormula::Not(f) => {
                // Negation flips polarity
                FOFFormula::Not(Box::new(
                    self.definitional_transform(*f, polarity.flip(), definitions),
                ))
            }

            FOFFormula::And(f1, f2) => FOFFormula::And(
                Box::new(self.definitional_transform(*f1, polarity, definitions)),
                Box::new(self.definitional_transform(*f2, polarity, definitions)),
            ),

            FOFFormula::Or(f1, f2) => FOFFormula::Or(
                Box::new(self.definitional_transform(*f1, polarity, definitions)),
                Box::new(self.definitional_transform(*f2, polarity, definitions)),
            ),

            FOFFormula::Implies(f1, f2) => {
                // A => B: A has flipped polarity, B keeps polarity
                FOFFormula::Implies(
                    Box::new(self.definitional_transform(*f1, polarity.flip(), definitions)),
                    Box::new(self.definitional_transform(*f2, polarity, definitions)),
                )
            }

            FOFFormula::Iff(f1, f2) => {
                // Inside biconditional, subformulas appear in both polarities
                let inner_polarity = polarity.in_biconditional();
                let f1 = self.definitional_transform(*f1, inner_polarity, definitions);
                let f2 = self.definitional_transform(*f2, inner_polarity, definitions);

                // Check if either side contains quantifiers
                if self.contains_quantifier(&f1) || self.contains_quantifier(&f2) {
                    self.create_iff_definition(f1, f2, polarity, definitions)
                } else {
                    FOFFormula::Iff(Box::new(f1), Box::new(f2))
                }
            }

            FOFFormula::Xor(f1, f2) => {
                // XOR is ~(A <=> B), so subformulas are in biconditional context
                let inner_polarity = polarity.in_biconditional();
                let f1 = self.definitional_transform(*f1, inner_polarity, definitions);
                let f2 = self.definitional_transform(*f2, inner_polarity, definitions);

                if self.contains_quantifier(&f1) || self.contains_quantifier(&f2) {
                    // Create definition for (A <=> B) with flipped polarity, then negate
                    let def_atom = self.create_iff_definition(f1, f2, polarity.flip(), definitions);
                    FOFFormula::Not(Box::new(def_atom))
                } else {
                    FOFFormula::Xor(Box::new(f1), Box::new(f2))
                }
            }

            FOFFormula::Nand(f1, f2) => FOFFormula::Nand(
                Box::new(self.definitional_transform(*f1, polarity.flip(), definitions)),
                Box::new(self.definitional_transform(*f2, polarity.flip(), definitions)),
            ),

            FOFFormula::Nor(f1, f2) => FOFFormula::Nor(
                Box::new(self.definitional_transform(*f1, polarity.flip(), definitions)),
                Box::new(self.definitional_transform(*f2, polarity.flip(), definitions)),
            ),

            FOFFormula::Quantified(q, var, f) => FOFFormula::Quantified(
                q,
                var,
                Box::new(self.definitional_transform(*f, polarity, definitions)),
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
    /// - Positive: D => (A <=> B)
    /// - Negative: (A <=> B) => D
    /// - Both: full equivalence D <=> (A <=> B)
    fn create_iff_definition(
        &mut self,
        a: FOFFormula,
        b: FOFFormula,
        polarity: Polarity,
        definitions: &mut Vec<FOFFormula>,
    ) -> FOFFormula {
        // Collect free variables from both sides
        let mut free_vars: Vec<Variable> = a.free_variables().into_iter().collect();
        for v in b.free_variables() {
            if !free_vars.contains(&v) {
                free_vars.push(v);
            }
        }
        // Sort by variable ID for deterministic ordering
        free_vars.sort_by(|x, y| x.id.cmp(&y.id));

        // Create definition predicate D
        let def_name = format!("def{}", self.def_counter);
        self.def_counter += 1;

        let def_pred = PredicateSymbol::new(
            self.interner.intern_predicate(&def_name),
            free_vars.len() as u8,
        );

        let def_args: Vec<Term> = free_vars.iter().map(|v| Term::Variable(*v)).collect();
        let def_atom = FOFFormula::Atom(Atom {
            predicate: def_pred,
            args: def_args,
        });

        // Positive direction: D => (A <=> B)
        // As: D => (A => B) and D => (B => A)
        if polarity == Polarity::Positive || polarity == Polarity::Both {
            let pos_def1 = FOFFormula::Implies(
                Box::new(def_atom.clone()),
                Box::new(FOFFormula::Implies(Box::new(a.clone()), Box::new(b.clone()))),
            );
            let pos_def2 = FOFFormula::Implies(
                Box::new(def_atom.clone()),
                Box::new(FOFFormula::Implies(Box::new(b.clone()), Box::new(a.clone()))),
            );
            definitions.push(self.wrap_with_forall(pos_def1, &free_vars));
            definitions.push(self.wrap_with_forall(pos_def2, &free_vars));
        }

        // Negative direction: (A <=> B) => D
        // As: (A & B) => D and (~A & ~B) => D
        if polarity == Polarity::Negative || polarity == Polarity::Both {
            let neg_def1 = FOFFormula::Implies(
                Box::new(FOFFormula::And(Box::new(a.clone()), Box::new(b.clone()))),
                Box::new(def_atom.clone()),
            );
            let neg_def2 = FOFFormula::Implies(
                Box::new(FOFFormula::And(
                    Box::new(FOFFormula::Not(Box::new(a))),
                    Box::new(FOFFormula::Not(Box::new(b))),
                )),
                Box::new(def_atom.clone()),
            );
            definitions.push(self.wrap_with_forall(neg_def1, &free_vars));
            definitions.push(self.wrap_with_forall(neg_def2, &free_vars));
        }

        def_atom
    }

    /// Wrap a formula with universal quantifiers for the given variables
    fn wrap_with_forall(&self, formula: FOFFormula, vars: &[Variable]) -> FOFFormula {
        vars.iter().rev().fold(formula, |f, v| {
            FOFFormula::Quantified(Quantifier::Forall, *v, Box::new(f))
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
                        self.universal_vars.push(var);
                        // When body is done, wrap it in Forall and pop the var
                        stack.push(WorkItem::CombineForall(var));
                        stack.push(WorkItem::Process(*f));
                    }

                    FOFFormula::Quantified(Quantifier::Exists, var, f) => {
                        // Create Skolem function/constant
                        let skolem_name = format!("sk{}", self.skolem_counter);
                        let skolem_term = if self.universal_vars.is_empty() {
                            Term::Constant(Constant::new(
                                self.interner.intern_constant(&skolem_name),
                            ))
                        } else {
                            Term::Function(
                                FunctionSymbol::new(
                                    self.interner.intern_function(&skolem_name),
                                    self.universal_vars.len() as u8,
                                ),
                                self.universal_vars
                                    .iter()
                                    .map(|v| Term::Variable(*v))
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
                Term::Function(*f, new_args)
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
            self.check_memory()?;

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
                    literals.push(Literal::from_atom(atom, true));
                }

                FOFFormula::Not(inner) => match *inner {
                    FOFFormula::Atom(atom) => {
                        literals.push(Literal::from_atom(atom, false));
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

    /// Test context for building FOF formulas with interned symbols
    struct TestContext {
        interner: Interner,
    }

    impl TestContext {
        fn new() -> Self {
            TestContext {
                interner: Interner::new(),
            }
        }

        fn var(&mut self, name: &str) -> Variable {
            Variable::new(self.interner.intern_variable(name))
        }

        fn var_term(&mut self, name: &str) -> Term {
            Term::Variable(self.var(name))
        }

        fn pred(&mut self, name: &str, arity: u8) -> PredicateSymbol {
            PredicateSymbol::new(self.interner.intern_predicate(name), arity)
        }

        fn atom(&mut self, name: &str, args: Vec<Term>) -> Atom {
            let pred = self.pred(name, args.len() as u8);
            Atom { predicate: pred, args }
        }

        fn atom_formula(&mut self, name: &str, args: Vec<Term>) -> FOFFormula {
            FOFFormula::Atom(self.atom(name, args))
        }
    }

    #[test]
    fn test_simple_cnf_conversion() {
        let mut ctx = TestContext::new();

        // Test: P & Q -> two unit clauses
        let p = ctx.atom_formula("P", vec![]);
        let q = ctx.atom_formula("Q", vec![]);

        let formula = FOFFormula::And(Box::new(p), Box::new(q));
        let cnf = fof_to_cnf(formula, &mut ctx.interner).unwrap();

        assert_eq!(cnf.clauses.len(), 2);
        assert_eq!(cnf.clauses[0].literals.len(), 1);
        assert_eq!(cnf.clauses[1].literals.len(), 1);
    }

    #[test]
    fn test_skolemization() {
        let mut ctx = TestContext::new();

        // Test: ∃x.P(x) -> P(sk0)
        let x = ctx.var("X");
        let x_term = ctx.var_term("X");
        let p_x = ctx.atom_formula("P", vec![x_term]);

        let formula = FOFFormula::Quantified(Quantifier::Exists, x, Box::new(p_x));
        let cnf = fof_to_cnf(formula, &mut ctx.interner).unwrap();

        assert_eq!(cnf.clauses.len(), 1);
        assert_eq!(cnf.clauses[0].literals.len(), 1);

        // Check that the variable was replaced with a Skolem constant
        match &cnf.clauses[0].literals[0].args[0] {
            Term::Constant(c) => {
                let name = ctx.interner.resolve_constant(c.id);
                assert!(name.starts_with("sk"), "Expected Skolem constant, got: {}", name);
            }
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
        let mut ctx = TestContext::new();

        // Build: (∃X.p(X) <=> ∀Y.p(Y))
        let x = ctx.var("X");
        let y = ctx.var("Y");

        // p(X)
        let x_term = ctx.var_term("X");
        let p_x = ctx.atom_formula("p", vec![x_term]);

        // p(Y)
        let y_term = ctx.var_term("Y");
        let p_y = ctx.atom_formula("p", vec![y_term]);

        // ∃X.p(X)
        let exists_x_px = FOFFormula::Quantified(Quantifier::Exists, x, Box::new(p_x));

        // ∀Y.p(Y)
        let forall_y_py = FOFFormula::Quantified(Quantifier::Forall, y, Box::new(p_y));

        // (∃X.p(X) <=> ∀Y.p(Y))
        let formula = FOFFormula::Iff(Box::new(exists_x_px), Box::new(forall_y_py));

        let cnf = fof_to_cnf(formula, &mut ctx.interner).unwrap();

        // With definitional CNF, we should have:
        // 1. A definition predicate (def0 or similar)
        // 2. The formula should NOT have duplicated quantified subformulas
        let has_definition_predicate = cnf
            .clauses
            .iter()
            .any(|c| c.literals.iter().any(|l| {
                ctx.interner.resolve_predicate(l.predicate.id).starts_with("def")
            }));

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
                    l.args.iter().any(|arg| {
                        if let Term::Constant(c) = arg {
                            ctx.interner.resolve_constant(c.id).starts_with("sk")
                        } else {
                            false
                        }
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
