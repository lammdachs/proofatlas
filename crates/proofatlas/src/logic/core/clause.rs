//! Clauses and CNF formulas

use crate::logic::interner::{Interner, VariableId};
use super::literal::Literal;
use super::term::Term;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;

/// Role of a clause in the proof (from TPTP or derived)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ClauseRole {
    /// Axiom from the problem
    #[default]
    Axiom,
    /// Hypothesis
    Hypothesis,
    /// Definition
    Definition,
    /// Negated conjecture (goal)
    NegatedConjecture,
    /// Derived clause (from inference)
    Derived,
}

impl ClauseRole {
    /// Convert to a numeric value for ML features
    pub fn to_feature_value(&self) -> f32 {
        match self {
            ClauseRole::Axiom => 0.0,
            ClauseRole::Hypothesis => 1.0,
            ClauseRole::Definition => 2.0,
            ClauseRole::NegatedConjecture => 3.0,
            ClauseRole::Derived => 4.0,
        }
    }

    /// Convert from a TPTP role string to ClauseRole
    pub fn from_tptp_role(role: &str) -> Self {
        match role {
            "axiom" | "lemma" | "theorem" | "corollary" | "assumption" => ClauseRole::Axiom,
            "hypothesis" => ClauseRole::Hypothesis,
            "definition" => ClauseRole::Definition,
            "negated_conjecture" => ClauseRole::NegatedConjecture,
            "conjecture" => ClauseRole::NegatedConjecture, // Will be negated in processing
            _ => ClauseRole::Axiom,
        }
    }
}

/// A clause (disjunction of literals)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Clause {
    pub literals: Vec<Literal>,
    pub id: Option<usize>,
    /// Role of the clause (axiom, hypothesis, negated conjecture, derived)
    pub role: ClauseRole,
    /// Age of the clause (derivation step when it was created, 0 for input clauses)
    pub age: usize,
    /// Derivation rule ID (0=input, 1=resolution, 2=factoring, 3=superposition,
    /// 4=equality_resolution, 5=equality_factoring, 6=demodulation)
    pub derivation_rule: u8,
}

/// A CNF formula (conjunction of clauses)
#[derive(Debug, Clone)]
pub struct CNFFormula {
    pub clauses: Vec<Clause>,
}

impl Clause {
    /// Create a new clause from literals
    pub fn new(literals: Vec<Literal>) -> Self {
        Clause {
            literals,
            id: None,
            role: ClauseRole::default(),
            age: 0,
            derivation_rule: 0,
        }
    }

    /// Create a new clause with a specific role
    pub fn with_role(literals: Vec<Literal>, role: ClauseRole) -> Self {
        Clause {
            literals,
            id: None,
            role,
            age: 0,
            derivation_rule: 0,
        }
    }

    /// Create a derived clause with age
    pub fn derived(literals: Vec<Literal>, age: usize) -> Self {
        Clause {
            literals,
            id: None,
            role: ClauseRole::Derived,
            age,
            derivation_rule: 0,
        }
    }

    /// Check if this clause is empty (contradiction)
    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }

    /// Check if this clause is a tautology (needs interner for equality check)
    pub fn is_tautology(&self, interner: &Interner) -> bool {
        // Check for complementary literals
        for i in 0..self.literals.len() {
            for j in (i + 1)..self.literals.len() {
                if self.literals[i].predicate == self.literals[j].predicate
                    && self.literals[i].args == self.literals[j].args
                    && self.literals[i].polarity != self.literals[j].polarity
                {
                    return true;
                }
            }
        }

        // Check for reflexive equality
        for lit in &self.literals {
            if lit.polarity && lit.is_equality(interner) {
                if let [ref t1, ref t2] = lit.args.as_slice() {
                    if t1 == t2 {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Count the total number of symbols in this clause
    pub fn symbol_count(&self) -> usize {
        self.literals
            .iter()
            .map(|lit| {
                // Count predicate symbol
                1 + lit
                    .args
                    .iter()
                    .map(Self::term_symbol_count)
                    .sum::<usize>()
            })
            .sum()
    }

    fn term_symbol_count(term: &Term) -> usize {
        match term {
            Term::Variable(_) => 1,
            Term::Constant(_) => 1,
            Term::Function(_, args) => {
                1 + args
                    .iter()
                    .map(Self::term_symbol_count)
                    .sum::<usize>()
            }
        }
    }

    /// Maximum term nesting depth across all literals
    pub fn max_depth(&self) -> usize {
        self.literals
            .iter()
            .flat_map(|lit| lit.args.iter())
            .map(Self::term_depth)
            .max()
            .unwrap_or(0)
    }

    fn term_depth(term: &Term) -> usize {
        match term {
            Term::Variable(_) | Term::Constant(_) => 0,
            Term::Function(_, args) => {
                1 + args.iter().map(Self::term_depth).max().unwrap_or(0)
            }
        }
    }

    /// Count of distinct function, constant, and predicate symbols
    pub fn distinct_symbol_count(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        for lit in &self.literals {
            // Predicate symbol (tag with high bit to distinguish from function/constant IDs)
            seen.insert(lit.predicate.id.as_u32() as u64 | (1u64 << 32));
            for arg in &lit.args {
                Self::collect_distinct_symbols(arg, &mut seen);
            }
        }
        seen.len()
    }

    fn collect_distinct_symbols(term: &Term, seen: &mut std::collections::HashSet<u64>) {
        match term {
            Term::Variable(_) => {}
            Term::Constant(c) => { seen.insert(c.id.as_u32() as u64); }
            Term::Function(f, args) => {
                seen.insert(f.id.as_u32() as u64);
                for arg in args {
                    Self::collect_distinct_symbols(arg, seen);
                }
            }
        }
    }

    /// Total variable occurrences across all literals
    pub fn variable_count(&self) -> usize {
        self.literals
            .iter()
            .flat_map(|lit| lit.args.iter())
            .map(Self::term_variable_count)
            .sum()
    }

    fn term_variable_count(term: &Term) -> usize {
        match term {
            Term::Variable(_) => 1,
            Term::Constant(_) => 0,
            Term::Function(_, args) => {
                args.iter().map(Self::term_variable_count).sum()
            }
        }
    }

    /// Count of distinct variables
    pub fn distinct_variable_count(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        for lit in &self.literals {
            for arg in &lit.args {
                Self::collect_distinct_variables(arg, &mut seen);
            }
        }
        seen.len()
    }

    fn collect_distinct_variables(term: &Term, seen: &mut std::collections::HashSet<u32>) {
        match term {
            Term::Variable(v) => { seen.insert(v.id.as_u32()); }
            Term::Constant(_) => {}
            Term::Function(_, args) => {
                for arg in args {
                    Self::collect_distinct_variables(arg, seen);
                }
            }
        }
    }

    /// Map a rule name string to its numeric ID for ML features.
    ///
    /// Must match the mapping in `python_bindings.rs::compute_clause_features_flat`.
    pub fn rule_name_to_id(rule_name: &str) -> u8 {
        match rule_name {
            "Input" => 0,
            "Resolution" => 1,
            "Factoring" => 2,
            "Superposition" => 3,
            "EqualityResolution" => 4,
            "EqualityFactoring" => 5,
            "Demodulation" | "Forward Demodulation" | "Backward Demodulation" => 6,
            _ => 0, // Unknown rules default to input
        }
    }

    /// Normalize variables to canonical sequential form (X0, X1, X2, ...)
    /// based on first-occurrence order (left-to-right, depth-first).
    ///
    /// This ensures α-equivalent clauses have identical representations,
    /// prevents cascading variable names from binary inference (X_1_1_1...),
    /// and keeps interner size bounded.
    pub fn normalize_variables(&mut self, interner: &mut Interner) {
        // Phase 1: Collect variables in first-occurrence order
        let mut seen = HashMap::new();
        let mut order = Vec::new();
        for lit in &self.literals {
            for arg in &lit.args {
                collect_variable_order(arg, &mut seen, &mut order);
            }
        }
        if order.is_empty() {
            return;
        }

        // Phase 2: Build old_id → new_id mapping
        let mut remap = HashMap::with_capacity(order.len());
        let mut all_canonical = true;
        for (i, old_id) in order.iter().enumerate() {
            let new_id = interner.intern_variable(&format!("X{}", i));
            if *old_id != new_id {
                all_canonical = false;
            }
            remap.insert(*old_id, new_id);
        }
        if all_canonical {
            return;
        }

        // Phase 3: Apply remapping in-place
        for lit in &mut self.literals {
            for arg in &mut lit.args {
                remap_term_variables(arg, &remap);
            }
        }
    }
}

/// Collect variable IDs in depth-first, left-to-right first-occurrence order.
fn collect_variable_order(
    term: &Term,
    seen: &mut HashMap<VariableId, ()>,
    order: &mut Vec<VariableId>,
) {
    match term {
        Term::Variable(v) => {
            if seen.insert(v.id, ()).is_none() {
                order.push(v.id);
            }
        }
        Term::Constant(_) => {}
        Term::Function(_, args) => {
            for arg in args {
                collect_variable_order(arg, seen, order);
            }
        }
    }
}

/// Remap all variable IDs in a term in-place.
fn remap_term_variables(term: &mut Term, remap: &HashMap<VariableId, VariableId>) {
    match term {
        Term::Variable(v) => {
            if let Some(&new_id) = remap.get(&v.id) {
                v.id = new_id;
            }
        }
        Term::Constant(_) => {}
        Term::Function(_, args) => {
            for arg in args {
                remap_term_variables(arg, remap);
            }
        }
    }
}

impl Clause {
    /// Format this clause with an interner for name resolution
    pub fn display<'a>(&'a self, interner: &'a Interner) -> ClauseDisplay<'a> {
        ClauseDisplay {
            clause: self,
            interner,
        }
    }
}

/// Display wrapper for Clause that includes an interner for name resolution
pub struct ClauseDisplay<'a> {
    clause: &'a Clause,
    interner: &'a Interner,
}

impl<'a> fmt::Display for ClauseDisplay<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.clause.is_empty() {
            write!(f, "⊥")
        } else {
            for (i, lit) in self.clause.literals.iter().enumerate() {
                if i > 0 {
                    write!(f, " ∨ ")?;
                }
                write!(f, "{}", lit.display(self.interner))?;
            }
            Ok(())
        }
    }
}

// Display implementation that shows IDs (for debugging without interner)
impl fmt::Display for Clause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            write!(f, "⊥")
        } else {
            for (i, lit) in self.literals.iter().enumerate() {
                if i > 0 {
                    write!(f, " ∨ ")?;
                }
                write!(f, "{}", lit)?;
            }
            Ok(())
        }
    }
}

// =============================================================================
// ClauseKey - Structural hash key for clause deduplication
// =============================================================================

/// A sortable representation of a literal for use in ClauseKey.
/// Sorting order: polarity (negative first), then predicate ID, then args.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct LiteralKey {
    /// Polarity: false (negative) sorts before true (positive)
    polarity: bool,
    /// Predicate ID
    predicate_id: u32,
    /// Predicate arity
    predicate_arity: u8,
    /// Serialized arguments (for consistent ordering)
    args: Vec<TermKey>,
}

/// A sortable representation of a term for use in LiteralKey.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum TermKey {
    /// Variable with ID
    Variable(u32),
    /// Constant with ID
    Constant(u32),
    /// Function with ID, arity, and args
    Function(u32, u8, Vec<TermKey>),
}

impl TermKey {
    fn from_term(term: &Term) -> Self {
        match term {
            Term::Variable(v) => TermKey::Variable(v.id.as_u32()),
            Term::Constant(c) => TermKey::Constant(c.id.as_u32()),
            Term::Function(f, args) => {
                TermKey::Function(
                    f.id.as_u32(),
                    f.arity,
                    args.iter().map(TermKey::from_term).collect(),
                )
            }
        }
    }
}

impl LiteralKey {
    fn from_literal(lit: &Literal) -> Self {
        LiteralKey {
            polarity: lit.polarity,
            predicate_id: lit.predicate.id.as_u32(),
            predicate_arity: lit.predicate.arity,
            args: lit.args.iter().map(TermKey::from_term).collect(),
        }
    }
}

/// A structural key for a clause that enables O(1) duplicate detection.
///
/// The key sorts literals in a canonical order to ensure that logically equivalent
/// clauses (differing only in literal order) produce the same key. This replaces
/// string-based hashing (`format!("{}", clause)`) which was a major bottleneck.
///
/// Sorting order for literals: polarity (negative first), then predicate ID, then args.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ClauseKey {
    /// Sorted literal keys
    literals: Vec<LiteralKey>,
}

impl ClauseKey {
    /// Create a ClauseKey from a clause.
    ///
    /// The literals are sorted to produce a canonical representation.
    pub fn from_clause(clause: &Clause) -> Self {
        let mut literals: Vec<LiteralKey> = clause
            .literals
            .iter()
            .map(LiteralKey::from_literal)
            .collect();
        literals.sort();
        ClauseKey { literals }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::{Constant, FunctionSymbol, PredicateSymbol, Variable};

    struct TestContext {
        interner: Interner,
    }

    impl TestContext {
        fn new() -> Self {
            TestContext {
                interner: Interner::new(),
            }
        }

        fn var(&mut self, name: &str) -> Term {
            let id = self.interner.intern_variable(name);
            Term::Variable(Variable::new(id))
        }

        fn const_(&mut self, name: &str) -> Term {
            let id = self.interner.intern_constant(name);
            Term::Constant(Constant::new(id))
        }

        fn func(&mut self, name: &str, args: Vec<Term>) -> Term {
            let id = self.interner.intern_function(name);
            Term::Function(FunctionSymbol::new(id, args.len() as u8), args)
        }

        fn pred(&mut self, name: &str, arity: u8) -> PredicateSymbol {
            let id = self.interner.intern_predicate(name);
            PredicateSymbol::new(id, arity)
        }
    }

    #[test]
    fn test_normalize_simple() {
        let mut ctx = TestContext::new();
        let p = ctx.pred("P", 2);
        let y = ctx.var("Y");
        let x = ctx.var("X");
        // P(Y, X) → P(X0, X1)
        let mut clause = Clause::new(vec![Literal::positive(p, vec![y, x])]);
        clause.normalize_variables(&mut ctx.interner);

        let x0 = ctx.var("X0");
        let x1 = ctx.var("X1");
        let expected = Clause::new(vec![Literal::positive(p, vec![x0, x1])]);
        assert_eq!(clause.literals, expected.literals);
    }

    #[test]
    fn test_normalize_with_functions() {
        let mut ctx = TestContext::new();
        let p = ctx.pred("P", 2);
        let z = ctx.var("Z");
        let y = ctx.var("Y");
        let fz = ctx.func("f", vec![z.clone()]);
        let gyz = ctx.func("g", vec![y, z]);
        // P(f(Z), g(Y, Z)) → P(f(X0), g(X1, X0))
        let mut clause = Clause::new(vec![Literal::positive(p, vec![fz, gyz])]);
        clause.normalize_variables(&mut ctx.interner);

        let x0 = ctx.var("X0");
        let x1 = ctx.var("X1");
        let fx0 = ctx.func("f", vec![x0.clone()]);
        let gx1x0 = ctx.func("g", vec![x1, x0]);
        let expected = Clause::new(vec![Literal::positive(p, vec![fx0, gx1x0])]);
        assert_eq!(clause.literals, expected.literals);
    }

    #[test]
    fn test_normalize_cascading_names() {
        let mut ctx = TestContext::new();
        let p = ctx.pred("P", 2);
        let x_1_1 = ctx.var("X_1_1");
        let x_1 = ctx.var("X_1");
        // P(X_1_1, X_1) → P(X0, X1)
        let mut clause = Clause::new(vec![Literal::positive(p, vec![x_1_1, x_1])]);
        clause.normalize_variables(&mut ctx.interner);

        let x0 = ctx.var("X0");
        let x1 = ctx.var("X1");
        let expected = Clause::new(vec![Literal::positive(p, vec![x0, x1])]);
        assert_eq!(clause.literals, expected.literals);
    }

    #[test]
    fn test_normalize_ground_clause() {
        let mut ctx = TestContext::new();
        let p = ctx.pred("P", 2);
        let a = ctx.const_("a");
        let b = ctx.const_("b");
        let mut clause = Clause::new(vec![Literal::positive(p, vec![a.clone(), b.clone()])]);
        clause.normalize_variables(&mut ctx.interner);

        let expected = Clause::new(vec![Literal::positive(p, vec![a, b])]);
        assert_eq!(clause.literals, expected.literals);
    }

    #[test]
    fn test_normalize_idempotent() {
        let mut ctx = TestContext::new();
        let p = ctx.pred("P", 2);
        let y = ctx.var("Y");
        let x = ctx.var("X");
        let mut clause = Clause::new(vec![Literal::positive(p, vec![y, x])]);
        clause.normalize_variables(&mut ctx.interner);
        let after_first = clause.literals.clone();
        clause.normalize_variables(&mut ctx.interner);
        assert_eq!(clause.literals, after_first);
    }

    #[test]
    fn test_normalize_multiple_literals() {
        let mut ctx = TestContext::new();
        let p = ctx.pred("P", 1);
        let q = ctx.pred("Q", 2);
        let y = ctx.var("Y");
        let x = ctx.var("X");
        // P(Y) | Q(X, Y) → P(X0) | Q(X1, X0)
        let mut clause = Clause::new(vec![
            Literal::positive(p, vec![y.clone()]),
            Literal::positive(q, vec![x, y]),
        ]);
        clause.normalize_variables(&mut ctx.interner);

        let x0 = ctx.var("X0");
        let x1 = ctx.var("X1");
        let expected = Clause::new(vec![
            Literal::positive(p, vec![x0.clone()]),
            Literal::positive(q, vec![x1, x0]),
        ]);
        assert_eq!(clause.literals, expected.literals);
    }
}
