//! Literal selection strategies based on Hoder et al. "Selecting the selection" (2016)
//!
//! These strategies determine which literals in a clause are eligible
//! for inference rules like resolution and superposition.
//!
//! Selection strategies from the paper:
//! - Selection 0: Select all literals
//! - Selection 20: Select all maximal literals
//! - Selection 21: Select unique maximal, else negative with max weight, else all maximal
//! - Selection 22: Select negative literal with max weight, else all maximal

use crate::fol::{Clause, KBOConfig, Literal, Term, TermOrdering, VariableId, KBO};
use std::collections::HashSet;

/// Trait for literal selection strategies
pub trait LiteralSelector: Send + Sync {
    /// Select eligible literals from a clause
    /// Returns indices of selected literals
    fn select(&self, clause: &Clause) -> HashSet<usize>;

    /// Get the name of this selection strategy
    fn name(&self) -> &str;
}

/// Calculate the weight of a literal (symbol count)
fn literal_weight(literal: &Literal) -> usize {
    1 + literal
        .args
        .iter()
        .map(|term| term_symbol_count(term))
        .sum::<usize>()
}

/// Count the number of symbols in a term
fn term_symbol_count(term: &Term) -> usize {
    match term {
        Term::Variable(_) => 1,
        Term::Constant(_) => 1,
        Term::Function(_, args) => 1 + args.iter().map(|t| term_symbol_count(t)).sum::<usize>(),
    }
}

/// Compare two literals using KBO extension to atoms
/// Returns true if lit1 > lit2 in the literal ordering
///
/// KBO for terms: s > t if
/// 1. #(x,s) >= #(x,t) for all variables x AND |s| > |t|, OR
/// 2. #(x,s) >= #(x,t) for all variables x AND |s| = |t| AND one of:
///    2.1. s = g(...), t = h(...) and g >> h by precedence (alphabetic)
///    2.2. s = g(s1,...,sm), t = g(t1,...,tm) and lexicographically s > t
///
/// For atoms, we extend this by treating predicates like function symbols.
fn literal_greater(lit1: &Literal, lit2: &Literal, kbo: &KBO) -> bool {
    // First check variable condition: #(x, lit1) >= #(x, lit2) for all x
    let vars1 = count_literal_variables(lit1);
    let vars2 = count_literal_variables(lit2);

    let var_cond_satisfied = vars2.iter().all(|(var, count2)| {
        let count1 = vars1.get(var).copied().unwrap_or(0);
        count1 >= *count2
    });

    if !var_cond_satisfied {
        return false;
    }

    // Compare weights (sum of all symbol weights)
    let weight1 = literal_weight(lit1);
    let weight2 = literal_weight(lit2);

    if weight1 > weight2 {
        return true;  // Case 1: variable condition + greater weight
    }

    if weight1 < weight2 {
        return false;
    }

    // Equal weight - use lexicographic comparison (cases 2.2 and 2.3)
    // Compare predicates by ID (stable precedence) - case 2.2
    if lit1.predicate.id != lit2.predicate.id {
        return lit1.predicate.id > lit2.predicate.id;
    }

    // Same predicate - lexicographic comparison of arguments (case 2.3)
    for (arg1, arg2) in lit1.args.iter().zip(lit2.args.iter()) {
        match kbo.compare(arg1, arg2) {
            TermOrdering::Greater => return true,
            TermOrdering::Less => return false,
            TermOrdering::Equal | TermOrdering::Incomparable => continue,
        }
    }

    // All compared arguments are equal/incomparable
    false
}

/// Count occurrences of each variable in a literal
fn count_literal_variables(lit: &Literal) -> std::collections::HashMap<VariableId, usize> {
    let mut counts = std::collections::HashMap::new();
    for arg in &lit.args {
        count_term_variables(arg, &mut counts);
    }
    counts
}

/// Recursively count variables in a term
fn count_term_variables(term: &Term, counts: &mut std::collections::HashMap<VariableId, usize>) {
    match term {
        Term::Variable(v) => {
            *counts.entry(v.id).or_insert(0) += 1;
        }
        Term::Constant(_) => {}
        Term::Function(_, args) => {
            for arg in args {
                count_term_variables(arg, counts);
            }
        }
    }
}

/// Find all maximal literals in a clause
fn find_maximal_literals(clause: &Clause, kbo: &KBO) -> HashSet<usize> {
    if clause.literals.is_empty() {
        return HashSet::new();
    }

    let mut maximal = HashSet::new();

    for i in 0..clause.literals.len() {
        let mut is_maximal = true;
        for j in 0..clause.literals.len() {
            if i != j && literal_greater(&clause.literals[j], &clause.literals[i], kbo) {
                is_maximal = false;
                break;
            }
        }
        if is_maximal {
            maximal.insert(i);
        }
    }

    maximal
}

/// Check if there is exactly one maximal literal
fn has_unique_maximal(clause: &Clause, kbo: &KBO) -> Option<usize> {
    let maximal = find_maximal_literals(clause, kbo);
    if maximal.len() == 1 {
        maximal.into_iter().next()
    } else {
        None
    }
}

/// Find a negative literal with maximum weight (if any)
fn find_max_weight_negative(clause: &Clause) -> Option<usize> {
    let negative_literals: Vec<(usize, usize)> = clause
        .literals
        .iter()
        .enumerate()
        .filter(|(_, lit)| !lit.polarity)
        .map(|(idx, lit)| (idx, literal_weight(lit)))
        .collect();

    if negative_literals.is_empty() {
        return None;
    }

    let max_weight = negative_literals.iter().map(|(_, w)| w).max().unwrap();
    negative_literals
        .iter()
        .filter(|(_, w)| w == max_weight)
        .map(|(idx, _)| *idx)
        .next()
}

// ============================================================================
// Selection 0: Select all literals
// ============================================================================

/// Select all literals - all literals are eligible for inference
///
/// Vampire selection 0: no literal selection, all literals participate.
pub struct SelectAll;

impl LiteralSelector for SelectAll {
    fn select(&self, clause: &Clause) -> HashSet<usize> {
        (0..clause.literals.len()).collect()
    }

    fn name(&self) -> &str {
        "sel0"
    }
}

// ============================================================================
// Selection 20: Select all maximal literals
// ============================================================================

/// Select all maximal literals in the clause
///
/// Vampire selection 20: select all literals that are maximal in the ordering.
pub struct SelectMaximal {
    kbo: KBO,
}

impl SelectMaximal {
    pub fn new() -> Self {
        SelectMaximal {
            kbo: KBO::new(KBOConfig::default()),
        }
    }

    pub fn with_kbo(kbo: KBO) -> Self {
        SelectMaximal { kbo }
    }
}

impl Default for SelectMaximal {
    fn default() -> Self {
        Self::new()
    }
}

impl LiteralSelector for SelectMaximal {
    fn select(&self, clause: &Clause) -> HashSet<usize> {
        find_maximal_literals(clause, &self.kbo)
    }

    fn name(&self) -> &str {
        "sel20"
    }
}

// ============================================================================
// Selection 22: Select max-weight negative literal, else all maximal
// ============================================================================

/// Select a negative literal with maximum weight if one exists,
/// otherwise select all maximal literals.
///
/// Vampire selection 22.
pub struct SelectNegMaxWeightOrMaximal {
    kbo: KBO,
}

impl SelectNegMaxWeightOrMaximal {
    pub fn new() -> Self {
        SelectNegMaxWeightOrMaximal {
            kbo: KBO::new(KBOConfig::default()),
        }
    }

    pub fn with_kbo(kbo: KBO) -> Self {
        SelectNegMaxWeightOrMaximal { kbo }
    }
}

impl Default for SelectNegMaxWeightOrMaximal {
    fn default() -> Self {
        Self::new()
    }
}

impl LiteralSelector for SelectNegMaxWeightOrMaximal {
    fn select(&self, clause: &Clause) -> HashSet<usize> {
        // First try to select negative literal with max weight
        if let Some(idx) = find_max_weight_negative(clause) {
            let mut selected = HashSet::new();
            selected.insert(idx);
            return selected;
        }

        // Fall back to all maximal literals
        find_maximal_literals(clause, &self.kbo)
    }

    fn name(&self) -> &str {
        "sel22"
    }
}

// ============================================================================
// Selection 21: Unique maximal, else max-weight negative, else all maximal
// ============================================================================

/// Select a unique maximal literal if one exists,
/// otherwise select a negative literal with maximum weight if one exists,
/// otherwise select all maximal literals.
///
/// Vampire selection 21.
pub struct SelectUniqueMaximalOrNegOrMaximal {
    kbo: KBO,
}

impl SelectUniqueMaximalOrNegOrMaximal {
    pub fn new() -> Self {
        SelectUniqueMaximalOrNegOrMaximal {
            kbo: KBO::new(KBOConfig::default()),
        }
    }

    pub fn with_kbo(kbo: KBO) -> Self {
        SelectUniqueMaximalOrNegOrMaximal { kbo }
    }
}

impl Default for SelectUniqueMaximalOrNegOrMaximal {
    fn default() -> Self {
        Self::new()
    }
}

impl LiteralSelector for SelectUniqueMaximalOrNegOrMaximal {
    fn select(&self, clause: &Clause) -> HashSet<usize> {
        // First try unique maximal
        if let Some(idx) = has_unique_maximal(clause, &self.kbo) {
            let mut selected = HashSet::new();
            selected.insert(idx);
            return selected;
        }

        // Then try negative literal with max weight
        if let Some(idx) = find_max_weight_negative(clause) {
            let mut selected = HashSet::new();
            selected.insert(idx);
            return selected;
        }

        // Fall back to all maximal literals
        find_maximal_literals(clause, &self.kbo)
    }

    fn name(&self) -> &str {
        "sel21"
    }
}

// ============================================================================
// Legacy aliases for backwards compatibility
// ============================================================================

/// Alias for SelectAll (legacy name)
pub type SelectMaxWeight = SelectMaximal;

/// Alias for SelectNegMaxWeightOrMaximal (legacy name)
pub type SelectLargestNegative = SelectNegMaxWeightOrMaximal;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fol::{Clause, Constant, FunctionSymbol, Interner, Literal, PredicateSymbol, Term, Variable};

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

        fn literal(&mut self, pred_name: &str, args: Vec<Term>, positive: bool) -> Literal {
            let pred = self.pred(pred_name, args.len() as u8);
            if positive {
                Literal::positive(pred, args)
            } else {
                Literal::negative(pred, args)
            }
        }
    }

    fn make_clause(literals: Vec<Literal>) -> Clause {
        Clause::new(literals)
    }

    #[test]
    fn test_select_all() {
        let mut ctx = TestContext::new();

        // Build args first to avoid nested mutable borrows
        let x = ctx.var("X");
        let lit0 = ctx.literal("P", vec![x], true);

        let a = ctx.const_("a");
        let lit1 = ctx.literal("Q", vec![a], false);

        let y = ctx.var("Y");
        let f_y = ctx.func("f", vec![y]);
        let lit2 = ctx.literal("R", vec![f_y], true);

        let clause = make_clause(vec![lit0, lit1, lit2]);

        let selector = SelectAll;
        let selected = selector.select(&clause);

        assert_eq!(selected.len(), 3);
        assert!(selected.contains(&0));
        assert!(selected.contains(&1));
        assert!(selected.contains(&2));
    }

    #[test]
    fn test_select_maximal() {
        // P(a) ∨ Q(f(g(a))) - both ground, Q has higher weight so Q > P
        let mut ctx = TestContext::new();

        let a = ctx.const_("a");
        let lit0 = ctx.literal("P", vec![a], true);

        let a2 = ctx.const_("a");
        let inner_g = ctx.func("g", vec![a2]);
        let outer_f = ctx.func("f", vec![inner_g]);
        let lit1 = ctx.literal("Q", vec![outer_f], true);

        let clause = make_clause(vec![lit0, lit1]);

        let selector = SelectMaximal::new();
        let selected = selector.select(&clause);

        // Q(f(g(a))) should be uniquely maximal (higher weight)
        assert_eq!(selected.len(), 1);
        assert!(selected.contains(&1));
    }

    #[test]
    fn test_select_maximal_with_variables() {
        // P(X) ∨ Q(f(g(a))) - incomparable due to variable condition, both maximal
        let mut ctx = TestContext::new();

        let x = ctx.var("X");
        let lit0 = ctx.literal("P", vec![x], true);

        let a = ctx.const_("a");
        let inner_g = ctx.func("g", vec![a]);
        let outer_f = ctx.func("f", vec![inner_g]);
        let lit1 = ctx.literal("Q", vec![outer_f], true);

        let clause = make_clause(vec![lit0, lit1]);

        let selector = SelectMaximal::new();
        let selected = selector.select(&clause);

        // Both are maximal: Q can't be > P (Q doesn't contain X), P can't be > Q (lower weight)
        assert_eq!(selected.len(), 2);
        assert!(selected.contains(&0));
        assert!(selected.contains(&1));
    }

    #[test]
    fn test_select_neg_max_weight_or_maximal() {
        // P(X) ∨ ~Q(f(a)) ∨ ~R(a) - should select ~Q(f(a)) as largest negative
        let mut ctx = TestContext::new();

        let x = ctx.var("X");
        let lit0 = ctx.literal("P", vec![x], true);

        let a1 = ctx.const_("a");
        let f_a = ctx.func("f", vec![a1]);
        let lit1 = ctx.literal("Q", vec![f_a], false);

        let a2 = ctx.const_("a");
        let lit2 = ctx.literal("R", vec![a2], false);

        let clause = make_clause(vec![lit0, lit1, lit2]);

        let selector = SelectNegMaxWeightOrMaximal::new();
        let selected = selector.select(&clause);

        // Should select ~Q(f(a)) (index 1) as it's the largest negative
        assert_eq!(selected.len(), 1);
        assert!(selected.contains(&1));
    }

    #[test]
    fn test_select_neg_max_weight_fallback() {
        // P(a) ∨ Q(f(a)) - all positive, should fall back to maximal
        let mut ctx = TestContext::new();

        let a = ctx.const_("a");
        let lit0 = ctx.literal("P", vec![a], true);

        let a2 = ctx.const_("a");
        let f_a = ctx.func("f", vec![a2]);
        let lit1 = ctx.literal("Q", vec![f_a], true);

        let clause = make_clause(vec![lit0, lit1]);

        let selector = SelectNegMaxWeightOrMaximal::new();
        let selected = selector.select(&clause);

        // Should select Q(f(a)) as uniquely maximal since no negatives
        assert_eq!(selected.len(), 1);
        assert!(selected.contains(&1));
    }

    #[test]
    fn test_select_unique_maximal() {
        // P(a) ∨ Q(f(g(h(a)))) - Q is uniquely maximal (ground terms, higher weight)
        let mut ctx = TestContext::new();

        let a = ctx.const_("a");
        let lit0 = ctx.literal("P", vec![a], true);

        let a2 = ctx.const_("a");
        let h_a = ctx.func("h", vec![a2]);
        let g_h_a = ctx.func("g", vec![h_a]);
        let f_g_h_a = ctx.func("f", vec![g_h_a]);
        let lit1 = ctx.literal("Q", vec![f_g_h_a], true);

        let clause = make_clause(vec![lit0, lit1]);

        let selector = SelectUniqueMaximalOrNegOrMaximal::new();
        let selected = selector.select(&clause);

        // Should select Q as unique maximal
        assert_eq!(selected.len(), 1);
        assert!(selected.contains(&1));
    }

    #[test]
    fn test_select_unique_maximal_fallback_to_negative() {
        // P(f(X)) ∨ Q(f(Y)) ∨ ~R(c) - P and Q have incomparable terms due to different variables,
        // so both are maximal. Should fall back to selecting the negative literal.
        let mut ctx = TestContext::new();

        let x = ctx.var("X");
        let f_x = ctx.func("f", vec![x]);
        let lit0 = ctx.literal("P", vec![f_x], true);

        let y = ctx.var("Y");
        let f_y = ctx.func("f", vec![y]);
        let lit1 = ctx.literal("Q", vec![f_y], true);

        let c = ctx.const_("c");
        let lit2 = ctx.literal("R", vec![c], false);

        let clause = make_clause(vec![lit0, lit1, lit2]);

        let selector = SelectUniqueMaximalOrNegOrMaximal::new();
        let selected = selector.select(&clause);

        // P(f(X)) and Q(f(Y)) should both be maximal (incomparable due to different variables)
        // so we fall back to selecting the negative literal ~R(c) at index 2
        assert_eq!(selected.len(), 1);
        assert!(selected.contains(&2));
    }

    #[test]
    fn test_select_one_negative_when_equal_weight() {
        // ~P(f(X)) ∨ ~Q(f(Y)) - two negatives with equal weight
        // Only one should be selected (completeness requires selecting A negative, not all)
        let mut ctx = TestContext::new();

        let x = ctx.var("X");
        let f_x = ctx.func("f", vec![x]);
        let lit0 = ctx.literal("P", vec![f_x], false);

        let y = ctx.var("Y");
        let f_y = ctx.func("f", vec![y]);
        let lit1 = ctx.literal("Q", vec![f_y], false);

        let clause = make_clause(vec![lit0, lit1]);

        let selector = SelectUniqueMaximalOrNegOrMaximal::new();
        let selected = selector.select(&clause);

        // One negative is selected (the first one with max weight)
        assert_eq!(selected.len(), 1);
        assert!(selected.contains(&0) || selected.contains(&1));
    }
}
