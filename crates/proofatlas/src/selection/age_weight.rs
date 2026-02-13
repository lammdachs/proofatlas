//! Age-weight heuristic clause selector
//!
//! This implements the classic age-weight ratio heuristic used in theorem provers.
//! It alternates between selecting the oldest clause (FIFO) and the lightest clause
//! (by symbol count) based on a configurable probability.

use crate::logic::Clause;
use indexmap::IndexSet;
use std::sync::Arc;

use super::ClauseSelector;

/// Age-weight heuristic clause selector.
///
/// This selector balances exploration (older clauses) with exploitation (lighter clauses).
/// The `age_probability` parameter controls the ratio:
/// - Higher values favor older clauses (more FIFO-like, better for completeness)
/// - Lower values favor lighter clauses (more weight-based, faster but may miss proofs)
///
/// A typical value is 0.5 (equal balance).
pub struct AgeWeightSelector {
    /// Probability of selecting by age (FIFO) vs weight
    age_probability: f64,
    /// Random number generator state (simple LCG)
    rng_state: u64,
}

impl AgeWeightSelector {
    /// Create a new age-weight selector with the given age probability.
    ///
    /// # Arguments
    /// * `age_probability` - Probability in [0, 1] of selecting the oldest clause.
    ///   The remaining probability selects the lightest clause.
    pub fn new(age_probability: f64) -> Self {
        Self {
            age_probability: age_probability.clamp(0.0, 1.0),
            rng_state: 12345,
        }
    }

    /// Create a selector with default 50% age probability.
    pub fn default_ratio() -> Self {
        Self::new(0.5)
    }

    /// Generate a random float in [0, 1)
    fn next_random(&mut self) -> f64 {
        // Simple LCG: x_{n+1} = (a * x_n + c) mod m
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        (self.rng_state >> 33) as f64 / (1u64 << 31) as f64
    }

    /// Calculate the weight of a clause (symbol count).
    fn clause_weight(clause: &Clause) -> usize {
        clause
            .literals
            .iter()
            .map(|lit| {
                // Count predicate symbol + weight of all argument terms
                1 + lit.args.iter().map(Self::term_weight).sum::<usize>()
            })
            .sum()
    }

    /// Calculate the weight of a term recursively.
    fn term_weight(term: &crate::logic::Term) -> usize {
        match term {
            crate::logic::Term::Variable(_) => 1,
            crate::logic::Term::Constant(_) => 1,
            crate::logic::Term::Function(_, args) => 1 + args.iter().map(Self::term_weight).sum::<usize>(),
        }
    }

    /// Find the position of the lightest clause.
    fn find_lightest(&self, unprocessed: &IndexSet<usize>, clauses: &[Arc<Clause>]) -> usize {
        unprocessed
            .iter()
            .enumerate()
            .min_by_key(|(_, &clause_idx)| Self::clause_weight(&clauses[clause_idx]))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }
}

impl ClauseSelector for AgeWeightSelector {
    fn select(&mut self, unprocessed: &mut IndexSet<usize>, clauses: &[Arc<Clause>]) -> Option<usize> {
        if unprocessed.is_empty() {
            return None;
        }

        let r = self.next_random();

        if r < self.age_probability {
            // Select oldest (FIFO - first in set, preserves insertion order)
            unprocessed.shift_remove_index(0)
        } else {
            // Select lightest
            let lightest_pos = self.find_lightest(unprocessed, clauses);
            unprocessed.shift_remove_index(lightest_pos)
        }
    }

    fn name(&self) -> &str {
        "AgeWeight"
    }
}

impl Default for AgeWeightSelector {
    fn default() -> Self {
        Self::default_ratio()
    }
}

/// FIFO clause selector: always selects the oldest clause.
///
/// Equivalent to `AgeWeightSelector` with `age_probability = 1.0`.
/// Guarantees fairness (every clause is eventually selected), which
/// is important for completeness.
pub struct FIFOSelector(AgeWeightSelector);

impl FIFOSelector {
    pub fn new() -> Self {
        FIFOSelector(AgeWeightSelector::new(1.0))
    }
}

impl Default for FIFOSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl ClauseSelector for FIFOSelector {
    fn select(&mut self, unprocessed: &mut IndexSet<usize>, clauses: &[Arc<Clause>]) -> Option<usize> {
        self.0.select(unprocessed, clauses)
    }

    fn name(&self) -> &str {
        "FIFO"
    }

    fn reset(&mut self) {
        self.0.reset();
    }
}

/// Weight-based clause selector: always selects the lightest clause.
///
/// Equivalent to `AgeWeightSelector` with `age_probability = 0.0`.
/// Aggressively pursues short clauses, which tends to find proofs faster
/// but may sacrifice completeness.
pub struct WeightSelector(AgeWeightSelector);

impl WeightSelector {
    pub fn new() -> Self {
        WeightSelector(AgeWeightSelector::new(0.0))
    }
}

impl Default for WeightSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl ClauseSelector for WeightSelector {
    fn select(&mut self, unprocessed: &mut IndexSet<usize>, clauses: &[Arc<Clause>]) -> Option<usize> {
        self.0.select(unprocessed, clauses)
    }

    fn name(&self) -> &str {
        "Weight"
    }

    fn reset(&mut self) {
        self.0.reset();
    }
}

// =============================================================================
// AgeWeightSink — ProverSink implementation
// =============================================================================

use indexmap::IndexMap;
use super::clause::ProverSink;

/// Age-weight heuristic as a `ProverSink`.
///
/// Tracks its own unprocessed set from prover signals. Caches clause weights
/// at transfer time so `select()` requires no external state.
pub struct AgeWeightSink {
    age_probability: f64,
    rng_state: u64,
    /// Own view of U: insertion-ordered, stores (clause_idx → weight)
    unprocessed: IndexMap<usize, usize>,
}

impl AgeWeightSink {
    pub fn new(age_probability: f64) -> Self {
        Self {
            age_probability: age_probability.clamp(0.0, 1.0),
            rng_state: 12345,
            unprocessed: IndexMap::new(),
        }
    }

    fn next_random(&mut self) -> f64 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        (self.rng_state >> 33) as f64 / (1u64 << 31) as f64
    }
}

impl ProverSink for AgeWeightSink {
    fn on_transfer(&mut self, clause_idx: usize, clause: &Arc<Clause>) {
        let weight = AgeWeightSelector::clause_weight(clause);
        self.unprocessed.insert(clause_idx, weight);
    }

    fn on_activate(&mut self, _clause_idx: usize) {
        // Already removed from unprocessed during select()
    }

    fn on_simplify(&mut self, clause_idx: usize) {
        self.unprocessed.shift_remove(&clause_idx);
    }

    fn select(&mut self) -> Option<usize> {
        if self.unprocessed.is_empty() {
            return None;
        }

        let r = self.next_random();
        let pos = if r < self.age_probability {
            // Oldest = first in insertion order
            0
        } else {
            // Lightest
            self.unprocessed
                .iter()
                .enumerate()
                .min_by_key(|(_, (_, &w))| w)
                .map(|(pos, _)| pos)
                .unwrap_or(0)
        };

        self.unprocessed
            .shift_remove_index(pos)
            .map(|(idx, _)| idx)
    }

    fn name(&self) -> &str {
        "AgeWeight"
    }

    fn reset(&mut self) {
        self.unprocessed.clear();
        self.rng_state = 12345;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::{FunctionSymbol, Interner, Literal, PredicateSymbol, Term, Variable};
    use std::sync::Arc;

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

        fn func(&mut self, name: &str, args: Vec<Term>) -> Term {
            let id = self.interner.intern_function(name);
            Term::Function(FunctionSymbol::new(id, args.len() as u8), args)
        }

        fn pred(&mut self, name: &str, arity: u8) -> PredicateSymbol {
            let id = self.interner.intern_predicate(name);
            PredicateSymbol::new(id, arity)
        }
    }

    /// Create a clause with a predicate of given arity (number of variable arguments).
    fn make_clause(ctx: &mut TestContext, predicate_name: &str, num_args: usize) -> Clause {
        // Build args first to avoid nested mutable borrows
        let args: Vec<Term> = (0..num_args)
            .map(|j| {
                let id = ctx.interner.intern_variable(&format!("X{}", j));
                Term::Variable(Variable::new(id))
            })
            .collect();

        let pred = ctx.pred(predicate_name, num_args as u8);

        Clause::new(vec![Literal::positive(pred, args)])
    }

    /// Create a clause with nested functions to increase weight.
    fn make_heavy_clause(ctx: &mut TestContext, depth: usize) -> Clause {
        // Build nested function: f(f(f(...f(X)...)))
        let mut term = ctx.var("X");
        for _ in 0..depth {
            term = ctx.func("f", vec![term]);
        }

        let pred = ctx.pred("P", 1);

        Clause::new(vec![Literal::positive(pred, vec![term])])
    }

    #[test]
    fn test_age_selection() {
        let mut ctx = TestContext::new();
        let mut selector = AgeWeightSelector::new(1.0); // Always select by age
        let clauses: Vec<Arc<Clause>> = vec![
            Arc::new(make_clause(&mut ctx, "P", 3)),
            Arc::new(make_clause(&mut ctx, "Q", 1)),
            Arc::new(make_clause(&mut ctx, "R", 2)),
        ];
        let mut unprocessed: IndexSet<usize> = (0..3).collect();

        // Should always select front (oldest)
        assert_eq!(selector.select(&mut unprocessed, &clauses), Some(0));
        assert_eq!(selector.select(&mut unprocessed, &clauses), Some(1));
        assert_eq!(selector.select(&mut unprocessed, &clauses), Some(2));
    }

    #[test]
    fn test_weight_selection() {
        let mut ctx = TestContext::new();
        let mut selector = AgeWeightSelector::new(0.0); // Always select by weight
        let clauses: Vec<Arc<Clause>> = vec![
            Arc::new(make_heavy_clause(&mut ctx, 5)), // Heavy: P(f(f(f(f(f(X)))))) = 1 + 6 = 7 symbols
            Arc::new(make_clause(&mut ctx, "Q", 0)),  // Light: Q() = 1 symbol
            Arc::new(make_heavy_clause(&mut ctx, 2)), // Medium: P(f(f(X))) = 1 + 3 = 4 symbols
        ];
        let mut unprocessed: IndexSet<usize> = (0..3).collect();

        // Should select lightest first (index 1, the nullary Q)
        assert_eq!(selector.select(&mut unprocessed, &clauses), Some(1));
    }

    #[test]
    fn test_empty_unprocessed() {
        let mut selector = AgeWeightSelector::default();
        let clauses: Vec<Arc<Clause>> = vec![];
        let mut unprocessed: IndexSet<usize> = IndexSet::new();

        assert_eq!(selector.select(&mut unprocessed, &clauses), None);
    }

    #[test]
    fn test_clause_weight() {
        let mut ctx = TestContext::new();

        // P(X, Y) = predicate(1) + var(1) + var(1) = 3
        let c1 = make_clause(&mut ctx, "P", 2);
        assert_eq!(AgeWeightSelector::clause_weight(&c1), 3);

        // P(f(X)) = predicate(1) + function(1) + var(1) = 3
        let c2 = make_heavy_clause(&mut ctx, 1);
        assert_eq!(AgeWeightSelector::clause_weight(&c2), 3);

        // P(f(f(X))) = predicate(1) + function(1) + function(1) + var(1) = 4
        let c3 = make_heavy_clause(&mut ctx, 2);
        assert_eq!(AgeWeightSelector::clause_weight(&c3), 4);
    }
}
