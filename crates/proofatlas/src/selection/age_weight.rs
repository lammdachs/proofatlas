//! Age-weight heuristic clause selection
//!
//! Implements the classic age-weight ratio heuristic used in theorem provers.
//! `AgeWeightSink` alternates between selecting the oldest clause (FIFO) and
//! the lightest clause (by symbol count) based on a configurable probability.

use crate::logic::Clause;

use indexmap::IndexMap;
use std::sync::Arc;
use super::clause::ProverSink;

/// Calculate the weight of a clause (symbol count).
fn clause_weight(clause: &Clause) -> usize {
    clause
        .literals
        .iter()
        .map(|lit| {
            // Count predicate symbol + weight of all argument terms
            1 + lit.args.iter().map(term_weight).sum::<usize>()
        })
        .sum()
}

/// Calculate the weight of a term recursively.
fn term_weight(term: &crate::logic::Term) -> usize {
    match term {
        crate::logic::Term::Variable(_) => 1,
        crate::logic::Term::Constant(_) => 1,
        crate::logic::Term::Function(_, args) => 1 + args.iter().map(term_weight).sum::<usize>(),
    }
}

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
        let weight = clause_weight(clause);
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
    fn test_clause_weight() {
        let mut ctx = TestContext::new();

        // P(X, Y) = predicate(1) + var(1) + var(1) = 3
        let c1 = make_clause(&mut ctx, "P", 2);
        assert_eq!(clause_weight(&c1), 3);

        // P(f(X)) = predicate(1) + function(1) + var(1) = 3
        let c2 = make_heavy_clause(&mut ctx, 1);
        assert_eq!(clause_weight(&c2), 3);

        // P(f(f(X))) = predicate(1) + function(1) + function(1) + var(1) = 4
        let c3 = make_heavy_clause(&mut ctx, 2);
        assert_eq!(clause_weight(&c3), 4);
    }
}
