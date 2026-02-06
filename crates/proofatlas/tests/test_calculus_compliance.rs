//! Test compliance with calculus_quick_reference.md

use proofatlas::{
    factoring, resolution, Clause, Constant, Interner, Literal, LiteralSelector,
    PredicateSymbol, SelectAll, Term, Variable,
};

/// Test context that holds the interner and provides helper methods
struct TestCtx {
    interner: Interner,
}

impl TestCtx {
    fn new() -> Self {
        Self {
            interner: Interner::new(),
        }
    }

    fn var(&mut self, name: &str) -> Term {
        Term::Variable(Variable {
            id: self.interner.intern_variable(name),
        })
    }

    fn const_(&mut self, name: &str) -> Term {
        Term::Constant(Constant {
            id: self.interner.intern_constant(name),
        })
    }

    fn pred(&mut self, name: &str, arity: u8) -> PredicateSymbol {
        PredicateSymbol {
            id: self.interner.intern_predicate(name),
            arity,
        }
    }
}

#[test]
fn test_resolution_should_use_selection() {
    let mut ctx = TestCtx::new();

    let p = ctx.pred("P", 1);
    let q = ctx.pred("Q", 1);

    let x = ctx.var("X");
    let a = ctx.const_("a");

    // Clause 1: P(X) ∨ Q(X)
    let clause1 = Clause::new(vec![
        Literal::positive(p, vec![x.clone()]),
        Literal::positive(q, vec![x.clone()]),
    ]);

    // Clause 2: ~P(a) ∨ Q(a)
    let clause2 = Clause::new(vec![
        Literal::negative(p, vec![a.clone()]),
        Literal::positive(q, vec![a.clone()]),
    ]);

    // With SelectAll, all literals are eligible for resolution
    let selector = SelectAll;
    let selected1 = selector.select(&clause1);
    let selected2 = selector.select(&clause2);

    println!("Clause 1 selected literals: {:?}", selected1);
    println!("Clause 2 selected literals: {:?}", selected2);

    assert_eq!(selected1.len(), 2);
    assert_eq!(selected2.len(), 2);

    // Apply resolution with SelectAll
    let selector = SelectAll;
    let results = resolution(&clause1, &clause2, 0, 1, &selector, &mut ctx.interner);

    println!("Number of resolvents: {}", results.len());

    // Since only P(X) and ~P(a) form a complementary pair, we get 1 resolvent
    assert_eq!(results.len(), 1);
}

#[test]
fn test_factoring_should_use_selection() {
    let mut ctx = TestCtx::new();

    let p = ctx.pred("P", 1);
    let q = ctx.pred("Q", 1);

    let x = ctx.var("X");
    let y = ctx.var("Y");
    let z = ctx.var("Z");

    // Mixed polarity clause: ~P(X) ∨ P(Y) ∨ P(Z) ∨ Q(X)
    let clause = Clause::new(vec![
        Literal::negative(p, vec![x.clone()]),
        Literal::positive(p, vec![y.clone()]),
        Literal::positive(p, vec![z.clone()]),
        Literal::positive(q, vec![x.clone()]),
    ]);

    // With SelectAll, all literals are selected
    let selector = SelectAll;
    let selected = selector.select(&clause);
    println!("Selected literals for factoring: {:?}", selected);
    assert_eq!(selected.len(), 4); // All 4 literals are selected

    // Current factoring implementation:
    let selector = SelectAll;
    let results = factoring(&clause, 0, &selector);

    // With SelectAll, all literals can participate in factoring
    println!("Number of factors: {}", results.len());

    // With SelectAll: all literals are selected
    // P(Y) can factor with P(Z), and P(Z) can factor with P(Y)
    // This produces 2 factors (both equivalent)
    assert_eq!(results.len(), 2);
}
