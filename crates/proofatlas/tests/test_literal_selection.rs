//! Test to verify literal selection behavior

use proofatlas::{
    factoring, resolution, Clause, Constant, Interner, Literal, PredicateSymbol, SelectAll,
    Term, Variable,
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
fn test_resolution_tries_all_literals() {
    let mut ctx = TestCtx::new();

    // Create predicates
    let p = ctx.pred("P", 1);
    let q = ctx.pred("Q", 1);
    let r = ctx.pred("R", 1);

    // Create terms
    let x = ctx.var("X");
    let y = ctx.var("Y");
    let z = ctx.var("Z");
    let a = ctx.const_("a");
    let b = ctx.const_("b");
    let c = ctx.const_("c");

    // Create a clause with multiple literals: P(X) ∨ Q(Y) ∨ R(Z)
    let clause1 = Clause::new(vec![
        Literal::positive(p, vec![x.clone()]),
        Literal::positive(q, vec![y.clone()]),
        Literal::positive(r, vec![z.clone()]),
    ]);

    // Create another clause with negations of all three: ~P(a) ∨ ~Q(b) ∨ ~R(c)
    let clause2 = Clause::new(vec![
        Literal::negative(p, vec![a.clone()]),
        Literal::negative(q, vec![b.clone()]),
        Literal::negative(r, vec![c.clone()]),
    ]);

    // Apply resolution with SelectAll (all literals eligible)
    let selector = SelectAll;
    let results = resolution(&clause1, &clause2, 0, 1, &selector, &mut ctx.interner);

    // WITHOUT literal selection, we should get 3 different resolvents
    // (one for each pair of complementary literals)
    println!("Number of resolvents: {}", results.len());
    assert_eq!(
        results.len(),
        3,
        "Resolution should produce 3 resolvents without literal selection"
    );

    // Each resolvent should have 4 literals (2 from each parent minus the resolved pair)
    for result in &results {
        if let proofatlas::StateChange::Add(clause, _, _) = result {
            assert_eq!(clause.literals.len(), 4);
        } else {
            panic!("Expected StateChange::Add");
        }
    }
}

#[test]
fn test_factoring_tries_all_pairs() {
    let mut ctx = TestCtx::new();

    // Create predicate
    let p = ctx.pred("P", 1);

    // Create terms
    let x = ctx.var("X");
    let y = ctx.var("Y");
    let z = ctx.var("Z");

    // Create a clause with multiple identical predicates: P(X) ∨ P(Y) ∨ P(Z)
    let clause = Clause::new(vec![
        Literal::positive(p, vec![x.clone()]),
        Literal::positive(p, vec![y.clone()]),
        Literal::positive(p, vec![z.clone()]),
    ]);

    // Apply factoring with SelectAll (all literals eligible)
    let selector = SelectAll;
    let results = factoring(&clause, 0, &selector);

    // WITHOUT literal selection, we get factors for each pair
    // But since we select each literal and try to factor with others,
    // we get: P(X) factors with P(Y) and P(Z), P(Y) factors with P(X) and P(Z), etc.
    // This gives us 6 factors (but some may be duplicates modulo variable renaming)
    println!("Number of factors: {}", results.len());
    assert!(
        results.len() >= 3,
        "Factoring should produce at least 3 factors without literal selection"
    );

    // Each factor should have 2 literals (after factoring one pair)
    for result in &results {
        if let proofatlas::StateChange::Add(clause, _, _) = result {
            assert_eq!(clause.literals.len(), 2);
        } else {
            panic!("Expected StateChange::Add");
        }
    }
}
