//! Test to verify how literal selection affects inference rules

use proofatlas::{factoring, Atom, Clause, Interner, Literal, PredicateSymbol, SelectAll, Term, Variable};

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

    fn pred(&mut self, name: &str, arity: u8) -> PredicateSymbol {
        PredicateSymbol {
            id: self.interner.intern_predicate(name),
            arity,
        }
    }
}

#[test]
fn test_factoring_with_select_all_detailed() {
    let mut ctx = TestCtx::new();

    // P(X) ∨ P(Y) ∨ P(Z)
    let p = ctx.pred("P", 1);

    let x = ctx.var("X");
    let y = ctx.var("Y");
    let z = ctx.var("Z");

    let clause = Clause::new(vec![
        Literal::positive(Atom {
            predicate: p.clone(),
            args: vec![x.clone()],
        }),
        Literal::positive(Atom {
            predicate: p.clone(),
            args: vec![y.clone()],
        }),
        Literal::positive(Atom {
            predicate: p.clone(),
            args: vec![z.clone()],
        }),
    ]);

    let selector = SelectAll;
    let results = factoring(&clause, 0, &selector);

    println!("Number of factors: {}", results.len());
    for (i, result) in results.iter().enumerate() {
        println!("Factor {}: {}", i + 1, result.conclusion);
    }

    // With SelectAll, all literals are selected
    // We can factor:
    // - P(X) with P(Y): produces P(X) ∨ P(Z)
    // - P(X) with P(Z): produces P(X) ∨ P(Y)
    // - P(Y) with P(X): produces P(Y) ∨ P(Z)
    // - P(Y) with P(Z): produces P(Y) ∨ P(X)
    // - P(Z) with P(X): produces P(Z) ∨ P(Y)
    // - P(Z) with P(Y): produces P(Z) ∨ P(X)
    // But duplicates should be removed, so we expect 3 unique factors
}
