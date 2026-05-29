//! Test to verify how literal selection affects inference rules

use proofatlas::{factoring, Clause, Interner, Literal, PredicateSymbol, SelectAll, Term, Variable};

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
        Literal::positive(p, vec![x.clone()]),
        Literal::positive(p, vec![y.clone()]),
        Literal::positive(p, vec![z.clone()]),
    ]);

    let selector = SelectAll;
    let results = factoring(&clause, 0, &selector);

    // With SelectAll every literal is selected, and `factoring` tries each
    // ordered pair (i, j) with i != j of same-predicate, same-polarity
    // literals. For P(X) ∨ P(Y) ∨ P(Z) that is the 6 ordered pairs, each of
    // which unifies, so the rule emits exactly 6 factors. (They are all
    // isomorphic modulo variable renaming — "two distinct singleton P
    // literals" — and the redundant copies are removed later by forward
    // simplification, not by `factoring` itself.)
    assert_eq!(
        results.len(),
        6,
        "SelectAll factoring of three P-literals should emit all 6 ordered pairs"
    );

    // Every factor must be a 2-literal clause whose literals are both the
    // positive predicate P (dropping exactly one literal after unification).
    for result in &results {
        let proofatlas::StateChange::Add(factor, rule, _) = result else {
            panic!("factoring should emit Add changes, got {result:?}");
        };
        assert_eq!(rule, "Factoring");
        assert_eq!(factor.literals.len(), 2, "each factor drops exactly one literal");
        for lit in &factor.literals {
            assert!(lit.polarity, "factor literals stay positive");
            assert_eq!(lit.predicate, p, "factor literals are the P predicate");
        }
    }
}
