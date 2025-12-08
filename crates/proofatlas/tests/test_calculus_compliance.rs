//! Test compliance with calculus_quick_reference.md

use proofatlas::{
    factoring, resolution, Atom, Clause, Constant, Literal, LiteralSelector, PredicateSymbol,
    SelectAll, Term, Variable,
};

#[test]
fn test_resolution_should_use_selection() {
    // According to calculus_quick_reference.md:
    // P(x) ∨ C₁    ¬P(t) ∨ C₂
    // ------------------------  where σ = mgu(x, t), P(x) and ¬P(t) selected.
    //       (C₁ ∨ C₂)σ

    let p = PredicateSymbol {
        name: "P".to_string(),
        arity: 1,
    };
    let q = PredicateSymbol {
        name: "Q".to_string(),
        arity: 1,
    };

    let x = Term::Variable(Variable {
        name: "X".to_string(),
    });
    let a = Term::Constant(Constant {
        name: "a".to_string(),
    });

    // Clause 1: P(X) ∨ Q(X)
    let clause1 = Clause::new(vec![
        Literal::positive(Atom {
            predicate: p.clone(),
            args: vec![x.clone()],
        }),
        Literal::positive(Atom {
            predicate: q.clone(),
            args: vec![x.clone()],
        }),
    ]);

    // Clause 2: ~P(a) ∨ Q(a)
    let clause2 = Clause::new(vec![
        Literal::negative(Atom {
            predicate: p.clone(),
            args: vec![a.clone()],
        }),
        Literal::positive(Atom {
            predicate: q.clone(),
            args: vec![a.clone()],
        }),
    ]);

    // With SelectAll, all literals are eligible for resolution
    let selector = SelectAll;
    let selected1 = selector.select(&clause1);
    let selected2 = selector.select(&clause2);

    println!("Clause 1 selected literals: {:?}", selected1);
    println!("Clause 2 selected literals: {:?}", selected2);

    // With SelectAll:
    // - Clause 1: all literals are selected: {0, 1}
    // - Clause 2: all literals are selected: {0, 1}
    assert_eq!(selected1.len(), 2);
    assert_eq!(selected2.len(), 2);

    // Apply resolution with SelectAll
    let selector = SelectAll;
    let results = resolution(&clause1, &clause2, 0, 1, &selector);

    // With SelectAll, resolution can happen on any complementary pair
    println!("Number of resolvents: {}", results.len());

    // Since only P(X) and ~P(a) form a complementary pair, we get 1 resolvent
    assert_eq!(results.len(), 1);
}

#[test]
fn test_factoring_should_use_selection() {
    // According to calculus_quick_reference.md:
    // P(s) ∨ P(t) ∨ C
    // ----------------  where σ = mgu(s, t), P(s) selected.
    //    (P(s) ∨ C)σ

    let p = PredicateSymbol {
        name: "P".to_string(),
        arity: 1,
    };
    let q = PredicateSymbol {
        name: "Q".to_string(),
        arity: 1,
    };

    let x = Term::Variable(Variable {
        name: "X".to_string(),
    });
    let y = Term::Variable(Variable {
        name: "Y".to_string(),
    });
    let z = Term::Variable(Variable {
        name: "Z".to_string(),
    });

    // Mixed polarity clause: ~P(X) ∨ P(Y) ∨ P(Z) ∨ Q(X)
    let clause = Clause::new(vec![
        Literal::negative(Atom {
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
        Literal::positive(Atom {
            predicate: q.clone(),
            args: vec![x.clone()],
        }),
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
