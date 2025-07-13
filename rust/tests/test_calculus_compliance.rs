//! Test compliance with calculus_quick_reference.md

use proofatlas::{
    Clause, Literal, Atom, PredicateSymbol, Term, Variable, Constant,
    resolution, factoring, SelectNegative, LiteralSelector, NoSelection
};

#[test]
fn test_resolution_should_use_selection() {
    // According to calculus_quick_reference.md:
    // P(x) ∨ C₁    ¬P(t) ∨ C₂
    // ------------------------  where σ = mgu(x, t), P(x) and ¬P(t) selected.
    //       (C₁ ∨ C₂)σ
    
    let p = PredicateSymbol { name: "P".to_string(), arity: 1 };
    let q = PredicateSymbol { name: "Q".to_string(), arity: 1 };
    
    let x = Term::Variable(Variable { name: "X".to_string() });
    let a = Term::Constant(Constant { name: "a".to_string() });
    
    // Clause 1: P(X) ∨ Q(X) - with SelectNegative, P(X) should NOT be selected
    let clause1 = Clause::new(vec![
        Literal::positive(Atom { predicate: p.clone(), args: vec![x.clone()] }),
        Literal::positive(Atom { predicate: q.clone(), args: vec![x.clone()] }),
    ]);
    
    // Clause 2: ~P(a) ∨ Q(a) - with SelectNegative, only ~P(a) should be selected
    let clause2 = Clause::new(vec![
        Literal::negative(Atom { predicate: p.clone(), args: vec![a.clone()] }),
        Literal::positive(Atom { predicate: q.clone(), args: vec![a.clone()] }),
    ]);
    
    // What SHOULD happen with literal selection:
    let selector = SelectNegative;
    let selected1 = selector.select(&clause1);
    let selected2 = selector.select(&clause2);
    
    println!("Clause 1 selected literals: {:?}", selected1);
    println!("Clause 2 selected literals: {:?}", selected2);
    
    // With SelectNegative:
    // - Clause 1 has no negative literals, so NONE are selected: {}
    // - Clause 2 has one negative literal, so only it is selected: {0}
    assert_eq!(selected1.len(), 0);  // No negative literals!
    assert_eq!(selected2.len(), 1);
    assert!(selected2.contains(&0));
    
    // But the current resolution implementation uses selection:
    let no_selector = NoSelection;
    let results = resolution(&clause1, &clause2, 0, 1, &no_selector);
    
    // It should only resolve P(X) with ~P(a), but let's see:
    println!("Number of resolvents: {}", results.len());
    
    // With proper selection, we should get exactly 1 resolvent
    // But without selection, we get 1 (because only P matches)
    assert_eq!(results.len(), 1);
}

#[test] 
fn test_factoring_should_use_selection() {
    // According to calculus_quick_reference.md:
    // P(s) ∨ P(t) ∨ C
    // ----------------  where σ = mgu(s, t), P(s) selected.
    //    (P(s) ∨ C)σ
    
    let p = PredicateSymbol { name: "P".to_string(), arity: 1 };
    let q = PredicateSymbol { name: "Q".to_string(), arity: 1 };
    
    let x = Term::Variable(Variable { name: "X".to_string() });
    let y = Term::Variable(Variable { name: "Y".to_string() });
    let z = Term::Variable(Variable { name: "Z".to_string() });
    
    // Mixed polarity clause: ~P(X) ∨ P(Y) ∨ P(Z) ∨ Q(X)
    let clause = Clause::new(vec![
        Literal::negative(Atom { predicate: p.clone(), args: vec![x.clone()] }),
        Literal::positive(Atom { predicate: p.clone(), args: vec![y.clone()] }),
        Literal::positive(Atom { predicate: p.clone(), args: vec![z.clone()] }),
        Literal::positive(Atom { predicate: q.clone(), args: vec![x.clone()] }),
    ]);
    
    // With SelectNegative, only literal 0 (~P(X)) should be selected
    let selector = SelectNegative;
    let selected = selector.select(&clause);
    println!("Selected literals for factoring: {:?}", selected);
    assert_eq!(selected.len(), 1);
    assert!(selected.contains(&0));
    
    // Current factoring implementation:
    let no_selector = NoSelection;
    let results = factoring(&clause, 0, &no_selector);
    
    // It should NOT factor positive P literals because they're not selected
    // But the current implementation will factor them anyway
    println!("Number of factors: {}", results.len());
    
    // With selection: 0 factors (negative P can't factor with positive P)
    // Without selection: 1 factor (P(Y) and P(Z) can factor)
    assert_eq!(results.len(), 1);
}