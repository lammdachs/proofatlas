//! Test to verify how literal selection affects inference rules

use proofatlas::{
    Clause, Literal, Atom, PredicateSymbol, Term, Variable,
    resolution, factoring, NoSelection
};

#[test]
fn test_factoring_with_no_selection_detailed() {
    // P(X) ∨ P(Y) ∨ P(Z)
    let p = PredicateSymbol { name: "P".to_string(), arity: 1 };
    
    let x = Term::Variable(Variable { name: "X".to_string() });
    let y = Term::Variable(Variable { name: "Y".to_string() });
    let z = Term::Variable(Variable { name: "Z".to_string() });
    
    let clause = Clause::new(vec![
        Literal::positive(Atom { predicate: p.clone(), args: vec![x.clone()] }),
        Literal::positive(Atom { predicate: p.clone(), args: vec![y.clone()] }),
        Literal::positive(Atom { predicate: p.clone(), args: vec![z.clone()] }),
    ]);
    
    let selector = NoSelection;
    let results = factoring(&clause, 0, &selector);
    
    println!("Number of factors: {}", results.len());
    for (i, result) in results.iter().enumerate() {
        println!("Factor {}: {}", i + 1, result.conclusion);
    }
    
    // With NoSelection, all literals are selected
    // We can factor:
    // - P(X) with P(Y): produces P(X) ∨ P(Z)
    // - P(X) with P(Z): produces P(X) ∨ P(Y)
    // - P(Y) with P(X): produces P(Y) ∨ P(Z)
    // - P(Y) with P(Z): produces P(Y) ∨ P(X)
    // - P(Z) with P(X): produces P(Z) ∨ P(Y)
    // - P(Z) with P(Y): produces P(Z) ∨ P(X)
    // But duplicates should be removed, so we expect 3 unique factors
}

#[test]
fn test_resolution_with_select_negative() {
    // ~P(X) ∨ Q(Y)
    // P(a) ∨ ~R(b)
    let p = PredicateSymbol { name: "P".to_string(), arity: 1 };
    let q = PredicateSymbol { name: "Q".to_string(), arity: 1 };
    let r = PredicateSymbol { name: "R".to_string(), arity: 1 };
    
    let x = Term::Variable(Variable { name: "X".to_string() });
    let y = Term::Variable(Variable { name: "Y".to_string() });
    let a = Term::Variable(Variable { name: "a".to_string() });
    let b = Term::Variable(Variable { name: "b".to_string() });
    
    let clause1 = Clause::new(vec![
        Literal::negative(Atom { predicate: p.clone(), args: vec![x.clone()] }),
        Literal::positive(Atom { predicate: q.clone(), args: vec![y.clone()] }),
    ]);
    
    let clause2 = Clause::new(vec![
        Literal::positive(Atom { predicate: p.clone(), args: vec![a.clone()] }),
        Literal::negative(Atom { predicate: r.clone(), args: vec![b.clone()] }),
    ]);
    
    let selector = SelectNegative;
    let results = resolution(&clause1, &clause2, 0, 1, &selector);
    
    println!("\nResolution with SelectNegative:");
    println!("Clause 1: {}", clause1);
    println!("Clause 2: {}", clause2);
    println!("Number of resolvents: {}", results.len());
    for (i, result) in results.iter().enumerate() {
        println!("Resolvent {}: {}", i + 1, result.conclusion);
    }
    
    // With SelectNegative:
    // - Clause 1: only ~P(X) is selected
    // - Clause 2: only ~R(b) is selected
    // Since they have different predicates, no resolution is possible
    assert_eq!(results.len(), 0);
}