//! Test to verify how literal selection affects inference rules

use proofatlas::{
    Clause, Literal, Atom, PredicateSymbol, Term, Variable, factoring, SelectAll
};

#[test]
fn test_factoring_with_select_all_detailed() {
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

