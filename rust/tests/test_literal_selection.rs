//! Test to verify literal selection behavior

use proofatlas::{
    Clause, Literal, Atom, PredicateSymbol, Term, Variable, Constant,
    resolution, factoring, NoSelection
};

#[test]
fn test_resolution_tries_all_literals() {
    // Create a clause with multiple literals:
    // P(X) ∨ Q(Y) ∨ R(Z)
    let p = PredicateSymbol { name: "P".to_string(), arity: 1 };
    let q = PredicateSymbol { name: "Q".to_string(), arity: 1 };
    let r = PredicateSymbol { name: "R".to_string(), arity: 1 };
    
    let x = Term::Variable(Variable { name: "X".to_string() });
    let y = Term::Variable(Variable { name: "Y".to_string() });
    let z = Term::Variable(Variable { name: "Z".to_string() });
    
    let clause1 = Clause::new(vec![
        Literal::positive(Atom { predicate: p.clone(), args: vec![x.clone()] }),
        Literal::positive(Atom { predicate: q.clone(), args: vec![y.clone()] }),
        Literal::positive(Atom { predicate: r.clone(), args: vec![z.clone()] }),
    ]);
    
    // Create another clause with negations of all three:
    // ~P(a) ∨ ~Q(b) ∨ ~R(c)
    let a = Term::Constant(Constant { name: "a".to_string() });
    let b = Term::Constant(Constant { name: "b".to_string() });
    let c = Term::Constant(Constant { name: "c".to_string() });
    
    let clause2 = Clause::new(vec![
        Literal::negative(Atom { predicate: p.clone(), args: vec![a.clone()] }),
        Literal::negative(Atom { predicate: q.clone(), args: vec![b.clone()] }),
        Literal::negative(Atom { predicate: r.clone(), args: vec![c.clone()] }),
    ]);
    
    // Apply resolution with NoSelection (all literals eligible)
    let selector = NoSelection;
    let results = resolution(&clause1, &clause2, 0, 1, &selector);
    
    // WITHOUT literal selection, we should get 3 different resolvents
    // (one for each pair of complementary literals)
    println!("Number of resolvents: {}", results.len());
    assert_eq!(results.len(), 3, "Resolution should produce 3 resolvents without literal selection");
    
    // Each resolvent should have 4 literals (2 from each parent minus the resolved pair)
    for result in &results {
        assert_eq!(result.conclusion.literals.len(), 4);
    }
}

#[test]
fn test_factoring_tries_all_pairs() {
    // Create a clause with multiple identical predicates:
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
    
    // Apply factoring with NoSelection (all literals eligible)
    let selector = NoSelection;
    let results = factoring(&clause, 0, &selector);
    
    // WITHOUT literal selection, we get factors for each pair
    // But since we select each literal and try to factor with others,
    // we get: P(X) factors with P(Y) and P(Z), P(Y) factors with P(X) and P(Z), etc.
    // This gives us 6 factors (but some may be duplicates modulo variable renaming)
    println!("Number of factors: {}", results.len());
    assert!(results.len() >= 3, "Factoring should produce at least 3 factors without literal selection");
    
    // Each factor should have 2 literals (after factoring one pair)
    for result in &results {
        assert_eq!(result.conclusion.literals.len(), 2);
    }
}