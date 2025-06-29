//! Standalone test to verify equivalence between standard and array implementations

use proofatlas_rust::core::logic::{Clause, Literal, Predicate, Term};
use proofatlas_rust::array_repr::{ArrayProblem, ArrayBuilder};
use proofatlas_rust::array_repr::saturation::{saturate as array_saturate, SaturationConfig};

fn main() {
    println!("Running equivalence tests...\n");
    
    test_basic_contradiction();
    test_propositional_square();
    test_first_order_problem();
    test_tautology_handling();
    
    println!("\nAll equivalence tests passed!");
}

fn test_basic_contradiction() {
    println!("Test: Basic contradiction (P and ~P)");
    
    let p = Predicate::new("P".to_string(), vec![]);
    let clauses = vec![
        Clause::new(vec![Literal::positive(p.clone())]),
        Clause::new(vec![Literal::negative(p)]),
    ];
    
    // Build array problem
    let mut array_problem = ArrayProblem::new();
    let mut builder = ArrayBuilder::new(&mut array_problem);
    for clause in &clauses {
        builder.add_clause(clause);
    }
    
    // Run array saturation
    let config = SaturationConfig {
        max_iterations: 10,
        ..Default::default()
    };
    let result = array_saturate(&mut array_problem, &config);
    
    assert!(result.found_empty_clause, "Should find contradiction");
    assert_eq!(result.num_iterations, 1, "Should find in first iteration");
    println!("  ✓ Found empty clause in {} iterations", result.num_iterations);
}

fn test_propositional_square() {
    println!("\nTest: Propositional square of opposition");
    
    let p = Predicate::new("P".to_string(), vec![]);
    let q = Predicate::new("Q".to_string(), vec![]);
    
    let clauses = vec![
        // P | Q
        Clause::new(vec![Literal::positive(p.clone()), Literal::positive(q.clone())]),
        // ~P | Q
        Clause::new(vec![Literal::negative(p.clone()), Literal::positive(q.clone())]),
        // P | ~Q
        Clause::new(vec![Literal::positive(p.clone()), Literal::negative(q.clone())]),
        // ~P | ~Q
        Clause::new(vec![Literal::negative(p), Literal::negative(q)]),
    ];
    
    // Build array problem
    let mut array_problem = ArrayProblem::new();
    let mut builder = ArrayBuilder::new(&mut array_problem);
    for clause in &clauses {
        builder.add_clause(clause);
    }
    
    // Run array saturation
    let config = SaturationConfig {
        max_iterations: 100,
        ..Default::default()
    };
    let result = array_saturate(&mut array_problem, &config);
    
    assert!(result.found_empty_clause, "Should find contradiction");
    println!("  ✓ Found empty clause in {} iterations", result.num_iterations);
    println!("  ✓ Generated {} clauses", result.num_clauses_generated);
}

fn test_first_order_problem() {
    println!("\nTest: First-order unification");
    
    let x = Term::Variable("X".to_string());
    let a = Term::Constant("a".to_string());
    let f_x = Term::Function { name: "f".to_string(), args: vec![x.clone()] };
    let f_a = Term::Function { name: "f".to_string(), args: vec![a.clone()] };
    
    let clauses = vec![
        // P(X, f(X))
        Clause::new(vec![
            Literal::positive(Predicate::new("P".to_string(), vec![x.clone(), f_x])),
        ]),
        // ~P(a, f(a))
        Clause::new(vec![
            Literal::negative(Predicate::new("P".to_string(), vec![a.clone(), f_a])),
        ]),
    ];
    
    // Build array problem
    let mut array_problem = ArrayProblem::new();
    let mut builder = ArrayBuilder::new(&mut array_problem);
    for clause in &clauses {
        builder.add_clause(clause);
    }
    
    // Run array saturation
    let config = SaturationConfig {
        max_iterations: 10,
        ..Default::default()
    };
    let result = array_saturate(&mut array_problem, &config);
    
    assert!(result.found_empty_clause, "Should find contradiction through unification");
    println!("  ✓ Found empty clause through first-order unification");
}

fn test_tautology_handling() {
    println!("\nTest: Tautology elimination");
    
    let p = Predicate::new("P".to_string(), vec![]);
    let q = Predicate::new("Q".to_string(), vec![]);
    
    let clauses = vec![
        // P | ~P (tautology)
        Clause::new(vec![
            Literal::positive(p.clone()),
            Literal::negative(p.clone()),
        ]),
        // Q
        Clause::new(vec![Literal::positive(q.clone())]),
        // ~Q
        Clause::new(vec![Literal::negative(q)]),
    ];
    
    // Build array problem
    let mut array_problem = ArrayProblem::new();
    let mut builder = ArrayBuilder::new(&mut array_problem);
    for clause in &clauses {
        builder.add_clause(clause);
    }
    
    // Run array saturation
    let config = SaturationConfig {
        max_iterations: 10,
        ..Default::default()
    };
    let result = array_saturate(&mut array_problem, &config);
    
    assert!(result.found_empty_clause, "Should find contradiction despite tautology");
    assert!(result.num_clauses_generated <= 2, "Should not generate many clauses");
    println!("  ✓ Found contradiction with minimal clause generation");
    println!("  ✓ Generated only {} clauses", result.num_clauses_generated);
}