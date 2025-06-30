//! Integration tests demonstrating array-based theorem prover

use proofatlas_rust::parsing::parse_types::{ParseTerm, ParsePredicate, ParseLiteral, ParseClause, ParseProblem};
use proofatlas_rust::core::parser_convert::parse_problem_to_array;
use proofatlas_rust::saturation::{saturate, SaturationConfig};
use proofatlas_rust::core::SaturationResult;

#[test]
fn test_simple_resolution_proof() {
    // Create a simple problem: P(a), ~P(a) ⊢ ⊥
    let mut problem = ParseProblem::new();
    
    // Clause 1: P(a)
    let term_a = ParseTerm::Constant("a".to_string());
    let p_a = ParsePredicate { name: "P".to_string(), args: vec![term_a.clone()] };
    let lit1 = ParseLiteral { predicate: p_a.clone(), polarity: true };
    let clause1 = ParseClause { literals: vec![lit1] };
    problem.add_clause(clause1, false);
    
    // Clause 2: ~P(a)
    let lit2 = ParseLiteral { predicate: p_a, polarity: false };
    let clause2 = ParseClause { literals: vec![lit2] };
    problem.add_clause(clause2, false);
    
    // Convert to array representation
    let mut array_problem = parse_problem_to_array(problem);
    
    // Run saturation
    let config = SaturationConfig::default();
    let result = saturate(&mut array_problem, &config);
    
    match result {
        SaturationResult::Proof(proof) => {
            // Should find empty clause
            assert!(proof.problem.has_empty_clause());
        }
        _ => panic!("Expected to find a proof"),
    }
}

#[test]
fn test_factoring() {
    // Create a problem that requires factoring: P(X) ∨ P(a) ⊢ P(a)
    let mut problem = ParseProblem::new();
    
    // Clause: P(X) ∨ P(a)
    let term_x = ParseTerm::Variable("X".to_string());
    let term_a = ParseTerm::Constant("a".to_string());
    let p_x = ParsePredicate { name: "P".to_string(), args: vec![term_x] };
    let p_a = ParsePredicate { name: "P".to_string(), args: vec![term_a] };
    let lit1 = ParseLiteral { predicate: p_x, polarity: true };
    let lit2 = ParseLiteral { predicate: p_a, polarity: true };
    let clause = ParseClause { literals: vec![lit1, lit2] };
    problem.add_clause(clause, false);
    
    // Convert to array representation
    let mut array_problem = parse_problem_to_array(problem);
    
    // Check initial state
    println!("Initial clauses: {}", array_problem.num_clauses);
    for i in 0..array_problem.num_clauses {
        let lits = array_problem.clause_literals(i);
        println!("Clause {}: {} literals", i, lits.len());
    }
    
    // Run saturation
    let config = SaturationConfig {
        max_iterations: 10,
        ..Default::default()
    };
    let result = saturate(&mut array_problem, &config);
    
    match result {
        SaturationResult::Saturated => {
            // The saturation should complete successfully
            // P(X) subsumes P(a), so only the original clause remains
            assert_eq!(array_problem.num_clauses, 1);
        }
        SaturationResult::Proof(proof) => {
            println!("Final clauses: {}", proof.problem.num_clauses);
            for i in 0..proof.problem.num_clauses {
                let lits = proof.problem.clause_literals(i);
                println!("Clause {}: {} literals", i, lits.len());
            }
            panic!("Found proof when expecting saturation: {} clauses", proof.problem.num_clauses);
        }
        SaturationResult::ResourceLimit => {
            panic!("Hit resource limit");
        }
    }
}

#[test]
fn test_subsumption() {
    // Test forward subsumption: P(X) subsumes P(a) ∨ Q(b)
    let mut problem = ParseProblem::new();
    
    // Clause 1: P(X)
    let term_x = ParseTerm::Variable("X".to_string());
    let p_x = ParsePredicate { name: "P".to_string(), args: vec![term_x] };
    let lit1 = ParseLiteral { predicate: p_x, polarity: true };
    let clause1 = ParseClause { literals: vec![lit1] };
    problem.add_clause(clause1, false);
    
    // Clause 2: P(a) ∨ Q(b)
    let term_a = ParseTerm::Constant("a".to_string());
    let term_b = ParseTerm::Constant("b".to_string());
    let p_a = ParsePredicate { name: "P".to_string(), args: vec![term_a] };
    let q_b = ParsePredicate { name: "Q".to_string(), args: vec![term_b] };
    let lit2 = ParseLiteral { predicate: p_a, polarity: true };
    let lit3 = ParseLiteral { predicate: q_b, polarity: true };
    let clause2 = ParseClause { literals: vec![lit2, lit3] };
    problem.add_clause(clause2, false);
    
    // Convert to array representation
    let mut array_problem = parse_problem_to_array(problem);
    
    // Run saturation with backward subsumption enabled
    let config = SaturationConfig {
        use_backward_subsumption: true,
        ..Default::default()
    };
    let result = saturate(&mut array_problem, &config);
    
    match result {
        SaturationResult::Saturated | SaturationResult::Proof(_) => {
            // The saturation should handle subsumption
            // With proper subsumption, redundant clauses should be removed
        }
        _ => panic!("Unexpected result"),
    }
}

#[test]
fn test_empty_clause_detection() {
    // Test immediate empty clause detection
    let mut problem = ParseProblem::new();
    
    // Add empty clause
    let empty = ParseClause { literals: vec![] };
    problem.add_clause(empty, false);
    
    // Convert to array representation
    let mut array_problem = parse_problem_to_array(problem);
    
    // Run saturation
    let config = SaturationConfig::default();
    let result = saturate(&mut array_problem, &config);
    
    match result {
        SaturationResult::Proof(proof) => {
            // Should immediately find the empty clause
            assert_eq!(proof.steps.len(), 0);
        }
        _ => panic!("Expected to find empty clause immediately"),
    }
}

#[test]
fn test_tautology_deletion() {
    // Test that tautologies are not kept: P(a) ∨ ~P(a)
    let mut problem = ParseProblem::new();
    
    // Tautological clause: P(a) ∨ ~P(a)
    let term_a = ParseTerm::Constant("a".to_string());
    let p_a = ParsePredicate { name: "P".to_string(), args: vec![term_a] };
    let lit1 = ParseLiteral { predicate: p_a.clone(), polarity: true };
    let lit2 = ParseLiteral { predicate: p_a, polarity: false };
    let tautology = ParseClause { literals: vec![lit1, lit2] };
    problem.add_clause(tautology, false);
    
    // Convert to array representation
    let mut array_problem = parse_problem_to_array(problem);
    
    // Run saturation
    let config = SaturationConfig {
        max_iterations: 5,
        ..Default::default()
    };
    let result = saturate(&mut array_problem, &config);
    
    match result {
        SaturationResult::Saturated => {
            // Should saturate quickly without generating many clauses
            // since tautologies should be deleted
            assert!(array_problem.num_clauses < 10);
        }
        SaturationResult::Proof(proof) => {
            panic!("Found proof when expecting saturation: {} clauses", proof.problem.num_clauses);
        }
        SaturationResult::ResourceLimit => {
            panic!("Hit resource limit");
        }
    }
}