//! Integration tests demonstrating all rules working together

use proofatlas_rust::core::logic::{Term, Predicate, Literal, Clause};
use proofatlas_rust::proofs::ProofState;
use proofatlas_rust::rules::{
    Rule, ResolutionRule, FactoringRule, SuperpositionRule, 
    SelectNegative, LiteralSelector,
    forward_subsumption_filter, DiscriminationIndex
};

#[test]
fn test_all_rules_integration() {
    // Create a simple problem with multiple rule opportunities
    let term_a = Term::Constant("a".to_string());
    let term_b = Term::Constant("b".to_string());
    let term_x = Term::Variable("X".to_string());
    let term_fx = Term::Function {
        name: "f".to_string(),
        args: vec![term_x.clone()]
    };
    let term_fa = Term::Function {
        name: "f".to_string(),
        args: vec![term_a.clone()]
    };
    
    // Clauses:
    // 1. P(X) ∨ Q(X)
    // 2. ~P(a) ∨ R(a)
    // 3. f(X) = a ∨ f(b) = b
    // 4. S(f(a))
    let clause1 = Clause::new(vec![
        Literal::new(true, Predicate::new("P".to_string(), vec![term_x.clone()])),
        Literal::new(true, Predicate::new("Q".to_string(), vec![term_x.clone()])),
    ]);
    
    let clause2 = Clause::new(vec![
        Literal::new(false, Predicate::new("P".to_string(), vec![term_a.clone()])),
        Literal::new(true, Predicate::new("R".to_string(), vec![term_a.clone()])),
    ]);
    
    let clause3 = Clause::new(vec![
        Literal::new(true, Predicate::new("=".to_string(), vec![term_fx, term_a.clone()])),
        Literal::new(true, Predicate::new("=".to_string(), vec![
            Term::Function { name: "f".to_string(), args: vec![term_b.clone()] },
            term_b.clone()
        ])),
    ]);
    
    let clause4 = Clause::new(vec![
        Literal::new(true, Predicate::new("S".to_string(), vec![term_fa.clone()])),
    ]);
    
    let state = ProofState::new(vec![clause1, clause2, clause3, clause4], vec![]);
    
    // Test Resolution
    let resolution = ResolutionRule::new();
    let res_result = resolution.apply(&state, &[0, 1], &[vec![], vec![]]);
    assert!(res_result.is_some());
    let res_app = res_result.unwrap();
    assert_eq!(res_app.rule_name, "resolution");
    // Should produce Q(a) ∨ R(a)
    
    // Test Factoring
    let factoring = FactoringRule::new();
    let fact_result = factoring.apply(&state, &[2], &[vec![]]);
    // Clause 3 has no unifiable literals with same polarity, so no factoring
    assert!(fact_result.is_none());
    
    // Test Superposition
    let superposition = SuperpositionRule::new();
    let sup_result = superposition.apply(&state, &[2, 3], &[vec![], vec![]]);
    assert!(sup_result.is_some());
    let sup_app = sup_result.unwrap();
    assert_eq!(sup_app.rule_name, "superposition");
    // Should produce S(a) from f(a) = a and S(f(a))
    
    // Test Forward Subsumption
    // forward_subsumption_filter removes new clauses that are subsumed by existing clauses
    let existing_clauses = vec![
        // Q(X) - already in the database
        Clause::new(vec![
            Literal::new(true, Predicate::new("Q".to_string(), vec![term_x.clone()])),
        ]),
    ];
    
    let new_clauses = vec![
        // Q(a) ∨ R(a) - will be subsumed by Q(X)
        Clause::new(vec![
            Literal::new(true, Predicate::new("Q".to_string(), vec![term_a.clone()])),
            Literal::new(true, Predicate::new("R".to_string(), vec![term_a.clone()])),
        ]),
        // T(b) - not subsumed by anything
        Clause::new(vec![
            Literal::new(true, Predicate::new("T".to_string(), vec![term_b.clone()])),
        ]),
    ];
    
    let filtered = forward_subsumption_filter(new_clauses, &existing_clauses);
    // Q(a) ∨ R(a) is subsumed by Q(X), so only T(b) remains
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].literals[0].predicate.name, "T")
}

#[test]
fn test_literal_selection_with_rules() {
    let term_x = Term::Variable("X".to_string());
    let term_a = Term::Constant("a".to_string());
    
    // Create clauses with mixed polarities
    let clause1 = Clause::new(vec![
        Literal::new(true, Predicate::new("P".to_string(), vec![term_x.clone()])),
        Literal::new(false, Predicate::new("Q".to_string(), vec![term_x.clone()])),
        Literal::new(true, Predicate::new("R".to_string(), vec![term_x.clone()])),
    ]);
    
    let clause2 = Clause::new(vec![
        Literal::new(true, Predicate::new("Q".to_string(), vec![term_a.clone()])),
        Literal::new(false, Predicate::new("P".to_string(), vec![term_a.clone()])),
    ]);
    
    let state = ProofState::new(vec![clause1, clause2], vec![]);
    let selector = SelectNegative;
    
    // Select negative literals from each clause
    let selected1 = selector.select(&state.processed[0]);
    let selected2 = selector.select(&state.processed[1]);
    
    assert_eq!(selected1, vec![1]); // ~Q(X)
    assert_eq!(selected2, vec![1]); // ~P(a)
    
    // Try resolution with selection
    let resolution = ResolutionRule::new();
    let result = resolution.apply(&state, &[0, 1], &[selected1, selected2]);
    
    // Resolution should fail: both selected literals are negative
    // ~Q(X) from clause1 and ~P(a) from clause2 cannot resolve
    assert!(result.is_none());
    
    // Now try without selection to show it would work
    let result_no_selection = resolution.apply(&state, &[0, 1], &[vec![], vec![]]);
    assert!(result_no_selection.is_some()); // Can resolve P(X) with ~P(a) or ~Q(X) with Q(a)
}

#[test]
fn test_discrimination_tree_vs_linear_search() {
    // Create many clauses to show the benefit of indexing
    let mut clauses = vec![];
    let mut index = DiscriminationIndex::new();
    
    // Create 100 different predicates with various terms
    for i in 0..100 {
        let term = if i % 3 == 0 {
            Term::Variable(format!("X{}", i))
        } else if i % 3 == 1 {
            Term::Constant(format!("c{}", i))
        } else {
            Term::Function {
                name: "f".to_string(),
                args: vec![Term::Constant(format!("a{}", i))],
            }
        };
        
        let lit = Literal::new(i % 2 == 0, Predicate::new("P".to_string(), vec![term]));
        let clause = Clause::new(vec![lit.clone()]);
        
        // Index the literal
        index.index_literal(&lit, i, 0);
        
        clauses.push(clause);
    }
    
    // Query literal: ~P(f(a50))
    let query_term = Term::Function {
        name: "f".to_string(),
        args: vec![Term::Constant("a50".to_string())],
    };
    let query_lit = Literal::new(false, Predicate::new("P".to_string(), vec![query_term]));
    
    // Find resolution partners using index
    let partners = index.find_resolution_partners(&query_lit);
    
    // Should find P(f(a50)) and any P(X_i) where i is even
    assert!(!partners.is_empty());
    
    // Verify at least one exact match
    let exact_match = partners.iter().any(|p| p.clause_id == 50);
    assert!(exact_match);
}