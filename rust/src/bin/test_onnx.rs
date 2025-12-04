//! Test ONNX inference with context-aware clause scoring

use proofatlas::core::{Atom, Clause, ClauseRole, Constant, FunctionSymbol, Literal, PredicateSymbol, Term, Variable};
use proofatlas::ml::ClauseScorer;
use std::env;

fn main() {
    println!("Testing context-aware ONNX inference...\n");

    let mut scorer = ClauseScorer::new();

    // Get model path from args or use default
    let args: Vec<String> = env::args().collect();
    let model_path = if args.len() > 1 {
        args[1].clone()
    } else {
        "../.selectors/age_weight.onnx".to_string()
    };
    println!("Loading model from: {}", model_path);

    match scorer.load_model(&model_path) {
        Ok(()) => println!("Model loaded successfully!\n"),
        Err(e) => {
            println!("Failed to load model: {}", e);
            return;
        }
    }

    // Create test clauses with different ages and roles
    let clauses = vec![
        // 1. P(X) - a simple unary predicate (age=0, axiom)
        {
            let mut c = Clause::new(vec![Literal::positive(Atom {
                predicate: PredicateSymbol {
                    name: "P".to_string(),
                    arity: 1,
                },
                args: vec![Term::Variable(Variable {
                    name: "X".to_string(),
                })],
            })]);
            c.age = 0;
            c.role = ClauseRole::Axiom;
            c
        },
        // 2. Q(X, Y) | ~R(Y) - a clause with two literals (age=50, derived)
        {
            let mut c = Clause::new(vec![
                Literal::positive(Atom {
                    predicate: PredicateSymbol {
                        name: "Q".to_string(),
                        arity: 2,
                    },
                    args: vec![
                        Term::Variable(Variable { name: "X".to_string() }),
                        Term::Variable(Variable { name: "Y".to_string() }),
                    ],
                }),
                Literal::negative(Atom {
                    predicate: PredicateSymbol {
                        name: "R".to_string(),
                        arity: 1,
                    },
                    args: vec![Term::Variable(Variable { name: "Y".to_string() })],
                }),
            ]);
            c.age = 50;
            c.role = ClauseRole::Derived;
            c
        },
        // 3. mult(e, X) = X - left identity (age=0, axiom)
        {
            let mut c = Clause::new(vec![Literal::positive(Atom {
                predicate: PredicateSymbol {
                    name: "=".to_string(),
                    arity: 2,
                },
                args: vec![
                    Term::Function(
                        FunctionSymbol { name: "mult".to_string(), arity: 2 },
                        vec![
                            Term::Constant(Constant { name: "e".to_string() }),
                            Term::Variable(Variable { name: "X".to_string() }),
                        ],
                    ),
                    Term::Variable(Variable { name: "X".to_string() }),
                ],
            })]);
            c.age = 0;
            c.role = ClauseRole::Axiom;
            c
        },
        // 4. mult(inv(X), X) = e - left inverse (age=100, derived - complex)
        {
            let mut c = Clause::new(vec![Literal::positive(Atom {
                predicate: PredicateSymbol {
                    name: "=".to_string(),
                    arity: 2,
                },
                args: vec![
                    Term::Function(
                        FunctionSymbol { name: "mult".to_string(), arity: 2 },
                        vec![
                            Term::Function(
                                FunctionSymbol { name: "inv".to_string(), arity: 1 },
                                vec![Term::Variable(Variable { name: "X".to_string() })],
                            ),
                            Term::Variable(Variable { name: "X".to_string() }),
                        ],
                    ),
                    Term::Constant(Constant { name: "e".to_string() }),
                ],
            })]);
            c.age = 100;
            c.role = ClauseRole::Derived;
            c
        },
        // 5. ~equal(a, b) - negated goal (age=0, negated_conjecture)
        {
            let mut c = Clause::new(vec![Literal::negative(Atom {
                predicate: PredicateSymbol {
                    name: "equal".to_string(),
                    arity: 2,
                },
                args: vec![
                    Term::Constant(Constant { name: "a".to_string() }),
                    Term::Constant(Constant { name: "b".to_string() }),
                ],
            })]);
            c.age = 0;
            c.role = ClauseRole::NegatedConjecture;
            c
        },
    ];

    // Print clauses with their properties
    println!("Clause set:");
    for (i, clause) in clauses.iter().enumerate() {
        println!("  {}: age={:3}, role={:?}, {}",
                 i + 1, clause.age, clause.role, clause);
    }
    println!();

    // Score all clauses together
    let clause_refs: Vec<&Clause> = clauses.iter().collect();

    match scorer.score_clauses(&clause_refs) {
        Ok(scores) => {
            println!("Scores (higher = better, should be selected first):");

            // Create sorted list by score (descending)
            let mut scored: Vec<(usize, f32, &Clause)> = clauses
                .iter()
                .enumerate()
                .zip(scores.iter())
                .map(|((i, c), s)| (i, *s, c))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            for (i, (orig_idx, score, clause)) in scored.iter().enumerate() {
                println!("  {} (was {}): {:.4} - age={}, {}",
                         i + 1, orig_idx + 1, score, clause.age, clause);
            }
        }
        Err(e) => {
            println!("Error scoring clauses: {}", e);
        }
    }
}
