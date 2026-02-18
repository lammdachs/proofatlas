//! Structured TPTP regression suite.
//!
//! Organized by calculus features (A), expected result (B), and structural property (C).
//! Every unsatisfiable problem runs verify_proof() automatically.

use proofatlas::{
    parse_tptp_file, AgeWeightSink, LiteralSelectionStrategy, ProofResult, Prover, ProverConfig,
    ProverSink,
};
use std::time::Duration;

fn create_sink() -> Box<dyn ProverSink> {
    Box::new(AgeWeightSink::new(0.5))
}

fn prove_problem(path: &str) -> (ProofResult, Prover) {
    let parsed = parse_tptp_file(path, &[], None, None).expect("Failed to parse TPTP file");
    let config = ProverConfig {
        max_clauses: 10000,
        max_iterations: 10000,
        timeout: Duration::from_secs(10),
        literal_selection: LiteralSelectionStrategy::Sel0,
        ..Default::default()
    };
    let mut prover = Prover::new(parsed.formula.clauses, config, create_sink(), parsed.interner);
    let result = prover.prove();
    (result, prover)
}

/// Macro for unsatisfiable problems: expects proof and verifies it.
macro_rules! tptp_proof {
    ($name:ident, $file:expr) => {
        #[test]
        fn $name() {
            let (result, prover) = prove_problem($file);
            match result {
                ProofResult::Proof { empty_clause_idx } => {
                    prover
                        .verify_proof(empty_clause_idx)
                        .unwrap_or_else(|e| panic!("{}: verification failed: {}", stringify!($name), e));
                }
                other => panic!("{}: expected Proof, got {:?}", stringify!($name), other),
            }
        }
    };
}

/// Macro for satisfiable problems: must NOT find a proof.
macro_rules! tptp_no_proof {
    ($name:ident, $file:expr) => {
        #[test]
        fn $name() {
            let (result, _) = prove_problem($file);
            assert!(
                !matches!(result, ProofResult::Proof { .. }),
                "{}: must NOT find proof for satisfiable problem",
                stringify!($name)
            );
        }
    };
}

// =========================================================================
// A1: Pure propositional (resolution + factoring, no variables, no equality)
// =========================================================================

tptp_proof!(a1_propositional_unsat, "tests/problems/propositional_unsat.p");
tptp_proof!(a1_trivial_empty, "tests/problems/trivial_empty.p");

tptp_no_proof!(a1_propositional_sat, "tests/problems/propositional_sat.p");

// =========================================================================
// A2: First-order without equality (resolution + factoring + unification)
// =========================================================================

tptp_proof!(a2_fo_no_equality, "tests/problems/fo_no_equality.p");
tptp_proof!(a2_fo_chain, "tests/problems/fo_chain.p");
tptp_proof!(a2_horn_unsat, "tests/problems/horn_unsat.p");
tptp_proof!(a2_non_horn, "tests/problems/non_horn.p");

// =========================================================================
// A3: Pure equality (superposition, demodulation, eq resolution, eq factoring)
// =========================================================================

tptp_proof!(a3_pure_equality_unsat, "tests/problems/pure_equality_unsat.p");
tptp_proof!(a3_pure_equality_symmetry, "tests/problems/pure_equality_symmetry.p");
tptp_proof!(a3_pure_equality_transitivity, "tests/problems/pure_equality_transitivity.p");
tptp_proof!(a3_equality_factoring_needed, "tests/problems/equality_factoring_needed.p");

tptp_no_proof!(a3_satisfiable_equality, "tests/problems/satisfiable_equality.p");

// =========================================================================
// A4: Mixed equality + predicates (full calculus)
// =========================================================================

tptp_proof!(a4_mixed_eq_pred, "tests/problems/mixed_eq_pred.p");
tptp_proof!(a4_unit_demod, "tests/problems/unit_demod.p");

// =========================================================================
// Complex unification (exercises scoped unification, variable propagation)
// =========================================================================

// 4-level variable chain: binding X forces Y forces Z forces W
tptp_proof!(unif_variable_chain, "tests/problems/unif_variable_chain.p");
// Same variable X appears 3 times in one literal — consistent binding required
tptp_proof!(unif_repeated_variable, "tests/problems/unif_repeated_variable.p");
// Variables shared across literals: 3-step resolution with binding propagation
tptp_proof!(unif_shared_variables, "tests/problems/unif_shared_variables.p");
// Same variable name X in two different clauses — tests scoped variable separation
tptp_proof!(unif_same_var_names, "tests/problems/unif_same_var_names.p");
// Deep nested subterm matching for superposition: f(g(X)) inside h(f(g(a)),b)
tptp_proof!(unif_nested_equality, "tests/problems/unif_nested_equality.p");
// 4-ary predicate with cross-referencing variables — simultaneous multi-variable binding
tptp_proof!(unif_multi_arity, "tests/problems/unif_multi_arity.p");

// =========================================================================
// Group theory (A3/A4, exercises full equality reasoning)
// =========================================================================

tptp_proof!(grp_right_identity, "tests/problems/right_identity.p");
tptp_proof!(grp_uniqueness_of_identity, "tests/problems/uniqueness_of_identity.p");
tptp_proof!(grp_right_inverse, "tests/problems/right_inverse.p");
tptp_proof!(grp_uniqueness_of_inverse, "tests/problems/uniqueness_of_inverse.p");
tptp_proof!(grp_inverse_of_identity, "tests/problems/inverse_of_identity.p");
tptp_proof!(grp_inverse_of_inverse, "tests/problems/inverse_of_inverse.p");
tptp_proof!(grp_order_2_implies_abelian, "tests/problems/order_2_implies_abelian.p");
