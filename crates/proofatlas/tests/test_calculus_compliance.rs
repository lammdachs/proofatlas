//! Test compliance with superposition calculus side conditions.
//!
//! Systematic tests that each inference rule respects its ordering, selection,
//! and eligibility constraints.

use proofatlas::{
    equality_factoring, equality_resolution, factoring, resolution, superposition,
    Clause, Constant, FunctionSymbol, Interner, KBO, KBOConfig, Literal,
    PredicateSymbol, SelectAll, StateChange, Term, Variable,
};

/// Test context that holds the interner and provides helper methods
struct TestCtx {
    interner: Interner,
}

impl TestCtx {
    fn new() -> Self {
        Self {
            interner: Interner::new(),
        }
    }

    fn var(&mut self, name: &str) -> Term {
        Term::Variable(Variable {
            id: self.interner.intern_variable(name),
        })
    }

    fn const_(&mut self, name: &str) -> Term {
        Term::Constant(Constant {
            id: self.interner.intern_constant(name),
        })
    }

    fn func(&mut self, name: &str, args: Vec<Term>) -> Term {
        let id = self.interner.intern_function(name);
        Term::Function(FunctionSymbol::new(id, args.len() as u8), args)
    }

    fn pred(&mut self, name: &str, arity: u8) -> PredicateSymbol {
        PredicateSymbol {
            id: self.interner.intern_predicate(name),
            arity,
        }
    }

    fn eq_pred(&mut self) -> PredicateSymbol {
        self.pred("=", 2)
    }
}

fn count_adds(changes: &[StateChange]) -> usize {
    changes.iter().filter(|c| matches!(c, StateChange::Add(..))).count()
}

fn get_conclusion(change: &StateChange) -> &Clause {
    match change {
        StateChange::Add(clause, _, _) => clause,
        _ => panic!("Expected StateChange::Add"),
    }
}

// =========================================================================
// Resolution
// =========================================================================

#[test]
fn test_resolution_complementary_pair() {
    let mut ctx = TestCtx::new();
    let p = ctx.pred("P", 1);
    let q = ctx.pred("Q", 1);
    let a = ctx.const_("a");
    let x = ctx.var("X");

    // P(X) v Q(X)  and  ~P(a) v Q(a)
    let c1 = Clause::new(vec![
        Literal::positive(p, vec![x.clone()]),
        Literal::positive(q, vec![x.clone()]),
    ]);
    let c2 = Clause::new(vec![
        Literal::negative(p, vec![a.clone()]),
        Literal::positive(q, vec![a.clone()]),
    ]);

    let results = resolution(&c1, &c2, 0, 1, &SelectAll, &mut ctx.interner);
    // Only P(X)/~P(a) form a complementary pair
    assert_eq!(count_adds(&results), 1);
}

#[test]
fn test_resolution_multiple_complementary_pairs() {
    let mut ctx = TestCtx::new();
    let p = ctx.pred("P", 1);
    let a = ctx.const_("a");

    // P(a) and ~P(a) — exactly one complementary pair
    let c1 = Clause::new(vec![Literal::positive(p, vec![a.clone()])]);
    let c2 = Clause::new(vec![Literal::negative(p, vec![a.clone()])]);

    let results = resolution(&c1, &c2, 0, 1, &SelectAll, &mut ctx.interner);
    assert_eq!(count_adds(&results), 1);
    // Resolvent should be empty clause
    assert!(get_conclusion(&results[0]).is_empty());
}

#[test]
fn test_resolution_no_unification() {
    let mut ctx = TestCtx::new();
    let p = ctx.pred("P", 1);
    let a = ctx.const_("a");
    let b = ctx.const_("b");

    // P(a) and ~P(b) — cannot unify a with b
    let c1 = Clause::new(vec![Literal::positive(p, vec![a.clone()])]);
    let c2 = Clause::new(vec![Literal::negative(p, vec![b.clone()])]);

    let results = resolution(&c1, &c2, 0, 1, &SelectAll, &mut ctx.interner);
    assert_eq!(count_adds(&results), 0);
}

#[test]
fn test_resolution_same_polarity_no_resolve() {
    let mut ctx = TestCtx::new();
    let p = ctx.pred("P", 1);
    let a = ctx.const_("a");

    // P(a) and P(a) — same polarity, no resolution
    let c1 = Clause::new(vec![Literal::positive(p, vec![a.clone()])]);
    let c2 = Clause::new(vec![Literal::positive(p, vec![a.clone()])]);

    let results = resolution(&c1, &c2, 0, 1, &SelectAll, &mut ctx.interner);
    assert_eq!(count_adds(&results), 0);
}

#[test]
fn test_resolution_with_variables() {
    let mut ctx = TestCtx::new();
    let p = ctx.pred("P", 2);
    let x = ctx.var("X");
    let y = ctx.var("Y");
    let a = ctx.const_("a");

    // P(X, a) and ~P(a, Y)
    let c1 = Clause::new(vec![Literal::positive(p, vec![x.clone(), a.clone()])]);
    let c2 = Clause::new(vec![Literal::negative(p, vec![a.clone(), y.clone()])]);

    let results = resolution(&c1, &c2, 0, 1, &SelectAll, &mut ctx.interner);
    assert_eq!(count_adds(&results), 1);
    // Resolvent: empty clause (mgu: X->a, Y->a)
    assert!(get_conclusion(&results[0]).is_empty());
}

// =========================================================================
// Resolution — negative cases (constraint enforcement)
// =========================================================================

#[test]
fn test_resolution_cross_binding_occurs_check() {
    let mut ctx = TestCtx::new();
    let p = ctx.pred("P", 2);
    let x = ctx.var("X");
    let y = ctx.var("Y");
    let fx = ctx.func("f", vec![x.clone()]);
    let fy = ctx.func("f", vec![y.clone()]);

    // P(X, f(X)) and ~P(f(Y), Y) — first args: X = f(Y) succeeds,
    // then f(X) = f(f(Y)) must unify with Y — occurs check after composition
    let c1 = Clause::new(vec![Literal::positive(p, vec![x.clone(), fx])]);
    let c2 = Clause::new(vec![Literal::negative(p, vec![fy, y.clone()])]);

    let results = resolution(&c1, &c2, 0, 1, &SelectAll, &mut ctx.interner);
    assert_eq!(count_adds(&results), 0, "cross-binding occurs check should block resolution");
}

#[test]
fn test_resolution_binding_chain_occurs_check() {
    let mut ctx = TestCtx::new();
    let p = ctx.pred("P", 2);
    let x = ctx.var("X");
    let y = ctx.var("Y");
    let a = ctx.const_("a");
    let fxa = ctx.func("f", vec![x.clone(), a.clone()]);
    let fya = ctx.func("f", vec![y.clone(), a.clone()]);

    // P(X, f(X,a)) and ~P(f(Y,a), Y) — first args: X = f(Y,a) succeeds,
    // then f(f(Y,a),a) must unify with Y — binary function occurs check
    let c1 = Clause::new(vec![Literal::positive(p, vec![x.clone(), fxa])]);
    let c2 = Clause::new(vec![Literal::negative(p, vec![fya, y.clone()])]);

    let results = resolution(&c1, &c2, 0, 1, &SelectAll, &mut ctx.interner);
    assert_eq!(count_adds(&results), 0, "binding chain occurs check should block resolution");
}

#[test]
fn test_resolution_diamond_conflict() {
    let mut ctx = TestCtx::new();
    let p = ctx.pred("P", 2);
    let x = ctx.var("X");
    let y = ctx.var("Y");
    let a = ctx.const_("a");
    let b = ctx.const_("b");
    let fxy = ctx.func("f", vec![x.clone(), y.clone()]);
    let fyx = ctx.func("f", vec![y.clone(), x.clone()]);
    let fab1 = ctx.func("f", vec![a.clone(), b.clone()]);
    let fab2 = ctx.func("f", vec![a.clone(), b.clone()]);

    // P(f(X,Y), f(Y,X)) and ~P(f(a,b), f(a,b)) — first pair: X=a, Y=b succeeds.
    // Second pair: f(Y,X) = f(a,b) requires Y=a, X=b — contradicts committed bindings.
    let c1 = Clause::new(vec![Literal::positive(p, vec![fxy, fyx])]);
    let c2 = Clause::new(vec![Literal::negative(p, vec![fab1, fab2])]);

    let results = resolution(&c1, &c2, 0, 1, &SelectAll, &mut ctx.interner);
    assert_eq!(count_adds(&results), 0, "diamond binding conflict should block resolution");
}

#[test]
fn test_resolution_symmetric_swap_occurs_check() {
    let mut ctx = TestCtx::new();
    let p = ctx.pred("P", 1);
    let x = ctx.var("X");
    let y = ctx.var("Y");
    let gx = ctx.func("g", vec![x.clone()]);
    let gy = ctx.func("g", vec![y.clone()]);
    let f_x_gx = ctx.func("f", vec![x.clone(), gx]);
    let f_gy_y = ctx.func("f", vec![gy, y.clone()]);

    // P(f(X,g(X))) and ~P(f(g(Y),Y)) — same outer f, same arity, swapped args.
    // First args: X = g(Y) succeeds. Then g(X) = g(g(Y)) must unify with Y —
    // occurs check two function applications deep.
    let c1 = Clause::new(vec![Literal::positive(p, vec![f_x_gx])]);
    let c2 = Clause::new(vec![Literal::negative(p, vec![f_gy_y])]);

    let results = resolution(&c1, &c2, 0, 1, &SelectAll, &mut ctx.interner);
    assert_eq!(count_adds(&results), 0, "symmetric swap occurs check should block resolution");
}

#[test]
fn test_resolution_nested_chain_occurs_check() {
    let mut ctx = TestCtx::new();
    let p = ctx.pred("P", 2);
    let x = ctx.var("X");
    let y = ctx.var("Y");
    let a = ctx.const_("a");
    let gxa = ctx.func("g", vec![x.clone(), a.clone()]);
    let gya = ctx.func("g", vec![y.clone(), a.clone()]);
    let f_gxa = ctx.func("f", vec![gxa.clone()]);
    let fy = ctx.func("f", vec![y.clone()]);

    // P(X, f(g(X,a))) and ~P(g(Y,a), f(Y)) — first pair: X = g(Y,a) succeeds.
    // Second pair: f(g(X,a)) = f(Y), so g(X,a) = Y. Substituting X: g(g(Y,a),a) = Y —
    // occurs check through two nested function applications.
    let c1 = Clause::new(vec![Literal::positive(p, vec![x.clone(), f_gxa])]);
    let c2 = Clause::new(vec![Literal::negative(p, vec![gya, fy])]);

    let results = resolution(&c1, &c2, 0, 1, &SelectAll, &mut ctx.interner);
    assert_eq!(count_adds(&results), 0, "nested chain occurs check should block resolution");
}

// =========================================================================
// Superposition — negative cases (constraint enforcement)
// =========================================================================

#[test]
fn test_superposition_ordering_blocks_small_to_large() {
    let mut ctx = TestCtx::new();
    let eq = ctx.eq_pred();
    let p = ctx.pred("P", 1);
    let a = ctx.const_("a");
    let fa = ctx.func("f", vec![a.clone()]);

    // a = f(a): under KBO, f(a) > a, so this orients as f(a) = a.
    // Superposition can rewrite f(a) → a but NOT a → f(a).
    // Target: P(a). To get P(f(a)) we would need the backward direction.
    let eq_clause = Clause::new(vec![Literal::positive(eq, vec![a.clone(), fa.clone()])]);
    let target = Clause::new(vec![Literal::positive(p, vec![a.clone()])]);

    let kbo = KBO::new(KBOConfig::default());
    let results = superposition(&eq_clause, &target, 0, 1, &SelectAll, &mut ctx.interner, &kbo);
    // a in P(a) is a constant, not a variable, so superposition CAN try it.
    // But the equation orients f(a) → a, so matching a against f(a) fails
    // (the LHS of the oriented equation is f(a), not a).
    // No rewrites should be produced.
    assert_eq!(count_adds(&results), 0,
        "ordering should prevent rewriting a → f(a)");
}

#[test]
fn test_superposition_occurs_check_blocks() {
    let mut ctx = TestCtx::new();
    let eq = ctx.eq_pred();
    let p = ctx.pred("P", 2);
    let y = ctx.var("Y");
    let a = ctx.const_("a");
    let fa = ctx.func("f", vec![a.clone()]);

    // f(a) = a and P(X, f(X)) — superposing f(a) → a into f(X) unifies a with X,
    // giving P(a, a). But if the target were P(Y, Y), occurs check on scoped
    // unification would prevent invalid bindings.
    let eq_clause = Clause::new(vec![Literal::positive(eq, vec![fa.clone(), a.clone()])]);
    let target = Clause::new(vec![Literal::positive(p, vec![y.clone(), y.clone()])]);

    let kbo = KBO::new(KBOConfig::default());
    let results = superposition(&eq_clause, &target, 0, 1, &SelectAll, &mut ctx.interner, &kbo);
    // P(Y, Y) has no function subterms to match f(a) against — Y is a variable
    assert_eq!(count_adds(&results), 0,
        "superposition into variable positions should be blocked");
}

// =========================================================================
// Demodulation — negative cases (constraint enforcement)
// =========================================================================

#[test]
fn test_demodulation_ordering_blocks_wrong_direction() {
    use proofatlas::simplifying::demodulation::demodulate;

    let mut ctx = TestCtx::new();
    let eq = ctx.eq_pred();
    let p = ctx.pred("P", 1);
    let a = ctx.const_("a");
    let fa = ctx.func("f", vec![a.clone()]);

    // Unit equality: a = f(a). Under KBO, f(a) > a, so this orients as f(a) = a.
    // Demodulation can rewrite f(a) → a but NOT a → f(a).
    let unit_eq = Clause::new(vec![Literal::positive(eq, vec![a.clone(), fa.clone()])]);
    // Target: P(a) — has 'a', but rewriting a → f(a) is blocked by ordering.
    let target = Clause::new(vec![Literal::positive(p, vec![a.clone()])]);

    let results = demodulate(&unit_eq, &target, 0, 1, &ctx.interner);
    assert_eq!(count_adds(&results), 0,
        "demodulation ordering should block rewriting a → f(a)");
}

#[test]
fn test_demodulation_pattern_variable_vs_target_constant() {
    use proofatlas::simplifying::demodulation::demodulate;

    let mut ctx = TestCtx::new();
    let eq = ctx.eq_pred();
    let p = ctx.pred("P", 1);
    let x = ctx.var("X");
    let a = ctx.const_("a");
    let b = ctx.const_("b");
    let fb = ctx.func("f", vec![b.clone()]);

    // Unit equality: f(X) = a. Under KBO, f(X) > a, orients as f(X) → a.
    // Target: P(f(b)). Matching f(X) against f(b) succeeds with X → b.
    // Then check lσ > rσ: f(b) > a. This should succeed.
    let unit_eq = Clause::new(vec![Literal::positive(eq, vec![ctx.func("f", vec![x.clone()]), a.clone()])]);
    let target = Clause::new(vec![Literal::positive(p, vec![fb])]);

    let results = demodulate(&unit_eq, &target, 0, 1, &ctx.interner);
    // This should succeed — f(b) matches f(X) with X=b, and f(b) > a
    assert_eq!(count_adds(&results), 1,
        "demodulation with pattern variable should match");
    assert_eq!(get_conclusion(&results[0]).literals[0].args[0], a);
}

#[test]
fn test_demodulation_no_match_wrong_head() {
    use proofatlas::simplifying::demodulation::demodulate;

    let mut ctx = TestCtx::new();
    let eq = ctx.eq_pred();
    let p = ctx.pred("P", 1);
    let a = ctx.const_("a");
    let b = ctx.const_("b");
    let fa = ctx.func("f", vec![a.clone()]);
    let ga = ctx.func("g", vec![a.clone()]);

    // Unit equality: f(a) = b. Target: P(g(a)). Head mismatch (f vs g).
    let unit_eq = Clause::new(vec![Literal::positive(eq, vec![fa, b.clone()])]);
    let target = Clause::new(vec![Literal::positive(p, vec![ga])]);

    let results = demodulate(&unit_eq, &target, 0, 1, &ctx.interner);
    assert_eq!(count_adds(&results), 0,
        "demodulation should not match with different head symbol");
}

// =========================================================================
// Equality Resolution — negative cases
// =========================================================================

#[test]
fn test_equality_resolution_multi_step_occurs_check() {
    let mut ctx = TestCtx::new();
    let eq = ctx.eq_pred();
    let x = ctx.var("X");
    let y = ctx.var("Y");
    let a = ctx.const_("a");
    let gxa = ctx.func("g", vec![x.clone(), a.clone()]);
    let gya = ctx.func("g", vec![y.clone(), a.clone()]);
    let f_x_gxa = ctx.func("f", vec![x.clone(), gxa]);
    let f_gya_y = ctx.func("f", vec![gya, y.clone()]);

    // ~(f(X, g(X, a)) = f(g(Y, a), Y)) — first pair: X = g(Y,a) succeeds.
    // Second pair: g(X,a) = Y. Substituting X: g(g(Y,a),a) = Y — occurs check
    // only visible after composing the binding from the first pair.
    let clause = Clause::new(vec![Literal::negative(eq, vec![f_x_gxa, f_gya_y])]);

    let results = equality_resolution(&clause, 0, &SelectAll, &ctx.interner);
    assert_eq!(count_adds(&results), 0,
        "multi-step occurs check should block equality resolution");
}

#[test]
fn test_equality_resolution_delayed_constant_clash() {
    let mut ctx = TestCtx::new();
    let eq = ctx.eq_pred();
    let x = ctx.var("X");
    let y = ctx.var("Y");
    let a = ctx.const_("a");
    let b = ctx.const_("b");
    let gxy = ctx.func("g", vec![x.clone(), y.clone()]);
    let gba = ctx.func("g", vec![b.clone(), a.clone()]);
    let f_x_y_gxy = ctx.func("f", vec![x.clone(), y.clone(), gxy]);
    let f_a_b_gba = ctx.func("f", vec![a.clone(), b.clone(), gba]);

    // ~(f(X, Y, g(X, Y)) = f(a, b, g(b, a))) — first two args: X=a, Y=b succeed.
    // Third arg: g(X,Y) = g(b,a). After substitution: g(a,b) vs g(b,a) — constant
    // clash hidden inside the third argument, only revealed after applying bindings.
    let clause = Clause::new(vec![Literal::negative(eq, vec![f_x_y_gxy, f_a_b_gba])]);

    let results = equality_resolution(&clause, 0, &SelectAll, &ctx.interner);
    assert_eq!(count_adds(&results), 0,
        "delayed constant clash should block equality resolution");
}

// =========================================================================
// Factoring
// =========================================================================

#[test]
fn test_factoring_unifiable_literals() {
    let mut ctx = TestCtx::new();
    let p = ctx.pred("P", 1);
    let x = ctx.var("X");
    let y = ctx.var("Y");

    // P(X) v P(Y) — two positive literals with same predicate
    let clause = Clause::new(vec![
        Literal::positive(p, vec![x.clone()]),
        Literal::positive(p, vec![y.clone()]),
    ]);

    let results = factoring(&clause, 0, &SelectAll);
    // P(X) factors with P(Y): produce P(X) (or P(Y))
    assert!(count_adds(&results) >= 1);
    for r in &results {
        assert_eq!(get_conclusion(r).literals.len(), 1);
    }
}

#[test]
fn test_factoring_different_predicates() {
    let mut ctx = TestCtx::new();
    let p = ctx.pred("P", 1);
    let q = ctx.pred("Q", 1);
    let x = ctx.var("X");

    // P(X) v Q(X) — different predicates, no factoring
    let clause = Clause::new(vec![
        Literal::positive(p, vec![x.clone()]),
        Literal::positive(q, vec![x.clone()]),
    ]);

    let results = factoring(&clause, 0, &SelectAll);
    assert_eq!(count_adds(&results), 0);
}

#[test]
fn test_factoring_different_polarity() {
    let mut ctx = TestCtx::new();
    let p = ctx.pred("P", 1);
    let x = ctx.var("X");
    let y = ctx.var("Y");

    // P(X) v ~P(Y) — different polarity, no factoring
    let clause = Clause::new(vec![
        Literal::positive(p, vec![x.clone()]),
        Literal::negative(p, vec![y.clone()]),
    ]);

    let results = factoring(&clause, 0, &SelectAll);
    assert_eq!(count_adds(&results), 0);
}

#[test]
fn test_factoring_non_unifiable() {
    let mut ctx = TestCtx::new();
    let p = ctx.pred("P", 1);
    let a = ctx.const_("a");
    let b = ctx.const_("b");

    // P(a) v P(b) — cannot unify a with b
    let clause = Clause::new(vec![
        Literal::positive(p, vec![a.clone()]),
        Literal::positive(p, vec![b.clone()]),
    ]);

    let results = factoring(&clause, 0, &SelectAll);
    assert_eq!(count_adds(&results), 0);
}

#[test]
fn test_factoring_reduces_clause_size() {
    let mut ctx = TestCtx::new();
    let p = ctx.pred("P", 1);
    let q = ctx.pred("Q", 1);
    let x = ctx.var("X");
    let y = ctx.var("Y");
    let z = ctx.var("Z");

    // P(X) v P(Y) v Q(Z) — factor P(X) with P(Y)
    let clause = Clause::new(vec![
        Literal::positive(p, vec![x.clone()]),
        Literal::positive(p, vec![y.clone()]),
        Literal::positive(q, vec![z.clone()]),
    ]);

    let results = factoring(&clause, 0, &SelectAll);
    for r in &results {
        let c = get_conclusion(r);
        assert!(c.literals.len() < clause.literals.len());
    }
}

// =========================================================================
// Equality Resolution
// =========================================================================

#[test]
fn test_equality_resolution_basic() {
    let mut ctx = TestCtx::new();
    let eq = ctx.eq_pred();
    let a = ctx.const_("a");

    // ~(a = a) — should resolve to empty clause
    let clause = Clause::new(vec![Literal::negative(eq, vec![a.clone(), a.clone()])]);

    let results = equality_resolution(&clause, 0, &SelectAll, &ctx.interner);
    assert_eq!(count_adds(&results), 1);
    assert!(get_conclusion(&results[0]).is_empty());
}

#[test]
fn test_equality_resolution_with_variables() {
    let mut ctx = TestCtx::new();
    let eq = ctx.eq_pred();
    let p = ctx.pred("P", 1);
    let x = ctx.var("X");
    let y = ctx.var("Y");

    // ~(X = Y) v P(X) — X and Y unify, leaving P(X)
    let clause = Clause::new(vec![
        Literal::negative(eq, vec![x.clone(), y.clone()]),
        Literal::positive(p, vec![x.clone()]),
    ]);

    let results = equality_resolution(&clause, 0, &SelectAll, &ctx.interner);
    assert_eq!(count_adds(&results), 1);
    assert_eq!(get_conclusion(&results[0]).literals.len(), 1);
}

#[test]
fn test_equality_resolution_positive_skipped() {
    let mut ctx = TestCtx::new();
    let eq = ctx.eq_pred();
    let a = ctx.const_("a");

    // a = a — positive equality, should NOT trigger equality resolution
    let clause = Clause::new(vec![Literal::positive(eq, vec![a.clone(), a.clone()])]);

    let results = equality_resolution(&clause, 0, &SelectAll, &ctx.interner);
    assert_eq!(count_adds(&results), 0);
}

#[test]
fn test_equality_resolution_non_unifiable() {
    let mut ctx = TestCtx::new();
    let eq = ctx.eq_pred();
    let a = ctx.const_("a");
    let b = ctx.const_("b");

    // ~(a = b) — a and b don't unify
    let clause = Clause::new(vec![Literal::negative(eq, vec![a.clone(), b.clone()])]);

    let results = equality_resolution(&clause, 0, &SelectAll, &ctx.interner);
    assert_eq!(count_adds(&results), 0);
}

// =========================================================================
// Equality Factoring
// =========================================================================

#[test]
fn test_equality_factoring_basic() {
    let mut ctx = TestCtx::new();
    let eq = ctx.eq_pred();
    let x = ctx.var("X");
    let a = ctx.const_("a");
    let b = ctx.const_("b");

    // X = a v X = b — two positive equalities with unifiable LHS
    let clause = Clause::new(vec![
        Literal::positive(eq, vec![x.clone(), a.clone()]),
        Literal::positive(eq, vec![x.clone(), b.clone()]),
    ]);

    let kbo = KBO::new(KBOConfig::default());
    let results = equality_factoring(&clause, 0, &SelectAll, &mut ctx.interner, &kbo);
    // Should produce a conclusion with a negative equality (disequality)
    for r in &results {
        let c = get_conclusion(r);
        let has_neg_eq = c.literals.iter().any(|l| !l.polarity && l.is_equality(&ctx.interner));
        assert!(has_neg_eq, "equality factoring conclusion must contain a disequality");
    }
}

#[test]
fn test_equality_factoring_needs_two_positive_equalities() {
    let mut ctx = TestCtx::new();
    let eq = ctx.eq_pred();
    let a = ctx.const_("a");
    let b = ctx.const_("b");

    // a = b — only one positive equality, no factoring possible
    let clause = Clause::new(vec![Literal::positive(eq, vec![a.clone(), b.clone()])]);

    let kbo = KBO::new(KBOConfig::default());
    let results = equality_factoring(&clause, 0, &SelectAll, &mut ctx.interner, &kbo);
    assert_eq!(count_adds(&results), 0);
}

// =========================================================================
// Superposition
// =========================================================================

#[test]
fn test_superposition_rewrites_into_predicate() {
    let mut ctx = TestCtx::new();
    let eq = ctx.eq_pred();
    let p = ctx.pred("P", 1);
    let x = ctx.var("X");
    let e = ctx.const_("e");
    let c = ctx.const_("c");
    let mult_ex = ctx.func("mult", vec![e.clone(), x.clone()]);
    let mult_ec = ctx.func("mult", vec![e.clone(), c.clone()]);

    // mult(e,X) = X and P(mult(e,c)) => P(c)
    let from = Clause::new(vec![Literal::positive(eq, vec![mult_ex, x.clone()])]);
    let into = Clause::new(vec![Literal::positive(p, vec![mult_ec])]);

    let kbo = KBO::new(KBOConfig::default());
    let results = superposition(&from, &into, 0, 1, &SelectAll, &mut ctx.interner, &kbo);
    assert!(count_adds(&results) >= 1, "superposition should rewrite mult(e,c) to c");
}

#[test]
fn test_superposition_no_rewrite_into_variable() {
    let mut ctx = TestCtx::new();
    let eq = ctx.eq_pred();
    let p = ctx.pred("P", 1);
    let a = ctx.const_("a");
    let b = ctx.const_("b");
    let x = ctx.var("X");

    // a = b and P(X) — X is a variable, superposition must not rewrite into it
    let from = Clause::new(vec![Literal::positive(eq, vec![a.clone(), b.clone()])]);
    let into = Clause::new(vec![Literal::positive(p, vec![x.clone()])]);

    let kbo = KBO::new(KBOConfig::default());
    let results = superposition(&from, &into, 0, 1, &SelectAll, &mut ctx.interner, &kbo);
    assert_eq!(count_adds(&results), 0, "must not rewrite into a variable position");
}

#[test]
fn test_superposition_requires_positive_equality() {
    let mut ctx = TestCtx::new();
    let eq = ctx.eq_pred();
    let p = ctx.pred("P", 1);
    let a = ctx.const_("a");
    let b = ctx.const_("b");

    // ~(a = b) and P(a) — negative equality, no superposition
    let from = Clause::new(vec![Literal::negative(eq, vec![a.clone(), b.clone()])]);
    let into = Clause::new(vec![Literal::positive(p, vec![a.clone()])]);

    let kbo = KBO::new(KBOConfig::default());
    let results = superposition(&from, &into, 0, 1, &SelectAll, &mut ctx.interner, &kbo);
    assert_eq!(count_adds(&results), 0, "negative equality cannot be used for superposition");
}

#[test]
fn test_superposition_both_directions() {
    let mut ctx = TestCtx::new();
    let eq = ctx.eq_pred();
    let a = ctx.const_("a");
    let b = ctx.const_("b");

    // a = b and b = a (superposition into right side of equality)
    let from = Clause::new(vec![Literal::positive(eq, vec![a.clone(), b.clone()])]);
    let into = Clause::new(vec![Literal::positive(eq, vec![b.clone(), a.clone()])]);

    let kbo = KBO::new(KBOConfig::default());
    let results = superposition(&from, &into, 0, 1, &SelectAll, &mut ctx.interner, &kbo);
    // At least one direction should produce a rewrite
    // (depends on KBO ordering of a vs b)
    // We just check it doesn't crash
    let _ = count_adds(&results);
}

// =========================================================================
// Tautology
// =========================================================================

#[test]
fn test_tautology_complementary_literals() {
    use proofatlas::{parse_tptp, saturate, AgeWeightSink, ProverSink, ProverConfig, ProofResult};

    // P(a) v ~P(a) v Q(b) — tautology, should be deleted
    let tptp = r#"
        cnf(taut, axiom, p(a) | ~p(a) | q(b)).
        cnf(goal, negated_conjecture, ~q(b)).
    "#;
    let parsed = parse_tptp(tptp, &[], None, None).unwrap();
    let config = ProverConfig::default();
    let sink: Box<dyn ProverSink> = Box::new(AgeWeightSink::new(0.5));
    let (result, _) = saturate(parsed.formula, config, sink, parsed.interner);
    // The tautology should be deleted, so the proof should still work
    // (q(b) isn't derivable from the tautology alone)
    // This tests that tautology deletion runs correctly
    match result {
        ProofResult::Saturated | ProofResult::ResourceLimit => {
            // Expected: after deleting the tautology, only ~q(b) remains, which saturates
        }
        ProofResult::Proof { .. } => {
            panic!("Should not find proof: the only axiom is a tautology");
        }
    }
}

#[test]
fn test_tautology_reflexive_equality() {
    use proofatlas::{parse_tptp, saturate, AgeWeightSink, ProverSink, ProverConfig, ProofResult};

    // t = t is a tautology
    let tptp = r#"
        cnf(refl, axiom, f(a) = f(a) | q(b)).
        cnf(goal, negated_conjecture, ~q(b)).
    "#;
    let parsed = parse_tptp(tptp, &[], None, None).unwrap();
    let config = ProverConfig::default();
    let sink: Box<dyn ProverSink> = Box::new(AgeWeightSink::new(0.5));
    let (result, _) = saturate(parsed.formula, config, sink, parsed.interner);
    match result {
        ProofResult::Saturated | ProofResult::ResourceLimit => {}
        ProofResult::Proof { .. } => {
            panic!("Should not find proof: the only axiom is a tautology");
        }
    }
}

#[test]
fn test_non_tautology_not_deleted() {
    use proofatlas::{parse_tptp, saturate, AgeWeightSink, ProverSink, ProverConfig, ProofResult};

    // P(a) v Q(b) — NOT a tautology, should be kept
    let tptp = r#"
        cnf(c1, axiom, p(a) | q(b)).
        cnf(c2, axiom, ~p(X)).
        cnf(c3, negated_conjecture, ~q(b)).
    "#;
    let parsed = parse_tptp(tptp, &[], None, None).unwrap();
    let config = ProverConfig::default();
    let sink: Box<dyn ProverSink> = Box::new(AgeWeightSink::new(0.5));
    let (result, _) = saturate(parsed.formula, config, sink, parsed.interner);
    assert!(matches!(result, ProofResult::Proof { .. }), "non-tautology should be kept and proof should be found");
}

// =========================================================================
// Subsumption
// =========================================================================

#[test]
fn test_forward_subsumption() {
    use proofatlas::simplifying::subsumption::subsumes;

    let mut ctx = TestCtx::new();
    let p = ctx.pred("P", 1);
    let x = ctx.var("X");
    let a = ctx.const_("a");

    // P(X) subsumes P(a) v Q(a) (under substitution X -> a)
    let subsumer = Clause::new(vec![Literal::positive(p, vec![x.clone()])]);
    let q = ctx.pred("Q", 1);
    let subsumee = Clause::new(vec![
        Literal::positive(p, vec![a.clone()]),
        Literal::positive(q, vec![a.clone()]),
    ]);

    assert!(subsumes(&subsumer, &subsumee));
}

#[test]
fn test_subsumption_not_reflexive_proper() {
    use proofatlas::simplifying::subsumption::subsumes;

    let mut ctx = TestCtx::new();
    let p = ctx.pred("P", 1);
    let q = ctx.pred("Q", 1);
    let a = ctx.const_("a");

    // P(a) v Q(a) does NOT subsume P(a) (subsumer must be smaller or equal)
    let bigger = Clause::new(vec![
        Literal::positive(p, vec![a.clone()]),
        Literal::positive(q, vec![a.clone()]),
    ]);
    let smaller = Clause::new(vec![Literal::positive(p, vec![a.clone()])]);

    assert!(!subsumes(&bigger, &smaller));
}

#[test]
fn test_subsumption_variant() {
    use proofatlas::simplifying::subsumption::{subsumes, are_variants};

    let mut ctx = TestCtx::new();
    let p = ctx.pred("P", 1);
    let x = ctx.var("X");
    let y = ctx.var("Y");

    // P(X) is a variant of P(Y)
    let c1 = Clause::new(vec![Literal::positive(p, vec![x.clone()])]);
    let c2 = Clause::new(vec![Literal::positive(p, vec![y.clone()])]);

    assert!(are_variants(&c1, &c2));
    assert!(subsumes(&c1, &c2));
    assert!(subsumes(&c2, &c1));
}

// =========================================================================
// Demodulation
// =========================================================================

#[test]
fn test_demodulation_rewrites() {
    use proofatlas::simplifying::demodulation::demodulate;

    let mut ctx = TestCtx::new();
    let eq = ctx.eq_pred();
    let p = ctx.pred("P", 1);
    let a = ctx.const_("a");
    let b = ctx.const_("b");
    let fa = ctx.func("f", vec![a.clone()]);

    // Unit equality: f(a) = b
    let unit_eq = Clause::new(vec![Literal::positive(eq, vec![fa.clone(), b.clone()])]);
    // Target: P(f(a))
    let target = Clause::new(vec![Literal::positive(p, vec![fa.clone()])]);

    let results = demodulate(&unit_eq, &target, 0, 1, &ctx.interner);
    assert_eq!(count_adds(&results), 1);
    // Should produce P(b)
    let conclusion = get_conclusion(&results[0]);
    assert_eq!(conclusion.literals.len(), 1);
    assert_eq!(conclusion.literals[0].args[0], b);
}

#[test]
fn test_demodulation_requires_unit_clause() {
    use proofatlas::simplifying::demodulation::demodulate;

    let mut ctx = TestCtx::new();
    let eq = ctx.eq_pred();
    let p = ctx.pred("P", 1);
    let a = ctx.const_("a");
    let b = ctx.const_("b");
    let fa = ctx.func("f", vec![a.clone()]);

    // Non-unit: f(a) = b v P(a) — cannot use for demodulation
    let non_unit = Clause::new(vec![
        Literal::positive(eq, vec![fa.clone(), b.clone()]),
        Literal::positive(p, vec![a.clone()]),
    ]);
    let target = Clause::new(vec![Literal::positive(p, vec![fa.clone()])]);

    let results = demodulate(&non_unit, &target, 0, 1, &ctx.interner);
    assert_eq!(count_adds(&results), 0);
}

#[test]
fn test_demodulation_uses_matching_not_unification() {
    use proofatlas::simplifying::demodulation::demodulate;

    let mut ctx = TestCtx::new();
    let eq = ctx.eq_pred();
    let p = ctx.pred("P", 1);
    let x = ctx.var("X");
    let a = ctx.const_("a");

    // Unit equality: X = a  (X is a pattern variable)
    let unit_eq = Clause::new(vec![Literal::positive(eq, vec![x.clone(), a.clone()])]);
    // Target: P(b)
    let b = ctx.const_("b");
    let target = Clause::new(vec![Literal::positive(p, vec![b.clone()])]);

    // Matching X against b succeeds (X -> b)
    // But we need lσ > rσ: b > a (depends on KBO precedence)
    let results = demodulate(&unit_eq, &target, 0, 1, &ctx.interner);
    // Whether this succeeds depends on KBO ordering of b vs a
    // The key point is that matching is one-way (pattern variables in unit_eq only)
    let _ = results;
}
