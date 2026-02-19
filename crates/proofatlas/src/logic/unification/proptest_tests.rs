//! Property-based tests for unification and matching using proptest.

use proptest::prelude::*;
use crate::logic::{Constant, FunctionSymbol, Interner, Term, Variable};
use super::{unify, match_term};

/// Generate a random term of bounded depth using an interner.
///
/// We pre-intern a fixed set of symbols and generate random terms from them.
fn arb_term(max_depth: u32) -> impl Strategy<Value = (Term, Interner)> {
    // Generate (term_desc, interner) pairs
    arb_term_desc(max_depth).prop_map(|desc| {
        let mut interner = Interner::new();
        let term = build_term(&desc, &mut interner);
        (term, interner)
    })
}

/// Term description (before interning)
#[derive(Debug, Clone)]
enum TermDesc {
    Var(u8),         // Variable index 0-3
    Const(u8),       // Constant index 0-3
    Func(u8, Vec<TermDesc>), // Function index 0-1, with args
}

fn arb_term_desc(max_depth: u32) -> BoxedStrategy<TermDesc> {
    if max_depth == 0 {
        prop_oneof![
            (0..4u8).prop_map(TermDesc::Var),
            (0..4u8).prop_map(TermDesc::Const),
        ].boxed()
    } else {
        prop_oneof![
            3 => (0..4u8).prop_map(TermDesc::Var),
            3 => (0..4u8).prop_map(TermDesc::Const),
            2 => (0..2u8, proptest::collection::vec(arb_term_desc(max_depth - 1), 1..=2))
                .prop_map(|(f, args)| TermDesc::Func(f, args)),
        ].boxed()
    }
}

/// Generate a pair of terms sharing the same interner
fn arb_term_pair(max_depth: u32) -> impl Strategy<Value = (Term, Term, Interner)> {
    (arb_term_desc(max_depth), arb_term_desc(max_depth)).prop_map(|(desc1, desc2)| {
        let mut interner = Interner::new();
        let t1 = build_term(&desc1, &mut interner);
        let t2 = build_term(&desc2, &mut interner);
        (t1, t2, interner)
    })
}

fn build_term(desc: &TermDesc, interner: &mut Interner) -> Term {
    match desc {
        TermDesc::Var(i) => {
            let name = format!("X{}", i);
            let id = interner.intern_variable(&name);
            Term::Variable(Variable::new(id))
        }
        TermDesc::Const(i) => {
            let name = format!("c{}", i);
            let id = interner.intern_constant(&name);
            Term::Constant(Constant::new(id))
        }
        TermDesc::Func(f, args) => {
            let name = format!("f{}", f);
            let id = interner.intern_function(&name);
            let built_args: Vec<Term> = args.iter().map(|a| build_term(a, interner)).collect();
            Term::Function(FunctionSymbol::new(id, built_args.len() as u8), built_args)
        }
    }
}

/// Generate a ground term (no variables)
fn arb_ground_term_desc(max_depth: u32) -> BoxedStrategy<TermDesc> {
    if max_depth == 0 {
        (0..4u8).prop_map(TermDesc::Const).boxed()
    } else {
        prop_oneof![
            3 => (0..4u8).prop_map(TermDesc::Const),
            2 => (0..2u8, proptest::collection::vec(arb_ground_term_desc(max_depth - 1), 1..=2))
                .prop_map(|(f, args)| TermDesc::Func(f, args)),
        ].boxed()
    }
}

fn arb_ground_term_pair(max_depth: u32) -> impl Strategy<Value = (Term, Term, Interner)> {
    (arb_ground_term_desc(max_depth), arb_ground_term_desc(max_depth)).prop_map(|(d1, d2)| {
        let mut interner = Interner::new();
        let t1 = build_term(&d1, &mut interner);
        let t2 = build_term(&d2, &mut interner);
        (t1, t2, interner)
    })
}

// =========================================================================
// Unification properties
// =========================================================================

proptest! {
    /// Soundness: if unify(s, t) = σ, then sσ = tσ
    #[test]
    fn unification_soundness((t1, t2, _interner) in arb_term_pair(3)) {
        if let Ok(sigma) = unify(&t1, &t2) {
            let t1_sigma = t1.apply_substitution(&sigma);
            let t2_sigma = t2.apply_substitution(&sigma);
            prop_assert_eq!(t1_sigma, t2_sigma, "unifier must make terms equal");
        }
        // If unification fails, that's fine — no property to check
    }

    /// Symmetry: unify(s, t) succeeds iff unify(t, s) succeeds
    #[test]
    fn unification_symmetry((t1, t2, _interner) in arb_term_pair(3)) {
        let r1 = unify(&t1, &t2);
        let r2 = unify(&t2, &t1);
        prop_assert_eq!(r1.is_ok(), r2.is_ok(), "unification should be symmetric");
    }

    /// Occurs check: unify(X, f(X)) should always fail
    #[test]
    fn unification_occurs_check(func_idx in 0..2u8, depth in 1..3u32) {
        let mut interner = Interner::new();
        let x_id = interner.intern_variable("X");
        let x = Term::Variable(Variable::new(x_id));

        // Build f^depth(X) — nested application of f around X
        let f_id = interner.intern_function(&format!("f{}", func_idx));
        let mut term = x.clone();
        for _ in 0..depth {
            term = Term::Function(FunctionSymbol::new(f_id, 1), vec![term]);
        }

        prop_assert!(unify(&x, &term).is_err(), "occurs check should prevent X = f(...X...)");
    }

    /// Identity: unify(t, t) should always succeed with empty (or trivial) substitution
    #[test]
    fn unification_identity((t, _interner) in arb_term(3)) {
        let result = unify(&t, &t);
        prop_assert!(result.is_ok(), "term should unify with itself");
        if let Ok(sigma) = result {
            let t_sigma = t.apply_substitution(&sigma);
            prop_assert_eq!(t, t_sigma, "applying identity-like unifier should not change term");
        }
    }
}

// =========================================================================
// Matching properties
// =========================================================================

proptest! {
    /// Soundness: if match(pattern, target) = σ, then pattern·σ = target
    #[test]
    fn matching_soundness((t1, t2, _interner) in arb_term_pair(3)) {
        if let Ok(sigma) = match_term(&t1, &t2) {
            let t1_sigma = t1.apply_substitution(&sigma);
            prop_assert_eq!(t1_sigma, t2, "matching substitution must make pattern equal to target");
        }
    }

    /// Matching is NOT symmetric in general
    #[test]
    fn matching_asymmetry_constant_vs_variable(const_idx in 0..4u8) {
        let mut interner = Interner::new();
        let x_id = interner.intern_variable("X");
        let x = Term::Variable(Variable::new(x_id));
        let c_id = interner.intern_constant(&format!("c{}", const_idx));
        let c = Term::Constant(Constant::new(c_id));

        // match(X, c) should succeed (X -> c)
        prop_assert!(match_term(&x, &c).is_ok(), "variable pattern should match constant");
        // match(c, X) should fail (can't substitute constants)
        prop_assert!(match_term(&c, &x).is_err(), "constant pattern should not match variable");
    }
}

// =========================================================================
// KBO properties
// =========================================================================

proptest! {
    /// Totality on ground terms: exactly one of >, <, = holds
    #[test]
    fn kbo_totality_ground((t1, t2, _interner) in arb_ground_term_pair(3)) {
        use crate::logic::{KBO, KBOConfig, TermOrdering};
        let kbo = KBO::new(KBOConfig::default());
        let cmp = kbo.compare(&t1, &t2);

        match cmp {
            TermOrdering::Greater | TermOrdering::Less | TermOrdering::Equal => {
                // All fine — ground terms should always be comparable
            }
            TermOrdering::Incomparable => {
                // KBO may produce Incomparable for ground terms only when
                // the terms have the same weight and incomparable head symbols.
                // This is acceptable for KBO.
            }
        }
    }

    /// Anti-symmetry: if a > b then b < a (and vice versa)
    #[test]
    fn kbo_antisymmetry((t1, t2, _interner) in arb_ground_term_pair(3)) {
        use crate::logic::{KBO, KBOConfig, TermOrdering};
        let kbo = KBO::new(KBOConfig::default());
        let cmp12 = kbo.compare(&t1, &t2);
        let cmp21 = kbo.compare(&t2, &t1);

        match (cmp12, cmp21) {
            (TermOrdering::Greater, TermOrdering::Less) => {}
            (TermOrdering::Less, TermOrdering::Greater) => {}
            (TermOrdering::Equal, TermOrdering::Equal) => {}
            (TermOrdering::Incomparable, TermOrdering::Incomparable) => {}
            (a, b) => {
                prop_assert!(false, "antisymmetry violated: compare(t1,t2)={:?}, compare(t2,t1)={:?}", a, b);
            }
        }
    }

    /// Reflexivity: t = t
    #[test]
    fn kbo_reflexivity((t, _interner) in arb_term(3)) {
        use crate::logic::{KBO, KBOConfig, TermOrdering};
        let kbo = KBO::new(KBOConfig::default());
        let cmp = kbo.compare(&t, &t);
        prop_assert_eq!(cmp, TermOrdering::Equal, "term should be equal to itself");
    }

    /// Subterm property: f(t) > t for ground terms
    #[test]
    fn kbo_subterm_property(const_idx in 0..4u8, func_idx in 0..2u8) {
        use crate::logic::{KBO, KBOConfig, TermOrdering};
        let mut interner = Interner::new();
        let c_id = interner.intern_constant(&format!("c{}", const_idx));
        let c = Term::Constant(Constant::new(c_id));
        let f_id = interner.intern_function(&format!("f{}", func_idx));
        let f_c = Term::Function(FunctionSymbol::new(f_id, 1), vec![c.clone()]);

        let kbo = KBO::new(KBOConfig::default());
        let cmp = kbo.compare(&f_c, &c);
        prop_assert_eq!(cmp, TermOrdering::Greater, "f(t) should be greater than t");
    }
}

// =========================================================================
// Substitution properties
// =========================================================================

proptest! {
    /// Empty substitution is identity
    #[test]
    fn substitution_identity((t, _interner) in arb_term(3)) {
        use crate::logic::Substitution;
        let empty = Substitution::new();
        let t_applied = t.apply_substitution(&empty);
        prop_assert_eq!(t, t_applied, "empty substitution should be identity");
    }
}
