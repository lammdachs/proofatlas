//! Property-based tests for KBO term ordering.

use proptest::prelude::*;
use crate::logic::{Constant, FunctionSymbol, Interner, Term};
use super::{KBO, KBOConfig, Ordering};

/// Term description before interning (shared with unification proptest)
#[derive(Debug, Clone)]
enum TermDesc {
    Const(u8),
    Func(u8, Vec<TermDesc>),
}

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

fn build_term(desc: &TermDesc, interner: &mut Interner) -> Term {
    match desc {
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

fn arb_ground_triple(max_depth: u32) -> impl Strategy<Value = (Term, Term, Term, Interner)> {
    (arb_ground_term_desc(max_depth), arb_ground_term_desc(max_depth), arb_ground_term_desc(max_depth))
        .prop_map(|(d1, d2, d3)| {
            let mut interner = Interner::new();
            let t1 = build_term(&d1, &mut interner);
            let t2 = build_term(&d2, &mut interner);
            let t3 = build_term(&d3, &mut interner);
            (t1, t2, t3, interner)
        })
}

proptest! {
    /// Transitivity: if a > b and b > c, then a > c
    #[test]
    fn kbo_transitivity((t1, t2, t3, _interner) in arb_ground_triple(3)) {
        let kbo = KBO::new(KBOConfig::default());
        let cmp12 = kbo.compare(&t1, &t2);
        let cmp23 = kbo.compare(&t2, &t3);
        let cmp13 = kbo.compare(&t1, &t3);

        if cmp12 == Ordering::Greater && cmp23 == Ordering::Greater {
            prop_assert_eq!(cmp13, Ordering::Greater,
                "transitivity: t1 > t2 and t2 > t3 implies t1 > t3");
        }
        if cmp12 == Ordering::Less && cmp23 == Ordering::Less {
            prop_assert_eq!(cmp13, Ordering::Less,
                "transitivity: t1 < t2 and t2 < t3 implies t1 < t3");
        }
    }
}
