//! Re-exports for backward compatibility.
//! Canonical locations: crate::state, crate::simplifying, crate::generating

pub use crate::state::{
    StateChange, EventLog, ClauseSet, ClauseNotification,
    SimplifyingInference, GeneratingInference, Derivation,
};

pub use crate::simplifying::{TautologyRule, SubsumptionRule, DemodulationRule};
pub use crate::generating::{
    ResolutionRule, SuperpositionRule, FactoringRule,
    EqualityResolutionRule, EqualityFactoringRule,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::{Clause, Constant, Interner, Literal, PredicateSymbol, Term, Variable};

    /// Test context with interner for creating FOL terms
    struct TestContext {
        interner: Interner,
    }

    impl TestContext {
        fn new() -> Self {
            TestContext {
                interner: Interner::new(),
            }
        }

        fn make_clause(&self, literals: Vec<Literal>) -> Clause {
            Clause::new(literals)
        }

        fn make_literal(&mut self, pred: &str, args: Vec<Term>, positive: bool) -> Literal {
            let pred_id = self.interner.intern_predicate(pred);
            let predicate = PredicateSymbol::new(pred_id, args.len() as u8);
            if positive {
                Literal::positive(predicate, args)
            } else {
                Literal::negative(predicate, args)
            }
        }

        fn var(&mut self, name: &str) -> Term {
            let var_id = self.interner.intern_variable(name);
            Term::Variable(Variable::new(var_id))
        }

        fn const_(&mut self, name: &str) -> Term {
            let const_id = self.interner.intern_constant(name);
            Term::Constant(Constant::new(const_id))
        }
    }

    #[test]
    fn test_tautology_rule() {
        let mut ctx = TestContext::new();

        let x1 = ctx.var("X");
        let a1 = ctx.const_("a");
        let x2 = ctx.var("X");
        let x3 = ctx.var("X");

        let lit1 = ctx.make_literal("P", vec![x1], true);
        let lit2 = ctx.make_literal("Q", vec![a1], false);
        let lit3 = ctx.make_literal("P", vec![x2], true);
        let lit4 = ctx.make_literal("P", vec![x3], false);

        let rule = TautologyRule::new(&ctx.interner);
        let clauses: Vec<Clause> = Vec::new();

        // Non-tautology
        let clause = ctx.make_clause(vec![lit1, lit2]);
        let changes = rule.simplify_forward(0, &clause, &clauses, &ctx.interner);
        assert!(changes.is_empty());

        // Tautology (complementary literals)
        let tautology = ctx.make_clause(vec![lit3, lit4]);
        let changes = rule.simplify_forward(0, &tautology, &clauses, &ctx.interner);
        assert_eq!(changes.len(), 1);
        assert!(matches!(changes[0], StateChange::Delete { clause_idx: 0, .. }));
    }

    #[test]
    fn test_indexset_operations() {
        use indexmap::IndexSet;

        let mut set: IndexSet<usize> = IndexSet::new();
        set.insert(0);
        set.insert(2);
        set.insert(5);

        assert_eq!(set.len(), 3);
        assert!(set.contains(&0));
        assert!(set.contains(&2));
        assert!(!set.contains(&1));

        let collected: Vec<_> = set.iter().copied().collect();
        assert_eq!(collected, vec![0, 2, 5]);

        set.shift_remove(&2);
        let collected: Vec<_> = set.iter().copied().collect();
        assert_eq!(collected, vec![0, 5]);
    }
}
