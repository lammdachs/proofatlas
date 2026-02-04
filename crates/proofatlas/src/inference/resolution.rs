//! Binary resolution inference rule

use super::common::{
    collect_literals_except, remove_duplicate_literals, rename_clause_variables, unify_atoms,
    InferenceResult,
};
use super::derivation::Derivation;
use crate::fol::{Clause, Interner};
use crate::selection::LiteralSelector;

/// Apply binary resolution between two clauses using literal selection
pub fn resolution(
    clause1: &Clause,
    clause2: &Clause,
    idx1: usize,
    idx2: usize,
    selector: &dyn LiteralSelector,
    interner: &mut Interner,
) -> Vec<InferenceResult> {
    let mut results = Vec::new();

    // Get selected literals from both clauses
    let selected1 = selector.select(clause1);
    let selected2 = selector.select(clause2);

    // If no literals are selected in either clause, no resolution is possible
    if selected1.is_empty() || selected2.is_empty() {
        return results;
    }

    // Rename variables in clause2 to avoid conflicts
    let renamed_clause2 = rename_clause_variables(clause2, &format!("c{}", idx2), interner);

    // Only try to resolve SELECTED literals
    for &i in &selected1 {
        let lit1 = &clause1.literals[i];

        for &j in &selected2 {
            let lit2 = &renamed_clause2.literals[j];

            // Check if literals have opposite polarity and same predicate
            if lit1.polarity != lit2.polarity && lit1.atom.predicate == lit2.atom.predicate {
                // Try to unify the atoms
                if let Ok(mgu) = unify_atoms(&lit1.atom, &lit2.atom) {
                    // Collect side literals from both clauses
                    let mut new_literals = collect_literals_except(clause1, &[i], &mgu);
                    new_literals.extend(collect_literals_except(&renamed_clause2, &[j], &mgu));

                    // Remove duplicates
                    new_literals = remove_duplicate_literals(new_literals);

                    let new_clause = Clause::new(new_literals);

                    // Tautology check delegated to TautologyRule during forward simplification
                    results.push(InferenceResult {
                        derivation: Derivation {
                            rule_name: "Resolution".into(),
                            premises: vec![idx1, idx2],
                        },
                        conclusion: new_clause,
                    });
                }
            }
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fol::{Atom, Constant, FunctionSymbol, Literal, PredicateSymbol, Term, Variable};
    use crate::selection::SelectAll;

    struct TestContext {
        interner: Interner,
    }

    impl TestContext {
        fn new() -> Self {
            TestContext {
                interner: Interner::new(),
            }
        }

        fn var(&mut self, name: &str) -> Term {
            let id = self.interner.intern_variable(name);
            Term::Variable(Variable::new(id))
        }

        fn const_(&mut self, name: &str) -> Term {
            let id = self.interner.intern_constant(name);
            Term::Constant(Constant::new(id))
        }

        fn func(&mut self, name: &str, args: Vec<Term>) -> Term {
            let id = self.interner.intern_function(name);
            Term::Function(FunctionSymbol::new(id, args.len() as u8), args)
        }

        fn pred(&mut self, name: &str, arity: u8) -> PredicateSymbol {
            let id = self.interner.intern_predicate(name);
            PredicateSymbol::new(id, arity)
        }
    }

    #[test]
    fn test_resolution_with_select_all() {
        let mut ctx = TestContext::new();

        // P(a) ∨ Q(X)
        // ~P(a) ∨ R(b)
        // Should resolve to Q(X) ∨ R(b)

        let p = ctx.pred("P", 1);
        let q = ctx.pred("Q", 1);
        let r = ctx.pred("R", 1);

        let a = ctx.const_("a");
        let b = ctx.const_("b");
        let x = ctx.var("X");

        let clause1 = Clause::new(vec![
            Literal::positive(Atom {
                predicate: p,
                args: vec![a.clone()],
            }),
            Literal::positive(Atom {
                predicate: q,
                args: vec![x.clone()],
            }),
        ]);

        let clause2 = Clause::new(vec![
            Literal::negative(Atom {
                predicate: p,
                args: vec![a.clone()],
            }),
            Literal::positive(Atom {
                predicate: r,
                args: vec![b.clone()],
            }),
        ]);

        let selector = SelectAll;
        let results = resolution(&clause1, &clause2, 0, 1, &selector, &mut ctx.interner);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].conclusion.literals.len(), 2);
    }
}
