//! Binary resolution inference rule

use super::common::{
    collect_literals_except, remove_duplicate_literals, rename_clause_variables, unify_atoms,
    InferenceResult,
};
use crate::state::{Derivation, StateChange, GeneratingInference};
use crate::logic::{Clause, Interner, Position};
use crate::selection::LiteralSelector;
use indexmap::IndexSet;

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
            if lit1.polarity != lit2.polarity && lit1.predicate == lit2.predicate {
                // Try to unify the atoms
                if let Ok(mgu) = unify_atoms(lit1.predicate, &lit1.args, lit2.predicate, &lit2.args) {
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
                            premises: vec![Position::clause(idx1), Position::clause(idx2)],
                        },
                        conclusion: new_clause,
                    });
                }
            }
        }
    }

    results
}

/// Resolution inference rule.
///
/// Generates resolvents between the given clause and processed clauses.
pub struct ResolutionRule;

impl ResolutionRule {
    pub fn new() -> Self {
        ResolutionRule
    }
}

impl Default for ResolutionRule {
    fn default() -> Self {
        Self::new()
    }
}

impl GeneratingInference for ResolutionRule {
    fn name(&self) -> &str {
        "Resolution"
    }

    fn generate(
        &self,
        given_idx: usize,
        given: &Clause,
        clauses: &[Clause],
        processed: &IndexSet<usize>,
        selector: &dyn LiteralSelector,
        interner: &mut Interner,
    ) -> Vec<StateChange> {
        let mut changes = Vec::new();

        // Resolution with processed clauses
        for &processed_idx in processed.iter() {
            if processed_idx == given_idx {
                continue;
            }
            if let Some(processed_clause) = clauses.get(processed_idx) {
                // Given as first clause
                for result in resolution(given, processed_clause, given_idx, processed_idx, selector, interner) {
                    changes.push(StateChange::Add {
                        clause: result.conclusion,
                        derivation: result.derivation,
                    });
                }
                // Given as second clause
                for result in resolution(processed_clause, given, processed_idx, given_idx, selector, interner) {
                    changes.push(StateChange::Add {
                        clause: result.conclusion,
                        derivation: result.derivation,
                    });
                }
            }
        }

        // Self-resolution
        for result in resolution(given, given, given_idx, given_idx, selector, interner) {
            changes.push(StateChange::Add {
                clause: result.conclusion,
                derivation: result.derivation,
            });
        }

        changes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::{Constant, FunctionSymbol, Literal, PredicateSymbol, Term, Variable};
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

        // P(a) v Q(X)
        // ~P(a) v R(b)
        // Should resolve to Q(X) v R(b)

        let p = ctx.pred("P", 1);
        let q = ctx.pred("Q", 1);
        let r = ctx.pred("R", 1);

        let a = ctx.const_("a");
        let b = ctx.const_("b");
        let x = ctx.var("X");

        let clause1 = Clause::new(vec![
            Literal::positive(p, vec![a.clone()]),
            Literal::positive(q, vec![x.clone()]),
        ]);

        let clause2 = Clause::new(vec![
            Literal::negative(p, vec![a.clone()]),
            Literal::positive(r, vec![b.clone()]),
        ]);

        let selector = SelectAll;
        let results = resolution(&clause1, &clause2, 0, 1, &selector, &mut ctx.interner);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].conclusion.literals.len(), 2);
    }
}
