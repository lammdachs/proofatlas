//! Factoring inference rule

use super::common::{collect_literals_except, remove_duplicate_literals, unify_atoms, InferenceResult};
use crate::logic::{Clause, Position};
use crate::state::{Derivation, StateChange, GeneratingInference};
use crate::selection::LiteralSelector;
use indexmap::IndexSet;
use crate::logic::Interner;

/// Apply factoring to a clause using literal selection
pub fn factoring(
    clause: &Clause,
    idx: usize,
    selector: &dyn LiteralSelector,
) -> Vec<InferenceResult> {
    let mut results = Vec::new();

    // Get selected literals
    let selected = selector.select(clause);

    // If no literals are selected, no factoring is possible
    if selected.is_empty() {
        return results;
    }

    // Only try to factor SELECTED literals
    for &i in &selected {
        let lit1 = &clause.literals[i];

        // Try to factor with other literals (both selected and non-selected)
        // According to the calculus, we factor the selected literal with any other literal
        for j in 0..clause.literals.len() {
            if i != j {
                let lit2 = &clause.literals[j];

                // Must have same polarity and predicate
                if lit1.polarity == lit2.polarity && lit1.predicate == lit2.predicate {
                    if let Ok(mgu) = unify_atoms(lit1.predicate, &lit1.args, lit2.predicate, &lit2.args) {
                        // Collect all literals except the factored one (j)
                        let new_literals = remove_duplicate_literals(
                            collect_literals_except(clause, &[j], &mgu)
                        );

                        let new_clause = Clause::new(new_literals);

                        // Tautology check delegated to TautologyRule during forward simplification
                        results.push(InferenceResult {
                            derivation: Derivation {
                                rule_name: "Factoring".into(),
                                premises: vec![Position::clause(idx)],
                            },
                            conclusion: new_clause,
                        });
                    }
                }
            }
        }
    }

    results
}

/// Factoring inference rule.
///
/// Generates factors of the given clause.
pub struct FactoringRule;

impl FactoringRule {
    pub fn new() -> Self {
        FactoringRule
    }
}

impl Default for FactoringRule {
    fn default() -> Self {
        Self::new()
    }
}

impl GeneratingInference for FactoringRule {
    fn name(&self) -> &str {
        "Factoring"
    }

    fn generate(
        &self,
        given_idx: usize,
        given: &Clause,
        _clauses: &[Clause],
        _processed: &IndexSet<usize>,
        selector: &dyn LiteralSelector,
        _interner: &mut Interner,
    ) -> Vec<StateChange> {
        factoring(given, given_idx, selector)
            .into_iter()
            .map(|result| StateChange::Add {
                clause: result.conclusion,
                derivation: result.derivation,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::{Interner, Literal, PredicateSymbol, Term, Variable};
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

        fn pred(&mut self, name: &str, arity: u8) -> PredicateSymbol {
            let id = self.interner.intern_predicate(name);
            PredicateSymbol::new(id, arity)
        }
    }

    #[test]
    fn test_factoring_with_select_all() {
        let mut ctx = TestContext::new();

        // P(X) v P(Y) v Q(Z)
        let p = ctx.pred("P", 1);
        let q = ctx.pred("Q", 1);

        let x = ctx.var("X");
        let y = ctx.var("Y");
        let z = ctx.var("Z");

        let clause = Clause::new(vec![
            Literal::positive(p, vec![x.clone()]),
            Literal::positive(p, vec![y.clone()]),
            Literal::positive(q, vec![z.clone()]),
        ]);

        let selector = SelectAll;
        let results = factoring(&clause, 0, &selector);

        // With SelectAll, both P literals are selected
        // P(X) factors with P(Y), and P(Y) factors with P(X)
        // Both produce the same clause: P(X) v Q(Z) (with appropriate substitution)
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].conclusion.literals.len(), 2);
        assert_eq!(results[1].conclusion.literals.len(), 2);
    }
}
