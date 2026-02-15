//! Superposition inference rule for equality reasoning

use super::common::{
    collect_scoped_literals_except, is_ordered_greater, remove_duplicate_literals,
};
use crate::logic::{Clause, Interner, KBOConfig, Literal, Position as FolPosition, PredicateSymbol, Term, KBO};
use crate::logic::unification::scoped::{ScopedVar, flatten_scoped, unify_scoped};
use crate::state::{SaturationState, StateChange, GeneratingInference};
use crate::logic::clause_manager::ClauseManager;
use crate::logic::ordering::orient_equalities::orient_clause_equalities;
use crate::index::IndexRegistry;
use crate::selection::LiteralSelector;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering as AtomicOrdering;

/// Position in a term/atom where unification can occur
struct Position {
    term: Term,
    path: Vec<usize>, // Path to this position
}

/// Apply superposition rule using literal selection
///
/// Superposition: l = r v C1    L[l'] v C2  =>  (L[r] v C1 v C2)sigma
///   where sigma = mgu(l, l'), l*sigma not smaller than r*sigma, l' is not a variable.
///   If L[l'] is an equality s[l'] +/- t, additionally s[l']*sigma not smaller than t*sigma.
pub fn superposition(
    from_clause: &Clause,
    into_clause: &Clause,
    idx1: usize,
    idx2: usize,
    selector: &dyn LiteralSelector,
    interner: &mut Interner,
) -> Vec<StateChange> {
    let mut results = Vec::new();
    let kbo = KBO::new(KBOConfig::default());

    // Get selected literals from both clauses
    let selected_from = selector.select(from_clause);
    let selected_into = selector.select(into_clause);

    // If no literals are selected in either clause, no superposition is possible
    if selected_from.is_empty() || selected_into.is_empty() {
        return results;
    }

    // No variable renaming needed — scoped unification handles variable separation.
    // from_clause is at scope 0, into_clause is at scope 1.

    // Find positive equality literals in selected literals of from_clause
    for &from_idx in &selected_from {
        let from_lit = &from_clause.literals[from_idx];

        if from_lit.polarity && from_lit.is_equality(interner) {
            if let [ref left, ref right] = from_lit.args.as_slice() {
                // Try superposition in BOTH directions:
                // 1. Find occurrences of left, replace with right (left -> right)
                // 2. Find occurrences of right, replace with left (right -> left)
                // The ordering constraint is checked AFTER computing the MGU

                let directions: [(_, _, &str); 2] = [
                    (left, right, "l->r"),
                    (right, left, "r->l"),
                ];

                for (pattern, replacement, _dir) in directions {
                    // For each selected literal in into_clause
                    for &into_idx in &selected_into {
                        let into_lit = &into_clause.literals[into_idx];

                        // Find positions where pattern can be unified with some subterm
                        let positions = find_unifiable_positions(&into_lit.args, pattern, &kbo);

                        for pos in positions {
                            // CRITICAL: l' (pos.term) must not be a variable
                            // This prevents unsound inferences
                            if matches!(pos.term, Term::Variable(_)) {
                                continue;
                            }

                            // Unify pattern (scope 0) with the subterm (scope 1)
                            if let Ok(mgu) = unify_scoped(pattern, 0, &pos.term, 1) {
                                // Flatten to get concrete Terms for ordering checks
                                let mut renaming = HashMap::new();
                                let pattern_sigma = flatten_scoped(pattern, 0, &mgu, &mut renaming, interner);
                                let replacement_sigma = flatten_scoped(replacement, 0, &mgu, &mut renaming, interner);

                                // Check ordering constraint: pattern_sigma not smaller than replacement_sigma
                                if !is_ordered_greater(&pattern_sigma, &replacement_sigma, &kbo) {
                                    continue;
                                }

                                // Additional check for superposition into equalities
                                // The side containing l' must not be smaller than the other side
                                if into_lit.is_equality(interner) && !pos.path.is_empty() {
                                    let s = &into_lit.args[0];
                                    let t = &into_lit.args[1];

                                    let s_sigma = flatten_scoped(s, 1, &mgu, &mut renaming, interner);
                                    let t_sigma = flatten_scoped(t, 1, &mgu, &mut renaming, interner);

                                    if pos.path[0] == 0 {
                                        if !is_ordered_greater(&s_sigma, &t_sigma, &kbo) {
                                            continue;
                                        }
                                    } else if pos.path[0] == 1 {
                                        if !is_ordered_greater(&t_sigma, &s_sigma, &kbo) {
                                            continue;
                                        }
                                    }
                                }

                                // Collect side literals from from_clause (scope 0)
                                let mut new_literals = collect_scoped_literals_except(
                                    from_clause, &[from_idx], 0, &mgu, &mut renaming, interner,
                                );

                                // Add the modified literal from into_clause:
                                // flatten all args at scope 1, then replace at path
                                let into_lit_modified = replace_at_position_scoped(
                                    into_lit.predicate,
                                    &into_lit.args,
                                    &pos.path,
                                    &replacement_sigma,
                                    1,
                                    &mgu,
                                    &mut renaming,
                                    interner,
                                    into_lit.polarity,
                                );
                                new_literals.push(into_lit_modified);

                                // Add other literals from into_clause (scope 1)
                                new_literals.extend(collect_scoped_literals_except(
                                    into_clause, &[into_idx], 1, &mgu, &mut renaming, interner,
                                ));

                                // Remove duplicates
                                new_literals = remove_duplicate_literals(new_literals);

                                let mut new_clause = Clause::new(new_literals);
                                orient_clause_equalities(&mut new_clause, interner);

                                // Tautology check delegated to TautologyRule during forward simplification
                                results.push(StateChange::Add(
                                    Arc::new(new_clause),
                                    "Superposition".into(),
                                    vec![FolPosition::clause(idx1), FolPosition::clause(idx2)],
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    results
}

/// Find all positions in a literal's arguments where a term can potentially unify with pattern
/// This is used to find occurrences of l in the literal that can unify with l
///
/// For equalities, we search BOTH sides. The ordering constraint (s[l']*sigma not smaller than t*sigma)
/// is checked later after computing the MGU, not here during position search.
/// This is important because the ordering depends on the substitution, which
/// we don't know until we find a unifier.
fn find_unifiable_positions(args: &[Term], pattern: &Term, _kbo: &KBO) -> Vec<Position> {
    let mut positions = Vec::new();

    // Search all arguments for potential unification positions
    // For equalities, this searches both sides; the ordering constraint
    // is checked later in the superposition function after computing the MGU
    for (i, arg) in args.iter().enumerate() {
        find_positions_in_term(arg, pattern, vec![i], &mut positions);
    }

    positions
}

/// Find positions in a term recursively
fn find_positions_in_term(
    term: &Term,
    pattern: &Term,
    path: Vec<usize>,
    positions: &mut Vec<Position>,
) {
    // Check if current position can unify
    if could_unify(term, pattern) {
        positions.push(Position {
            term: term.clone(),
            path: path.clone(),
        });
    }

    // Recurse into subterms
    if let Term::Function(_, args) = term {
        for (i, arg) in args.iter().enumerate() {
            let mut new_path = path.clone();
            new_path.push(i);
            find_positions_in_term(arg, pattern, new_path, positions);
        }
    }
}

/// Quick check if two terms could potentially unify (without doing full unification)
fn could_unify(term1: &Term, term2: &Term) -> bool {
    match (term1, term2) {
        (Term::Variable(_), _) | (_, Term::Variable(_)) => true,
        (Term::Constant(c1), Term::Constant(c2)) => c1 == c2,
        (Term::Function(f1, args1), Term::Function(f2, args2)) => {
            f1.id == f2.id && args1.len() == args2.len()
        }
        _ => false,
    }
}

/// Replace a term at a specific position in a literal's arguments using scoped substitution.
///
/// Flattens all args at the given scope through the scoped substitution, then replaces
/// the subterm at `path` with `replacement_sigma` (already a concrete flattened Term).
fn replace_at_position_scoped(
    predicate: PredicateSymbol,
    args: &[Term],
    path: &[usize],
    replacement_sigma: &Term,
    scope: u8,
    subst: &crate::logic::unification::scoped::ScopedSubstitution,
    renaming: &mut HashMap<ScopedVar, crate::logic::VariableId>,
    interner: &mut Interner,
    polarity: bool,
) -> Literal {
    // Flatten all args through the scoped substitution
    let mut new_args: Vec<Term> = args
        .iter()
        .map(|a| flatten_scoped(a, scope, subst, renaming, interner))
        .collect();
    // Replace the subterm at path with the already-flattened replacement
    if !path.is_empty() {
        new_args[path[0]] = replace_in_term(&new_args[path[0]], &path[1..], replacement_sigma);
    }
    Literal { predicate, args: new_args, polarity }
}

/// Replace a term at a specific position in another term
fn replace_in_term(term: &Term, path: &[usize], replacement: &Term) -> Term {
    if path.is_empty() {
        replacement.clone()
    } else {
        match term {
            Term::Variable(_) | Term::Constant(_) => term.clone(),
            Term::Function(f, args) => {
                let mut new_args = args.clone();
                new_args[path[0]] = replace_in_term(&new_args[path[0]], &path[1..], replacement);
                Term::Function(*f, new_args)
            }
        }
    }
}

/// Superposition inference rule.
///
/// Generates superposition inferences between the given clause and processed clauses.
pub struct SuperpositionRule;

impl SuperpositionRule {
    pub fn new() -> Self {
        SuperpositionRule
    }
}

impl Default for SuperpositionRule {
    fn default() -> Self {
        Self::new()
    }
}

impl GeneratingInference for SuperpositionRule {
    fn name(&self) -> &str {
        "Superposition"
    }

    fn generate(
        &self,
        given_idx: usize,
        state: &SaturationState,
        cm: &mut ClauseManager,
        indices: &IndexRegistry,
    ) -> Vec<StateChange> {
        let given = &state.clauses[given_idx];
        let cancel = cm.cancel.clone();
        let start_time = cm.start_time;
        let timeout = cm.timeout;
        let memory_limit = cm.memory_limit;
        let baseline_rss = cm.baseline_rss_mb;
        let selector = cm.literal_selector.as_ref();
        let interner = &mut cm.interner;
        let mut changes = Vec::new();

        let stopped = || -> bool {
            if cancel.load(AtomicOrdering::Relaxed) { return true; }
            if let Some(start) = start_time {
                if start.elapsed() > timeout {
                    cancel.store(true, AtomicOrdering::Relaxed);
                    return true;
                }
            }
            if let Some(limit) = memory_limit {
                if let Some(rss) = crate::config::process_memory_mb() {
                    if rss.saturating_sub(baseline_rss) >= limit {
                        cancel.store(true, AtomicOrdering::Relaxed);
                        return true;
                    }
                }
            }
            false
        };

        if let Some(sli) = indices.selected_literals() {
            let eq_clauses = sli.equality_clauses();
            let given_has_eq = eq_clauses.contains(&given_idx);

            for &processed_idx in state.processed.iter() {
                if stopped() { break; }
                if processed_idx == given_idx {
                    continue;
                }
                if let Some(processed_clause) = state.clauses.get(processed_idx) {
                    if given_has_eq {
                        changes.extend(superposition(given, processed_clause, given_idx, processed_idx, selector, interner));
                    }
                    if eq_clauses.contains(&processed_idx) {
                        changes.extend(superposition(processed_clause, given, processed_idx, given_idx, selector, interner));
                    }
                }
            }

            if given_has_eq && !stopped() {
                changes.extend(superposition(given, given, given_idx, given_idx, selector, interner));
            }
        } else {
            for &processed_idx in state.processed.iter() {
                if stopped() { break; }
                if processed_idx == given_idx {
                    continue;
                }
                if let Some(processed_clause) = state.clauses.get(processed_idx) {
                    changes.extend(superposition(given, processed_clause, given_idx, processed_idx, selector, interner));
                    changes.extend(superposition(processed_clause, given, processed_idx, given_idx, selector, interner));
                }
            }
            if !stopped() {
                changes.extend(superposition(given, given, given_idx, given_idx, selector, interner));
            }
        }

        changes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::{Constant, FunctionSymbol, PredicateSymbol, Variable};
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

    /// Test superposition into the RIGHT side of an equality
    /// This verifies that we search both sides of equalities for superposition positions.
    /// The ordering constraint s[l']*sigma not smaller than t*sigma must be satisfied (the side containing l'
    /// must not be smaller than the other side).
    #[test]
    fn test_superposition_into_right_side_of_equality() {
        let mut ctx = TestContext::new();

        // From: f(X) = X (f(X) > X in KBO)
        // Into: a = f(b) (f(b) > a in KBO, so RIGHT side is larger)
        //
        // The pattern f(X) should unify with f(b) in the RIGHT side of the into clause.
        // s[l'] = f(b), t = a
        // Constraint: s[l']*sigma not smaller than t*sigma means f(b) not smaller than a, which is satisfied since f(b) > a
        // After superposition: a = b

        let eq = ctx.pred("=", 2);

        let a = ctx.const_("a");
        let b = ctx.const_("b");
        let x = ctx.var("X");
        let f_x = ctx.func("f", vec![x.clone()]);
        let f_b = ctx.func("f", vec![b.clone()]);

        // f(X) = X
        let clause1 = Clause::new(vec![Literal::positive(eq, vec![f_x.clone(), x.clone()])]);

        // a = f(b) (note: right side f(b) is larger than left side a)
        let clause2 = Clause::new(vec![Literal::positive(eq, vec![a.clone(), f_b.clone()])]);

        let selector = SelectAll;
        let results = superposition(&clause1, &clause2, 0, 1, &selector, &mut ctx.interner);

        // Should derive: a = b
        assert!(
            !results.is_empty(),
            "Superposition should find positions in the right side of equalities"
        );

        // Find the expected result: a = b
        let found = results.iter().any(|r| {
            if let StateChange::Add(clause, _, _) = r {
                clause.literals.len() == 1
                    && clause.literals[0].polarity
                    && clause.literals[0].predicate.name(&ctx.interner) == "="
                    && clause.literals[0].args.len() == 2
            } else {
                false
            }
        });

        assert!(
            found,
            "Expected to derive a = b, got: {:?}",
            results
                .iter()
                .filter_map(|r| if let StateChange::Add(clause, _, _) = r { Some(clause.display(&ctx.interner).to_string()) } else { None })
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_superposition_with_selection() {
        let mut ctx = TestContext::new();

        // Test superposition with corrected implementation
        // From: mult(e,X) = X
        // Into: P(mult(e,c))
        // Should derive: P(c)

        let eq = ctx.pred("=", 2);
        let p = ctx.pred("P", 1);

        let e = ctx.const_("e");
        let c = ctx.const_("c");
        let x = ctx.var("X");
        let mult_ex = ctx.func("mult", vec![e.clone(), x.clone()]);
        let mult_ec = ctx.func("mult", vec![e.clone(), c.clone()]);

        // mult(e,X) = X
        let clause1 = Clause::new(vec![Literal::positive(eq, vec![mult_ex.clone(), x.clone()])]);

        // P(mult(e,c))
        let clause2 = Clause::new(vec![Literal::positive(p, vec![mult_ec.clone()])]);

        let selector = SelectAll;
        let results = superposition(&clause1, &clause2, 0, 1, &selector, &mut ctx.interner);

        // Should derive P(c)
        assert_eq!(results.len(), 1);
        if let StateChange::Add(clause, rule, _) = &results[0] {
            assert_eq!(rule, "Superposition");
            assert_eq!(clause.literals.len(), 1);
            assert!(clause.literals[0].polarity);
            assert_eq!(
                clause.literals[0].predicate.name(&ctx.interner),
                "P"
            );
            assert_eq!(clause.literals[0].args.len(), 1);
            match &clause.literals[0].args[0] {
                Term::Constant(constant) => assert_eq!(constant.name(&ctx.interner), "c"),
                _ => panic!("Expected constant c"),
            }
        } else {
            panic!("Expected StateChange::Add");
        }
    }
}
