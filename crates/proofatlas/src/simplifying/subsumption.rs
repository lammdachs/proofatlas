//! Subsumption checking for redundancy elimination in theorem proving
//!
//! This module implements a pragmatic approach to subsumption that balances
//! completeness with performance. Subsumption is a key redundancy elimination
//! technique where a clause C subsumes clause D if there exists a substitution σ
//! such that Cσ ⊆ D (every literal in Cσ appears in D).
//!
//! ## Implementation Strategy
//!
//! Our implementation uses a tiered approach:
//!
//! 1. **Exact Duplicate Detection** (100% complete, O(1))
//!    - Uses string representation hashing for instant detection
//!    - Catches identical clauses regardless of literal order
//!
//! 2. **Variant Detection** (100% complete for variants)
//!    - Detects clauses that are identical up to variable renaming
//!    - Example: P(X,Y) subsumes P(A,B) as a variant
//!
//! 3. **Unit Subsumption** (100% complete for unit clauses)
//!    - Special handling for single-literal clauses
//!    - Very effective in practice as many derived clauses are units
//!
//! 4. **Complete Subsumption for Small Clauses** (≤3 literals)
//!    - Full subsumption checking with proper backtracking
//!    - Feasible for small clauses where the search space is limited
//!
//! 5. **Greedy Heuristic for Large Clauses** (>3 literals)
//!    - Uses a greedy matching strategy that may miss some subsumptions
//!    - Trades completeness for performance on larger clauses
//!
//! ## Design Rationale
//!
//! This design is based on empirical observations:
//! - Most redundant clauses are exact duplicates or variants
//! - Unit clauses are common and unit subsumption is very effective
//! - Full subsumption checking becomes expensive for larger clauses
//! - A greedy heuristic catches many subsumptions with reasonable cost

use crate::logic::{Clause, Literal, Substitution, Term, VariableId, Interner};
use crate::logic::clause_manager::ClauseManager;
use crate::index::IndexRegistry;
use crate::logic::Position;
use crate::state::{SaturationState, SimplifyingInference, StateChange};
use std::collections::{HashMap, HashSet};

// =============================================================================
// SubsumptionRule (stateless rule adapter)
// =============================================================================

/// Forward/backward subsumption rule.
///
/// Deletes clauses that are subsumed by existing clauses in U∪P.
/// Queries the SubsumptionChecker from the IndexRegistry.
pub struct SubsumptionRule;

impl SubsumptionRule {
    pub fn new(_interner: &Interner) -> Self {
        SubsumptionRule
    }
}

impl Default for SubsumptionRule {
    fn default() -> Self {
        SubsumptionRule
    }
}

impl SimplifyingInference for SubsumptionRule {
    fn name(&self) -> &str {
        "Subsumption"
    }

    fn simplify_forward(
        &self,
        clause_idx: usize,
        state: &SaturationState,
        _cm: &ClauseManager,
        indices: &IndexRegistry,
    ) -> Option<StateChange> {
        let clause = &state.clauses[clause_idx];
        if let Some(checker) = indices.subsumption_checker() {
            if let Some(subsumer_idx) = checker.find_subsumer(clause) {
                return Some(StateChange::Simplify(clause_idx, None, self.name().into(), vec![Position::clause(subsumer_idx)]));
            }
        }
        None
    }

    fn simplify_backward(
        &self,
        clause_idx: usize,
        state: &SaturationState,
        _cm: &ClauseManager,
        indices: &IndexRegistry,
    ) -> Vec<StateChange> {
        let checker = match indices.subsumption_checker() {
            Some(c) => c,
            None => return vec![],
        };

        // Collect all candidate indices from U∪P
        let all_indices: Vec<usize> = state.unprocessed
            .iter()
            .chain(state.processed.iter())
            .copied()
            .collect();

        // Find clauses subsumed by this clause
        let subsumed = checker.find_subsumed_by(clause_idx, &all_indices);

        let rule_name: String = self.name().into();
        subsumed
            .into_iter()
            .filter(|&idx| state.processed.contains(&idx) || state.unprocessed.contains(&idx))
            .map(|idx| StateChange::Simplify(idx, None, rule_name.clone(), vec![Position::clause(clause_idx)]))
            .collect()
    }
}

// =============================================================================
// Pure Subsumption Functions
// =============================================================================

/// Check if two clauses are variants (identical up to variable renaming)
pub fn are_variants(clause1: &Clause, clause2: &Clause) -> bool {
    if clause1.literals.len() != clause2.literals.len() {
        return false;
    }

    // Try to find a variable mapping
    let mut var_map: HashMap<VariableId, VariableId> = HashMap::new();

    for (lit1, lit2) in clause1.literals.iter().zip(&clause2.literals) {
        if lit1.polarity != lit2.polarity {
            return false;
        }

        if !literals_match_with_mapping(lit1, lit2, &mut var_map) {
            return false;
        }
    }

    true
}

/// Check if two literals match with a variable mapping (comparing predicate and args directly)
fn literals_match_with_mapping(
    lit1: &Literal,
    lit2: &Literal,
    var_map: &mut HashMap<VariableId, VariableId>,
) -> bool {
    if lit1.predicate != lit2.predicate {
        return false;
    }

    if lit1.args.len() != lit2.args.len() {
        return false;
    }

    for (term1, term2) in lit1.args.iter().zip(&lit2.args) {
        if !terms_match_with_mapping(term1, term2, var_map) {
            return false;
        }
    }

    true
}

/// Check if terms match with a variable mapping
fn terms_match_with_mapping(
    term1: &Term,
    term2: &Term,
    var_map: &mut HashMap<VariableId, VariableId>,
) -> bool {
    match (term1, term2) {
        (Term::Variable(v1), Term::Variable(v2)) => match var_map.get(&v1.id) {
            Some(&mapped) => mapped == v2.id,
            None => {
                var_map.insert(v1.id, v2.id);
                true
            }
        },
        (Term::Constant(c1), Term::Constant(c2)) => c1 == c2,
        (Term::Function(f1, args1), Term::Function(f2, args2)) => {
            f1 == f2
                && args1.len() == args2.len()
                && args1
                    .iter()
                    .zip(args2)
                    .all(|(a1, a2)| terms_match_with_mapping(a1, a2, var_map))
        }
        _ => false,
    }
}

/// Check if a unit clause subsumes another clause using trail-based matching
pub fn subsumes_unit(unit: &Clause, clause: &Clause) -> bool {
    if unit.literals.len() != 1 {
        return false;
    }

    let unit_lit = &unit.literals[0];
    let var_count = count_variables(unit);
    let mut subst = Substitution::with_capacity(var_count);

    // Try to match the unit literal with each literal in the clause
    for lit in &clause.literals {
        if lit.polarity == unit_lit.polarity {
            let mark = subst.mark();
            if match_literals_trail(unit_lit, lit, &mut subst) {
                return true;
            }
            subst.backtrack(mark);
        }
    }

    false
}

/// Full subsumption check using trail-based backtracking
pub fn subsumes(subsumer: &Clause, subsumee: &Clause) -> bool {
    if subsumer.literals.len() > subsumee.literals.len() {
        return false;
    }

    // Use trail-based substitution for efficient backtracking
    let var_count = count_variables(subsumer);
    let mut subst = Substitution::with_capacity(var_count);

    find_subsumption_mapping_trail(
        subsumer,
        subsumee,
        0,
        &mut subst,
        &mut vec![false; subsumee.literals.len()],
    )
}

/// Greedy subsumption for larger clauses using trail-based backtracking
pub fn subsumes_greedy(subsumer: &Clause, subsumee: &Clause) -> bool {
    if subsumer.literals.len() > subsumee.literals.len() {
        return false;
    }

    let var_count = count_variables(subsumer);
    let mut subst = Substitution::with_capacity(var_count);
    let mut used = vec![false; subsumee.literals.len()];

    // Greedy matching: for each literal in subsumer, find the first compatible match
    for subsumer_lit in &subsumer.literals {
        let mut found = false;

        for (i, subsumee_lit) in subsumee.literals.iter().enumerate() {
            if used[i] || subsumee_lit.polarity != subsumer_lit.polarity {
                continue;
            }

            let mark = subst.mark();
            if match_literals_trail(subsumer_lit, subsumee_lit, &mut subst) {
                // For greedy, we commit to this match immediately
                used[i] = true;
                found = true;
                break;
            }
            subst.backtrack(mark);
        }

        if !found {
            return false;
        }
    }

    true
}

/// Check if two clauses have compatible structure for subsumption.
/// Returns true if subsumer's predicates are a subset of subsumee's predicates.
pub fn compatible_structure(clause1: &Clause, clause2: &Clause) -> bool {
    // Check predicate symbols
    let preds1: HashSet<_> = clause1.literals.iter().map(|l| &l.predicate).collect();
    let preds2: HashSet<_> = clause2.literals.iter().map(|l| &l.predicate).collect();

    // subsumer's predicates must be subset of subsumee's
    preds1.is_subset(&preds2)
}

// =============================================================================
// Trail-Based Subsumption
// =============================================================================

/// Count the number of unique variables in a clause (for capacity hints)
fn count_variables(clause: &Clause) -> usize {
    let mut vars = HashSet::new();
    for lit in &clause.literals {
        for term in &lit.args {
            term.collect_variable_ids(&mut vars);
        }
    }
    vars.len()
}

/// Check if two terms are equal
fn terms_equal(term1: &Term, term2: &Term) -> bool {
    match (term1, term2) {
        (Term::Variable(v1), Term::Variable(v2)) => v1 == v2,
        (Term::Constant(c1), Term::Constant(c2)) => c1 == c2,
        (Term::Function(f1, args1), Term::Function(f2, args2)) => {
            f1 == f2
                && args1.len() == args2.len()
                && args1.iter().zip(args2).all(|(a1, a2)| terms_equal(a1, a2))
        }
        _ => false,
    }
}

/// Try to match two terms with a trailed substitution
fn match_terms_trail(term1: &Term, term2: &Term, subst: &mut Substitution) -> bool {
    match term1 {
        Term::Variable(v) => {
            if let Some(bound_term) = subst.get(v.id) {
                terms_equal(bound_term, term2)
            } else {
                subst.bind(*v, term2.clone());
                true
            }
        }
        Term::Constant(c1) => match term2 {
            Term::Constant(c2) => c1 == c2,
            _ => false,
        },
        Term::Function(f1, args1) => match term2 {
            Term::Function(f2, args2) => {
                f1 == f2
                    && args1.len() == args2.len()
                    && args1
                        .iter()
                        .zip(args2)
                        .all(|(a1, a2)| match_terms_trail(a1, a2, subst))
            }
            _ => false,
        },
    }
}

/// Try to match two literals with a trailed substitution
fn match_literals_trail(lit1: &Literal, lit2: &Literal, subst: &mut Substitution) -> bool {
    if lit1.polarity != lit2.polarity {
        return false;
    }

    if lit1.predicate != lit2.predicate {
        return false;
    }

    if lit1.args.len() != lit2.args.len() {
        return false;
    }

    for (term1, term2) in lit1.args.iter().zip(&lit2.args) {
        if !match_terms_trail(term1, term2, subst) {
            return false;
        }
    }

    true
}

/// Recursive function to find subsumption mapping using trail-based backtracking
fn find_subsumption_mapping_trail(
    subsumer: &Clause,
    subsumee: &Clause,
    subsumer_idx: usize,
    subst: &mut Substitution,
    used: &mut Vec<bool>,
) -> bool {
    if subsumer_idx >= subsumer.literals.len() {
        return true; // All literals matched
    }

    let subsumer_lit = &subsumer.literals[subsumer_idx];

    // Try to match with each unused literal in subsumee
    for (i, subsumee_lit) in subsumee.literals.iter().enumerate() {
        if used[i] || subsumee_lit.polarity != subsumer_lit.polarity {
            continue;
        }

        let mark = subst.mark(); // O(1) instead of clone
        if match_literals_trail(subsumer_lit, subsumee_lit, subst) {
            used[i] = true;
            if find_subsumption_mapping_trail(subsumer, subsumee, subsumer_idx + 1, subst, used) {
                return true;
            }
            used[i] = false;
        }
        subst.backtrack(mark); // O(k) undo instead of discard clone
    }

    false
}
