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

use crate::logic::{Clause, Literal, Term, VariableId, Interner};
use crate::logic::clause_manager::ClauseManager;
use crate::index::IndexRegistry;
use crate::logic::Position;
use crate::state::{SaturationState, SimplifyingInference, StateChange};
use std::collections::HashMap;

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

    fn verify(
        &self,
        clause_idx: usize,
        replacement: Option<&Clause>,
        premises: &[Position],
        state: &SaturationState,
        _cm: &ClauseManager,
    ) -> Result<(), crate::state::VerificationError> {
        use crate::state::VerificationError;

        // Subsumption deletion: no replacement, and the subsuming clause must subsume the deleted clause
        if replacement.is_some() {
            return Err(VerificationError::InvalidConclusion {
                step_idx: 0,
                rule: "Subsumption".into(),
                reason: "subsumption should not produce a replacement clause".into(),
            });
        }

        if premises.len() != 1 {
            return Err(VerificationError::InvalidConclusion {
                step_idx: 0,
                rule: "Subsumption".into(),
                reason: format!("expected 1 premise (subsumer), got {}", premises.len()),
            });
        }

        let subsumed = &state.clauses[clause_idx];
        let subsumer = &state.clauses[premises[0].clause];

        // Verify using the full subsumption check
        if subsumes(subsumer, subsumed) || subsumes_greedy(subsumer, subsumed) {
            Ok(())
        } else {
            Err(VerificationError::InvalidConclusion {
                step_idx: 0,
                rule: "Subsumption".into(),
                reason: "subsumer does not subsume the deleted clause".into(),
            })
        }
    }

    fn simplify_forward(
        &self,
        clause_idx: usize,
        state: &SaturationState,
        _cm: &ClauseManager,
        indices: &mut IndexRegistry,
    ) -> Option<StateChange> {
        let clause = &state.clauses[clause_idx];
        if let Some(checker) = indices.subsumption_checker_mut() {
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
        indices: &mut IndexRegistry,
    ) -> Vec<StateChange> {
        let checker = match indices.subsumption_checker_mut() {
            Some(c) => c,
            None => return vec![],
        };

        // Find clauses subsumed by this clause
        // (find_subsumed_by uses the literal tree's active set, not candidate_indices)
        let subsumed = checker.find_subsumed_by(clause_idx, &[]);

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

/// Check if a unit clause subsumes another clause.
pub fn subsumes_unit(unit: &Clause, clause: &Clause) -> bool {
    if unit.literals.len() != 1 {
        return false;
    }

    let unit_lit = &unit.literals[0];
    let mut subst = MatchSubst::new(max_var_id(unit));

    for lit in &clause.literals {
        if lit.polarity == unit_lit.polarity {
            let mark = subst.mark();
            if match_literals(unit_lit, lit, &mut subst) {
                return true;
            }
            subst.backtrack(mark);
        }
    }

    false
}

/// Full subsumption check with backtracking.
pub fn subsumes(subsumer: &Clause, subsumee: &Clause) -> bool {
    if subsumer.literals.len() > subsumee.literals.len() {
        return false;
    }

    let mut subst = MatchSubst::new(max_var_id(subsumer));
    let mut used = vec![false; subsumee.literals.len()];

    find_subsumption_mapping(subsumer, subsumee, 0, &mut subst, &mut used)
}

/// Greedy subsumption for larger clauses.
pub fn subsumes_greedy(subsumer: &Clause, subsumee: &Clause) -> bool {
    if subsumer.literals.len() > subsumee.literals.len() {
        return false;
    }

    let mut subst = MatchSubst::new(max_var_id(subsumer));
    let mut used = vec![false; subsumee.literals.len()];

    for subsumer_lit in &subsumer.literals {
        let mut found = false;

        for (i, subsumee_lit) in subsumee.literals.iter().enumerate() {
            if used[i] || subsumee_lit.polarity != subsumer_lit.polarity {
                continue;
            }

            let mark = subst.mark();
            if match_literals(subsumer_lit, subsumee_lit, &mut subst) {
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
pub fn compatible_structure(clause1: &Clause, clause2: &Clause) -> bool {
    let preds1: std::collections::HashSet<_> = clause1.literals.iter().map(|l| &l.predicate).collect();
    let preds2: std::collections::HashSet<_> = clause2.literals.iter().map(|l| &l.predicate).collect();
    preds1.is_subset(&preds2)
}

// =============================================================================
// Flat-Array Matching Substitution
// =============================================================================

/// Lightweight matching substitution using a flat array indexed by VariableId.
///
/// Stores term references (no cloning) and supports O(1) bind/get/backtrack.
/// Much faster than HashMap-based Substitution for the small variable counts
/// typical in subsumption matching.
struct MatchSubst<'a> {
    /// Bindings indexed by VariableId.as_u32(). None = unbound.
    bindings: Vec<Option<&'a Term>>,
    /// Trail of bound variable indices for backtracking.
    trail: Vec<u32>,
}

impl<'a> MatchSubst<'a> {
    /// Create a new substitution sized for the given max variable ID.
    fn new(max_var_id: u32) -> Self {
        MatchSubst {
            bindings: vec![None; max_var_id as usize + 1],
            trail: Vec::new(),
        }
    }

    #[inline(always)]
    fn get(&self, var_id: VariableId) -> Option<&'a Term> {
        let idx = var_id.as_u32() as usize;
        if idx < self.bindings.len() {
            self.bindings[idx]
        } else {
            None
        }
    }

    #[inline(always)]
    fn bind(&mut self, var_id: VariableId, term: &'a Term) {
        let idx = var_id.as_u32();
        self.bindings[idx as usize] = Some(term);
        self.trail.push(idx);
    }

    #[inline(always)]
    fn mark(&self) -> usize {
        self.trail.len()
    }

    #[inline(always)]
    fn backtrack(&mut self, mark: usize) {
        while self.trail.len() > mark {
            let idx = self.trail.pop().unwrap();
            self.bindings[idx as usize] = None;
        }
    }
}

/// Find the maximum variable ID in a clause (for sizing MatchSubst).
/// Returns 0 if no variables (ground clause).
fn max_var_id(clause: &Clause) -> u32 {
    let mut max_id: u32 = 0;
    for lit in &clause.literals {
        for term in &lit.args {
            max_var_id_term(term, &mut max_id);
        }
    }
    max_id
}

fn max_var_id_term(term: &Term, max_id: &mut u32) {
    match term {
        Term::Variable(v) => {
            let id = v.id.as_u32();
            if id > *max_id {
                *max_id = id;
            }
        }
        Term::Constant(_) => {}
        Term::Function(_, args) => {
            for arg in args {
                max_var_id_term(arg, max_id);
            }
        }
    }
}

// =============================================================================
// Term Matching
// =============================================================================

/// Check if two terms are structurally equal.
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

/// Try to match term1 (pattern) against term2 (target) with flat-array substitution.
/// Only variables in term1 can be bound.
#[inline]
fn match_terms<'a>(term1: &Term, term2: &'a Term, subst: &mut MatchSubst<'a>) -> bool {
    match term1 {
        Term::Variable(v) => {
            if let Some(bound) = subst.get(v.id) {
                terms_equal(bound, term2)
            } else {
                subst.bind(v.id, term2);
                true
            }
        }
        Term::Constant(c1) => matches!(term2, Term::Constant(c2) if c1 == c2),
        Term::Function(f1, args1) => match term2 {
            Term::Function(f2, args2) => {
                f1 == f2
                    && args1.len() == args2.len()
                    && args1
                        .iter()
                        .zip(args2)
                        .all(|(a1, a2)| match_terms(a1, a2, subst))
            }
            _ => false,
        },
    }
}

/// Try to match two literals (pattern lit1 against target lit2).
#[inline]
fn match_literals<'a>(lit1: &Literal, lit2: &'a Literal, subst: &mut MatchSubst<'a>) -> bool {
    if lit1.polarity != lit2.polarity || lit1.predicate != lit2.predicate || lit1.args.len() != lit2.args.len() {
        return false;
    }
    lit1.args.iter().zip(&lit2.args).all(|(t1, t2)| match_terms(t1, t2, subst))
}

/// Recursive backtracking search for a subsumption mapping.
fn find_subsumption_mapping<'a>(
    subsumer: &Clause,
    subsumee: &'a Clause,
    lit_idx: usize,
    subst: &mut MatchSubst<'a>,
    used: &mut [bool],
) -> bool {
    if lit_idx >= subsumer.literals.len() {
        return true;
    }

    let subsumer_lit = &subsumer.literals[lit_idx];

    for (i, subsumee_lit) in subsumee.literals.iter().enumerate() {
        if used[i] || subsumee_lit.polarity != subsumer_lit.polarity {
            continue;
        }

        let mark = subst.mark();
        if match_literals(subsumer_lit, subsumee_lit, subst) {
            used[i] = true;
            if find_subsumption_mapping(subsumer, subsumee, lit_idx + 1, subst, used) {
                return true;
            }
            used[i] = false;
        }
        subst.backtrack(mark);
    }

    false
}
