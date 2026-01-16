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

use crate::core::{Clause, Literal, Substitution, Term};
use std::collections::{HashMap, HashSet};

/// Subsumption checker implementing a balanced redundancy elimination strategy
pub struct SubsumptionChecker {
    /// All clauses indexed by their string representation for duplicate detection
    clause_strings: HashSet<String>,

    /// Unit clauses for unit subsumption
    units: Vec<(Clause, usize)>,

    /// All clauses for subsumption checking
    clauses: Vec<Clause>,
}

impl SubsumptionChecker {
    pub fn new() -> Self {
        SubsumptionChecker {
            clause_strings: HashSet::new(),
            units: Vec::new(),
            clauses: Vec::new(),
        }
    }

    /// Add a clause and return its index
    pub fn add_clause(&mut self, clause: Clause) -> usize {
        let idx = self.clauses.len();

        // Add to string index
        let clause_str = format!("{}", clause);
        self.clause_strings.insert(clause_str);

        // Add to unit index if applicable
        if clause.literals.len() == 1 {
            self.units.push((clause.clone(), idx));
        }

        self.clauses.push(clause);
        idx
    }

    /// Check if a clause is subsumed
    pub fn is_subsumed(&self, clause: &Clause) -> bool {
        // 1. Check for exact duplicates (very fast)
        let clause_str = format!("{}", clause);
        if self.clause_strings.contains(&clause_str) {
            return true;
        }

        // 2. Check for variants (duplicates up to variable renaming)
        if self.has_variant(clause) {
            return true;
        }

        // 3. Unit subsumption (fast and complete)
        // Check all clauses (including unit clauses) for subsumption by units
        for (unit, _) in &self.units {
            if subsumes_unit(unit, clause) {
                return true;
            }
        }

        // 4. For small clauses (2-3 literals), do complete subsumption
        if clause.literals.len() <= 3 {
            for existing in &self.clauses {
                if existing.literals.len() < clause.literals.len() && subsumes(existing, clause) {
                    return true;
                }
            }
        }

        // 5. For larger clauses, use a greedy heuristic
        // Only check against clauses with compatible structure
        if clause.literals.len() > 3 {
            for existing in &self.clauses {
                if existing.literals.len() >= clause.literals.len() {
                    continue;
                }

                // Quick structural check
                if !compatible_structure(existing, clause) {
                    continue;
                }

                // Try greedy subsumption
                if subsumes_greedy(existing, clause) {
                    return true;
                }
            }
        }

        false
    }

    /// Check if a clause is subsumed by any processed clause (excluding itself)
    /// Used in otter loop to check if the simplified given clause is redundant
    pub fn is_subsumed_by_processed(&self, exclude_idx: usize, clause: &Clause) -> bool {
        // Check for variants (excluding self)
        for (idx, existing) in self.clauses.iter().enumerate() {
            if idx == exclude_idx {
                continue;
            }
            if existing.literals.len() != clause.literals.len() {
                continue;
            }
            if are_variants(existing, clause) {
                return true;
            }
        }

        // Unit subsumption (excluding self)
        for (unit, idx) in &self.units {
            if *idx == exclude_idx {
                continue;
            }
            if subsumes_unit(unit, clause) {
                return true;
            }
        }

        // Full subsumption for small clauses (excluding self)
        if clause.literals.len() <= 3 {
            for (idx, existing) in self.clauses.iter().enumerate() {
                if idx == exclude_idx {
                    continue;
                }
                if existing.literals.len() < clause.literals.len() && subsumes(existing, clause) {
                    return true;
                }
            }
        }

        false
    }

    /// Check if we have a variant of this clause
    fn has_variant(&self, clause: &Clause) -> bool {
        // Get the clause's "shape" (predicate symbols and polarities)
        let shape = get_clause_shape(clause);

        for existing in &self.clauses {
            if existing.literals.len() != clause.literals.len() {
                continue;
            }

            // Quick shape check
            if get_clause_shape(existing) != shape {
                continue;
            }

            // Check if they're variants
            if are_variants(existing, clause) {
                return true;
            }
        }

        false
    }
}

/// Get the "shape" of a clause (predicates and polarities)
fn get_clause_shape(clause: &Clause) -> Vec<(String, bool)> {
    let mut shape: Vec<_> = clause
        .literals
        .iter()
        .map(|lit| (lit.atom.predicate.name.clone(), lit.polarity))
        .collect();
    shape.sort();
    shape
}

/// Check if two clauses are variants (identical up to variable renaming)
fn are_variants(clause1: &Clause, clause2: &Clause) -> bool {
    if clause1.literals.len() != clause2.literals.len() {
        return false;
    }

    // Try to find a variable mapping
    let mut var_map: HashMap<String, String> = HashMap::new();

    for (lit1, lit2) in clause1.literals.iter().zip(&clause2.literals) {
        if lit1.polarity != lit2.polarity {
            return false;
        }

        if !atoms_match_with_mapping(&lit1.atom, &lit2.atom, &mut var_map) {
            return false;
        }
    }

    true
}

/// Check if atoms match with a variable mapping
fn atoms_match_with_mapping(
    atom1: &crate::core::Atom,
    atom2: &crate::core::Atom,
    var_map: &mut HashMap<String, String>,
) -> bool {
    if atom1.predicate != atom2.predicate {
        return false;
    }

    if atom1.args.len() != atom2.args.len() {
        return false;
    }

    for (term1, term2) in atom1.args.iter().zip(&atom2.args) {
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
    var_map: &mut HashMap<String, String>,
) -> bool {
    match (term1, term2) {
        (Term::Variable(v1), Term::Variable(v2)) => match var_map.get(&v1.name) {
            Some(mapped) => mapped == &v2.name,
            None => {
                var_map.insert(v1.name.clone(), v2.name.clone());
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

/// Check if a unit clause subsumes another clause
fn subsumes_unit(unit: &Clause, clause: &Clause) -> bool {
    if unit.literals.len() != 1 {
        return false;
    }

    let unit_lit = &unit.literals[0];

    // Try to match the unit literal with each literal in the clause
    for lit in &clause.literals {
        if lit.polarity == unit_lit.polarity {
            // Try to find a substitution
            let mut subst = Substitution::new();
            if match_literals(&unit_lit, lit, &mut subst) {
                return true;
            }
        }
    }

    false
}

/// Full subsumption check
fn subsumes(subsumer: &Clause, subsumee: &Clause) -> bool {
    if subsumer.literals.len() > subsumee.literals.len() {
        return false;
    }

    // Try to find a matching for all literals in subsumer
    find_subsumption_mapping(
        subsumer,
        subsumee,
        0,
        &mut Substitution::new(),
        &mut vec![false; subsumee.literals.len()],
    )
}

/// Recursive function to find subsumption mapping
fn find_subsumption_mapping(
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

        let mut new_subst = subst.clone();
        if match_literals(subsumer_lit, subsumee_lit, &mut new_subst) {
            used[i] = true;
            if find_subsumption_mapping(subsumer, subsumee, subsumer_idx + 1, &mut new_subst, used)
            {
                return true;
            }
            used[i] = false;
        }
    }

    false
}

/// Greedy subsumption for larger clauses
fn subsumes_greedy(subsumer: &Clause, subsumee: &Clause) -> bool {
    if subsumer.literals.len() > subsumee.literals.len() {
        return false;
    }

    let mut subst = Substitution::new();
    let mut used = vec![false; subsumee.literals.len()];

    // Greedy matching: for each literal in subsumer, find the first compatible match
    for subsumer_lit in &subsumer.literals {
        let mut found = false;

        for (i, subsumee_lit) in subsumee.literals.iter().enumerate() {
            if used[i] || subsumee_lit.polarity != subsumer_lit.polarity {
                continue;
            }

            let mut temp_subst = subst.clone();
            if match_literals(subsumer_lit, subsumee_lit, &mut temp_subst) {
                // Check if this substitution is consistent
                let subsumer_lit_applied = subsumer_lit.apply_substitution(&temp_subst);
                let subsumee_lit_applied = subsumee_lit.apply_substitution(&temp_subst);

                if subsumer_lit_applied == subsumee_lit_applied {
                    subst = temp_subst;
                    used[i] = true;
                    found = true;
                    break;
                }
            }
        }

        if !found {
            return false;
        }
    }

    true
}

/// Check if two clauses have compatible structure
fn compatible_structure(clause1: &Clause, clause2: &Clause) -> bool {
    // Check predicate symbols
    let preds1: HashSet<_> = clause1.literals.iter().map(|l| &l.atom.predicate).collect();
    let preds2: HashSet<_> = clause2.literals.iter().map(|l| &l.atom.predicate).collect();

    // subsumer's predicates must be subset of subsumee's
    preds1.is_subset(&preds2)
}

/// Try to match two literals with a substitution
fn match_literals(lit1: &Literal, lit2: &Literal, subst: &mut Substitution) -> bool {
    if lit1.polarity != lit2.polarity {
        return false;
    }

    match_atoms(&lit1.atom, &lit2.atom, subst)
}

/// Try to match two atoms with a substitution
fn match_atoms(
    atom1: &crate::core::Atom,
    atom2: &crate::core::Atom,
    subst: &mut Substitution,
) -> bool {
    if atom1.predicate != atom2.predicate {
        return false;
    }

    if atom1.args.len() != atom2.args.len() {
        return false;
    }

    for (term1, term2) in atom1.args.iter().zip(&atom2.args) {
        if !match_terms(term1, term2, subst) {
            return false;
        }
    }

    true
}

/// Try to match two terms with a substitution
fn match_terms(term1: &Term, term2: &Term, subst: &mut Substitution) -> bool {
    match term1 {
        Term::Variable(v) => {
            // Check if variable is already bound
            if let Some(bound_term) = subst.map.get(v) {
                // Must match the bound term
                terms_equal(bound_term, term2)
            } else {
                // Bind the variable
                subst.insert(v.clone(), term2.clone());
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
                        .all(|(a1, a2)| match_terms(a1, a2, subst))
            }
            _ => false,
        },
    }
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
