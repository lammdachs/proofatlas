//! Polymorphic rule architecture for saturation-based theorem proving.
//!
//! This module provides a modular framework for defining and applying inference rules:
//! - **SimplificationRule**: Rules that simplify or delete clauses (tautology, subsumption, demodulation)
//! - **GeneratingInference**: Rules that generate new clauses (resolution, superposition, factoring)
//!
//! All rules return `Vec<ProofStateChange>` representing atomic modifications to the proof state.

use crate::fol::Clause;
use crate::inference::Derivation;
use crate::selection::LiteralSelector;
use serde::{Deserialize, Serialize};

/// Atomic operations on the proof state.
///
/// These operations represent all possible modifications to the clause sets:
/// - N (new): Fresh clauses awaiting simplification
/// - U (unprocessed): Simplified clauses awaiting selection
/// - P (processed): Selected clauses used for inferences
///
/// Semantics distinguish between transitions (implicit removal from source set)
/// and deletions (explicit removal due to simplification):
/// - `New`: clause enters N
/// - `Transfer`: clause moves N→U (implicit N removal)
/// - `Select`: clause moves U→P (implicit U removal)
/// - `DeleteN/U/P`: clause deleted from respective set (simplification)
///
/// This is the raw event log format. All derived views (proof extraction, training data,
/// iteration structure) come from replaying these events.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ProofStateChange {
    /// New clause added to N (from inference or input)
    New { clause: Clause, derivation: Derivation },
    /// Clause deleted from N (forward simplification)
    DeleteN { clause_idx: usize, rule_name: String },
    /// Clause transferred from N to U (survived forward simplification)
    Transfer { clause_idx: usize },
    /// Clause deleted from U (backward simplification)
    DeleteU { clause_idx: usize, rule_name: String },
    /// Clause selected and transferred from U to P
    Select { clause_idx: usize },
    /// Clause deleted from P (backward simplification)
    DeleteP { clause_idx: usize, rule_name: String },
}

/// Type alias for the event log (replaces semantic SaturationTrace)
pub type SaturationEventLog = Vec<ProofStateChange>;

/// Which clause set a notification refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClauseSet {
    /// Unprocessed set (U)
    Unprocessed,
    /// Processed set (P)
    Processed,
}

/// Notifications sent to rules when clauses are added/removed from U or P.
///
/// Rules can maintain internal indices by listening to these notifications.
#[derive(Debug, Clone)]
pub enum ClauseNotification<'a> {
    /// Clause was added to a set
    Added { clause_idx: usize, clause: &'a Clause },
    /// Clause was removed from a set
    Removed { clause_idx: usize, clause: &'a Clause },
}

/// Read-only view into a clause set.
///
/// Provides iteration and lookup for rules that need to examine clauses in U or P.
pub struct ClauseView<'a> {
    /// Indices of clauses in this view
    indices: &'a [usize],
    /// All clauses (indexed by clause_idx)
    clauses: &'a [Clause],
}

impl<'a> ClauseView<'a> {
    /// Create a new clause view
    pub fn new(indices: &'a [usize], clauses: &'a [Clause]) -> Self {
        ClauseView { indices, clauses }
    }

    /// Iterate over (index, clause) pairs in this view
    pub fn iter(&self) -> impl Iterator<Item = (usize, &Clause)> + '_ {
        self.indices
            .iter()
            .filter_map(|&idx| self.clauses.get(idx).map(|c| (idx, c)))
    }

    /// Get a clause by index (returns None if not in this view)
    pub fn get(&self, idx: usize) -> Option<&Clause> {
        if self.indices.contains(&idx) {
            self.clauses.get(idx)
        } else {
            None
        }
    }

    /// Get the number of clauses in this view
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Check if this view is empty
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Get all clause indices in this view
    pub fn indices(&self) -> &[usize] {
        self.indices
    }
}

/// Trait for simplification rules (tautology, subsumption, demodulation).
///
/// Simplification rules can:
/// - **Forward simplify**: Simplify/delete a new clause using existing clauses in U∪P
/// - **Backward simplify**: Simplify/delete existing clauses in U∪P using a new clause
///
/// Lifecycle methods allow rules to maintain indices:
/// - `initialize`: Called once with all input clauses
/// - `on_clause_pending`: Called when a clause is added to N
/// - `on_clause_activated`: Called when a clause transfers from N to U
pub trait SimplificationRule: Send + Sync {
    /// Get the name of this rule
    fn name(&self) -> &str;

    /// Initialize the rule with input clauses.
    /// Called once before saturation begins.
    fn initialize(&mut self, _clauses: &[Clause]) {}

    /// Called when a clause is added to N (pending, not yet active).
    /// Rules can use this to prepare internal indices.
    fn on_clause_pending(&mut self, _clause_idx: usize, _clause: &Clause) {}

    /// Called when a clause is activated (transferred from N to U).
    /// Rules can use this to update their active clause indices.
    fn on_clause_activated(&mut self, _clause_idx: usize, _clause: &Clause) {}

    /// Notified when a clause is added to or removed from U or P.
    fn notify(&mut self, _set: ClauseSet, _notif: ClauseNotification) {}

    /// Forward simplification: try to simplify/delete a clause in N using U∪P.
    ///
    /// Returns:
    /// - `[]` if no simplification applies
    /// - `[DeleteN { clause_idx }]` if the clause should be deleted
    /// - `[DeleteN { clause_idx }, New { clause, derivation }]` if the clause is replaced
    fn simplify_forward(
        &self,
        clause_idx: usize,
        clause: &Clause,
        unprocessed: &ClauseView,
        processed: &ClauseView,
        all_clauses: &[Clause],
    ) -> Vec<ProofStateChange>;

    /// Backward simplification: simplify clauses in U∪P using this clause.
    ///
    /// Returns DeleteU/DeleteP for deleted clauses, New for replacements.
    /// Default implementation returns empty (no backward simplification).
    fn simplify_backward(
        &self,
        _clause_idx: usize,
        _clause: &Clause,
        _unprocessed: &ClauseView,
        _processed: &ClauseView,
    ) -> Vec<ProofStateChange> {
        vec![]
    }
}

/// Trait for generating inference rules (resolution, superposition, factoring, etc.).
///
/// Generating rules produce new clauses by combining the given clause with processed clauses.
pub trait GeneratingInferenceRule: Send + Sync {
    /// Get the name of this rule
    fn name(&self) -> &str;

    /// Notified when a clause is added to or removed from P (processed)
    fn notify(&mut self, _notif: ClauseNotification) {}

    /// Generate inferences with the given clause.
    ///
    /// Should generate all inferences between:
    /// - The given clause and each clause in P (processed)
    /// - The given clause with itself (self-inferences)
    ///
    /// Returns New changes for each new clause generated.
    fn generate(
        &self,
        given_idx: usize,
        given: &Clause,
        processed: &ClauseView,
        selector: &dyn LiteralSelector,
    ) -> Vec<ProofStateChange>;
}

// =============================================================================
// Simplification Rule Implementations
// =============================================================================

/// Tautology deletion rule.
///
/// Deletes clauses that are tautologies (contain complementary literals or
/// reflexive equalities like t=t).
pub struct TautologyRule;

impl TautologyRule {
    pub fn new() -> Self {
        TautologyRule
    }
}

impl Default for TautologyRule {
    fn default() -> Self {
        Self::new()
    }
}

impl SimplificationRule for TautologyRule {
    fn name(&self) -> &str {
        "Tautology"
    }

    fn simplify_forward(
        &self,
        clause_idx: usize,
        clause: &Clause,
        _unprocessed: &ClauseView,
        _processed: &ClauseView,
        _all_clauses: &[Clause],
    ) -> Vec<ProofStateChange> {
        if clause.is_tautology() {
            vec![ProofStateChange::DeleteN { clause_idx, rule_name: self.name().into() }]
        } else {
            vec![]
        }
    }
}

/// Forward subsumption rule.
///
/// Deletes clauses that are subsumed by existing clauses in U∪P.
/// Uses the SubsumptionChecker for efficient subsumption testing.
use super::subsumption::SubsumptionChecker;

pub struct SubsumptionRule {
    checker: SubsumptionChecker,
}

impl SubsumptionRule {
    pub fn new() -> Self {
        SubsumptionRule {
            checker: SubsumptionChecker::new(),
        }
    }

    /// Initialize symbols for the feature index
    pub fn initialize_symbols(&mut self, clauses: &[Clause]) {
        self.checker.initialize_symbols(clauses);
    }

    /// Add a clause to the subsumption checker (pending, not yet active)
    pub fn add_clause_pending(&mut self, clause: Clause) -> usize {
        self.checker.add_clause_pending(clause)
    }

    /// Activate a clause (transfer from N to U or P)
    pub fn activate_clause(&mut self, idx: usize) {
        self.checker.activate_clause(idx);
    }

    /// Find clauses subsumed by a given clause
    pub fn find_subsumed_by(&self, subsumer_idx: usize, candidate_indices: &[usize]) -> Vec<usize> {
        self.checker.find_subsumed_by(subsumer_idx, candidate_indices)
    }
}

impl Default for SubsumptionRule {
    fn default() -> Self {
        Self::new()
    }
}

impl SimplificationRule for SubsumptionRule {
    fn name(&self) -> &str {
        "Subsumption"
    }

    fn initialize(&mut self, clauses: &[Clause]) {
        self.checker.initialize_symbols(clauses);
    }

    fn on_clause_pending(&mut self, _clause_idx: usize, clause: &Clause) {
        self.checker.add_clause_pending(clause.clone());
    }

    fn on_clause_activated(&mut self, clause_idx: usize, _clause: &Clause) {
        self.checker.activate_clause(clause_idx);
    }

    fn simplify_forward(
        &self,
        clause_idx: usize,
        clause: &Clause,
        _unprocessed: &ClauseView,
        _processed: &ClauseView,
        _all_clauses: &[Clause],
    ) -> Vec<ProofStateChange> {
        if let Some(_subsumer_idx) = self.checker.find_subsumer(clause) {
            vec![ProofStateChange::DeleteN { clause_idx, rule_name: self.name().into() }]
        } else {
            vec![]
        }
    }

    fn simplify_backward(
        &self,
        clause_idx: usize,
        _clause: &Clause,
        unprocessed: &ClauseView,
        processed: &ClauseView,
    ) -> Vec<ProofStateChange> {
        let mut changes = Vec::new();

        // Collect all candidate indices
        let all_indices: Vec<usize> = unprocessed
            .indices()
            .iter()
            .chain(processed.indices().iter())
            .copied()
            .collect();

        // Find clauses subsumed by this clause
        let subsumed = self.checker.find_subsumed_by(clause_idx, &all_indices);

        let rule_name: String = self.name().into();
        for idx in subsumed {
            if processed.indices().contains(&idx) {
                changes.push(ProofStateChange::DeleteP { clause_idx: idx, rule_name: rule_name.clone() });
            } else if unprocessed.indices().contains(&idx) {
                changes.push(ProofStateChange::DeleteU { clause_idx: idx, rule_name: rule_name.clone() });
            }
        }

        changes
    }
}

/// Demodulation rule (rewriting with unit equalities).
///
/// Rewrites terms in clauses using oriented unit equalities.
use crate::inference::demodulation;
use crate::parser::orient_equalities::orient_clause_equalities;

pub struct DemodulationRule {
    /// Unit equality clause indices in U∪P
    unit_equalities: std::collections::HashSet<usize>,
}

impl DemodulationRule {
    pub fn new() -> Self {
        DemodulationRule {
            unit_equalities: std::collections::HashSet::new(),
        }
    }

    /// Check if a clause is a unit positive equality
    fn is_unit_equality(clause: &Clause) -> bool {
        clause.literals.len() == 1
            && clause.literals[0].polarity
            && clause.literals[0].atom.is_equality()
    }

    /// Add a unit equality to the index
    pub fn add_unit_equality(&mut self, clause_idx: usize) {
        self.unit_equalities.insert(clause_idx);
    }

    /// Remove a unit equality from the index
    pub fn remove_unit_equality(&mut self, clause_idx: usize) {
        self.unit_equalities.remove(&clause_idx);
    }

    /// Get all unit equality indices
    pub fn unit_equalities(&self) -> &std::collections::HashSet<usize> {
        &self.unit_equalities
    }
}

impl Default for DemodulationRule {
    fn default() -> Self {
        Self::new()
    }
}

impl SimplificationRule for DemodulationRule {
    fn name(&self) -> &str {
        "Demodulation"
    }

    fn on_clause_activated(&mut self, clause_idx: usize, clause: &Clause) {
        if Self::is_unit_equality(clause) {
            self.unit_equalities.insert(clause_idx);
        }
    }

    fn notify(&mut self, _set: ClauseSet, notif: ClauseNotification) {
        // Track unit equalities in both U and P
        match notif {
            ClauseNotification::Added { clause_idx, clause } => {
                if Self::is_unit_equality(clause) {
                    self.unit_equalities.insert(clause_idx);
                }
            }
            ClauseNotification::Removed { clause_idx, .. } => {
                self.unit_equalities.remove(&clause_idx);
            }
        }
    }

    fn simplify_forward(
        &self,
        clause_idx: usize,
        clause: &Clause,
        _unprocessed: &ClauseView,
        _processed: &ClauseView,
        all_clauses: &[Clause],
    ) -> Vec<ProofStateChange> {
        // Try to demodulate using each unit equality
        for &unit_idx in &self.unit_equalities {
            if let Some(unit_clause) = all_clauses.get(unit_idx) {
                let results = demodulation::demodulate(unit_clause, clause, unit_idx, clause_idx);
                if !results.is_empty() {
                    let mut simplified_clause = results[0].conclusion.clone();
                    orient_clause_equalities(&mut simplified_clause);

                    return vec![
                        ProofStateChange::DeleteN { clause_idx, rule_name: self.name().into() },
                        ProofStateChange::New {
                            clause: simplified_clause,
                            derivation: Derivation {
                                rule_name: "Demodulation".into(),
                                premises: vec![unit_idx, clause_idx],
                            },
                        },
                    ];
                }
            }
        }
        vec![]
    }

    fn simplify_backward(
        &self,
        clause_idx: usize,
        clause: &Clause,
        unprocessed: &ClauseView,
        processed: &ClauseView,
    ) -> Vec<ProofStateChange> {
        // Only unit equalities can backward-demodulate
        if !Self::is_unit_equality(clause) {
            return vec![];
        }

        let mut changes = Vec::new();
        let rule_name: String = self.name().into();

        // Try to demodulate each clause in U∪P
        for (target_idx, target_clause) in unprocessed.iter().chain(processed.iter()) {
            if target_idx == clause_idx {
                continue;
            }

            let results = demodulation::demodulate(clause, target_clause, clause_idx, target_idx);
            if !results.is_empty() {
                let mut simplified_clause = results[0].conclusion.clone();
                orient_clause_equalities(&mut simplified_clause);

                // Determine which set to remove from
                if processed.indices().contains(&target_idx) {
                    changes.push(ProofStateChange::DeleteP { clause_idx: target_idx, rule_name: rule_name.clone() });
                } else {
                    changes.push(ProofStateChange::DeleteU { clause_idx: target_idx, rule_name: rule_name.clone() });
                }

                // Add the simplified clause to N
                changes.push(ProofStateChange::New {
                    clause: simplified_clause,
                    derivation: Derivation {
                        rule_name: "Demodulation".into(),
                        premises: vec![clause_idx, target_idx],
                    },
                });
            }
        }

        changes
    }
}

// =============================================================================
// Generating Inference Rule Implementations
// =============================================================================

use crate::inference::{
    equality_factoring, equality_resolution, factoring, resolution, superposition,
};

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

impl GeneratingInferenceRule for ResolutionRule {
    fn name(&self) -> &str {
        "Resolution"
    }

    fn generate(
        &self,
        given_idx: usize,
        given: &Clause,
        processed: &ClauseView,
        selector: &dyn LiteralSelector,
    ) -> Vec<ProofStateChange> {
        let mut changes = Vec::new();

        // Resolution with processed clauses
        for (processed_idx, processed_clause) in processed.iter() {
            // Given as first clause
            for result in resolution(given, processed_clause, given_idx, processed_idx, selector) {
                changes.push(ProofStateChange::New {
                    clause: result.conclusion,
                    derivation: result.derivation,
                });
            }
            // Given as second clause
            for result in resolution(processed_clause, given, processed_idx, given_idx, selector) {
                changes.push(ProofStateChange::New {
                    clause: result.conclusion,
                    derivation: result.derivation,
                });
            }
        }

        // Self-resolution
        for result in resolution(given, given, given_idx, given_idx, selector) {
            changes.push(ProofStateChange::New {
                clause: result.conclusion,
                derivation: result.derivation,
            });
        }

        changes
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

impl GeneratingInferenceRule for SuperpositionRule {
    fn name(&self) -> &str {
        "Superposition"
    }

    fn generate(
        &self,
        given_idx: usize,
        given: &Clause,
        processed: &ClauseView,
        selector: &dyn LiteralSelector,
    ) -> Vec<ProofStateChange> {
        let mut changes = Vec::new();

        // Superposition with processed clauses
        for (processed_idx, processed_clause) in processed.iter() {
            // Given as first clause (rewriter)
            for result in superposition(given, processed_clause, given_idx, processed_idx, selector)
            {
                changes.push(ProofStateChange::New {
                    clause: result.conclusion,
                    derivation: result.derivation,
                });
            }
            // Given as second clause (target)
            for result in superposition(processed_clause, given, processed_idx, given_idx, selector)
            {
                changes.push(ProofStateChange::New {
                    clause: result.conclusion,
                    derivation: result.derivation,
                });
            }
        }

        // Self-superposition
        for result in superposition(given, given, given_idx, given_idx, selector) {
            changes.push(ProofStateChange::New {
                clause: result.conclusion,
                derivation: result.derivation,
            });
        }

        changes
    }
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

impl GeneratingInferenceRule for FactoringRule {
    fn name(&self) -> &str {
        "Factoring"
    }

    fn generate(
        &self,
        given_idx: usize,
        given: &Clause,
        _processed: &ClauseView,
        selector: &dyn LiteralSelector,
    ) -> Vec<ProofStateChange> {
        factoring(given, given_idx, selector)
            .into_iter()
            .map(|result| ProofStateChange::New {
                clause: result.conclusion,
                derivation: result.derivation,
            })
            .collect()
    }
}

/// Equality resolution inference rule.
///
/// Resolves negative equalities of the form s≠s.
pub struct EqualityResolutionRule;

impl EqualityResolutionRule {
    pub fn new() -> Self {
        EqualityResolutionRule
    }
}

impl Default for EqualityResolutionRule {
    fn default() -> Self {
        Self::new()
    }
}

impl GeneratingInferenceRule for EqualityResolutionRule {
    fn name(&self) -> &str {
        "EqualityResolution"
    }

    fn generate(
        &self,
        given_idx: usize,
        given: &Clause,
        _processed: &ClauseView,
        selector: &dyn LiteralSelector,
    ) -> Vec<ProofStateChange> {
        equality_resolution(given, given_idx, selector)
            .into_iter()
            .map(|result| ProofStateChange::New {
                clause: result.conclusion,
                derivation: result.derivation,
            })
            .collect()
    }
}

/// Equality factoring inference rule.
///
/// Factors positive equalities.
pub struct EqualityFactoringRule;

impl EqualityFactoringRule {
    pub fn new() -> Self {
        EqualityFactoringRule
    }
}

impl Default for EqualityFactoringRule {
    fn default() -> Self {
        Self::new()
    }
}

impl GeneratingInferenceRule for EqualityFactoringRule {
    fn name(&self) -> &str {
        "EqualityFactoring"
    }

    fn generate(
        &self,
        given_idx: usize,
        given: &Clause,
        _processed: &ClauseView,
        selector: &dyn LiteralSelector,
    ) -> Vec<ProofStateChange> {
        equality_factoring(given, given_idx, selector)
            .into_iter()
            .map(|result| ProofStateChange::New {
                clause: result.conclusion,
                derivation: result.derivation,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fol::{Atom, Constant, Literal, PredicateSymbol, Term, Variable};

    fn make_clause(literals: Vec<Literal>) -> Clause {
        Clause::new(literals)
    }

    fn make_literal(pred: &str, args: Vec<Term>, positive: bool) -> Literal {
        let atom = Atom {
            predicate: PredicateSymbol {
                name: pred.to_string(),
                arity: args.len(),
            },
            args,
        };
        if positive {
            Literal::positive(atom)
        } else {
            Literal::negative(atom)
        }
    }

    fn var(name: &str) -> Term {
        Term::Variable(Variable {
            name: name.to_string(),
        })
    }

    fn const_(name: &str) -> Term {
        Term::Constant(Constant {
            name: name.to_string(),
        })
    }

    #[test]
    fn test_tautology_rule() {
        let rule = TautologyRule::new();
        let empty_u = Vec::new();
        let empty_p = Vec::new();
        let all_clauses = Vec::new();
        let u_view = ClauseView::new(&empty_u, &all_clauses);
        let p_view = ClauseView::new(&empty_p, &all_clauses);

        // Non-tautology
        let clause = make_clause(vec![
            make_literal("P", vec![var("X")], true),
            make_literal("Q", vec![const_("a")], false),
        ]);
        let changes = rule.simplify_forward(0, &clause, &u_view, &p_view, &all_clauses);
        assert!(changes.is_empty());

        // Tautology (complementary literals)
        let tautology = make_clause(vec![
            make_literal("P", vec![var("X")], true),
            make_literal("P", vec![var("X")], false),
        ]);
        let changes = rule.simplify_forward(0, &tautology, &u_view, &p_view, &all_clauses);
        assert_eq!(changes.len(), 1);
        assert!(matches!(changes[0], ProofStateChange::DeleteN { clause_idx: 0, .. }));
    }

    #[test]
    fn test_clause_view() {
        let clauses = vec![
            make_clause(vec![make_literal("P", vec![const_("a")], true)]),
            make_clause(vec![make_literal("Q", vec![const_("b")], true)]),
            make_clause(vec![make_literal("R", vec![const_("c")], true)]),
        ];
        let indices = vec![0, 2];
        let view = ClauseView::new(&indices, &clauses);

        assert_eq!(view.len(), 2);
        assert!(!view.is_empty());

        // Can get clauses in the view
        assert!(view.get(0).is_some());
        assert!(view.get(2).is_some());

        // Cannot get clause not in the view
        assert!(view.get(1).is_none());

        // Iteration works
        let collected: Vec<_> = view.iter().collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0].0, 0);
        assert_eq!(collected[1].0, 2);
    }
}
