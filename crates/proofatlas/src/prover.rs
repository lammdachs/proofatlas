//! ProofAtlas prover: orchestrates saturation-based theorem proving.
//!
//! The `ProofAtlas` struct combines clause management, inference rules, and
//! selection strategies to implement the given-clause saturation algorithm.
//!
//! Use `prove()` to run to completion, or `step()` for incremental execution.

use crate::logic::clause_manager::ClauseManager;
use crate::logic::{CNFFormula, Clause, Interner};
use crate::state::{
    EventLog, GeneratingInference,
    ProofResult, SaturationState, SimplifyingInference,
    StateChange,
};
use crate::config::{LiteralSelectionStrategy, ProverConfig};
use crate::profile::SaturationProfile;
use crate::index::{IndexKind, IndexRegistry, SelectedLiteralIndex};
use crate::simplifying::{TautologyRule, SubsumptionRule, DemodulationRule};
use std::sync::Arc;
use crate::generating::{
    ResolutionRule, SuperpositionRule, FactoringRule,
    EqualityResolutionRule, EqualityFactoringRule,
};
use crate::selection::{
    ClauseSelector, SelectAll, SelectMaximal, SelectNegMaxWeightOrMaximal,
    SelectUniqueMaximalOrNegOrMaximal,
};
use crate::time_compat::Instant;
use std::collections::HashSet;

/// ProofAtlas prover: orchestrates saturation-based theorem proving.
///
/// Combines clause management, simplifying/generating inference rules,
/// index registry, and clause selection into the given-clause algorithm.
pub struct ProofAtlas {
    /// Prover configuration (limits, timeouts, etc.)
    pub config: ProverConfig,
    /// Centralized clause management (interner, literal selector, term ordering)
    pub clause_manager: ClauseManager,
    /// Clause state: clauses, N/U/P sets, event log
    pub state: SaturationState,
    /// Simplifying inference rules (tautology, demodulation, subsumption)
    simplifying_inferences: Vec<Box<dyn SimplifyingInference>>,
    /// Generating inference rules (resolution, superposition, factoring, etc.)
    generating_inferences: Vec<Box<dyn GeneratingInference>>,
    /// Central registry for shared indices
    index_registry: IndexRegistry,
    /// Clause selection strategy
    clause_selector: Box<dyn ClauseSelector>,
    /// Profiling data (None if profiling disabled)
    profile: Option<SaturationProfile>,
    /// Start time of the proof search
    start_time: Option<Instant>,
}

impl ProofAtlas {
    /// Create a new ProofAtlas prover from initial clauses.
    ///
    /// # Arguments
    /// * `initial_clauses` - The initial clause set
    /// * `config` - Prover configuration
    /// * `clause_selector` - Clause selection strategy
    /// * `interner` - Symbol interner for resolving symbol names
    pub fn new(
        initial_clauses: Vec<Clause>,
        config: ProverConfig,
        clause_selector: Box<dyn ClauseSelector>,
        interner: Interner,
    ) -> Self {
        let initial_clause_count = initial_clauses.len();

        // Initialize simplification rules (stateless — no lifecycle methods)
        let simplifying_inferences: Vec<Box<dyn SimplifyingInference>> = vec![
            Box::new(TautologyRule::new(&interner)),
            Box::new(DemodulationRule::new(&interner)),
            Box::new(SubsumptionRule::new(&interner)),
        ];

        // Create IndexRegistry with all required indices
        let required_indices: HashSet<IndexKind> = [
            IndexKind::UnitEqualities,
            IndexKind::Subsumption,
        ].into_iter().collect();
        let mut index_registry = IndexRegistry::new(&required_indices, &interner);
        index_registry.initialize(&initial_clauses);

        // Create literal selector based on configuration
        let literal_selector: Arc<dyn crate::selection::LiteralSelector> =
            match config.literal_selection {
                LiteralSelectionStrategy::Sel0 => Arc::new(SelectAll),
                LiteralSelectionStrategy::Sel20 => Arc::new(SelectMaximal::new()),
                LiteralSelectionStrategy::Sel21 => {
                    Arc::new(SelectUniqueMaximalOrNegOrMaximal::new())
                }
                LiteralSelectionStrategy::Sel22 => Arc::new(SelectNegMaxWeightOrMaximal::new()),
            };

        // Register SelectedLiteralIndex
        let eq_pred_id = interner.get_predicate("=");
        let sl_index = SelectedLiteralIndex::new(literal_selector.clone(), eq_pred_id);
        index_registry.add_index(Box::new(sl_index));

        // Create clause manager
        let clause_manager = ClauseManager::new(interner, literal_selector);

        // Build clause storage and N set
        let mut clauses = Vec::new();
        let mut new = Vec::new();
        let mut clause_memory_bytes = 0usize;

        let mut clause_idx = 0;
        for mut clause in initial_clauses.into_iter() {
            // Orient equalities before adding
            let mut oriented = clause.clone();
            clause_manager.orient_equalities(&mut oriented);

            clause.id = Some(clause_idx);

            // Track clause memory
            clause_memory_bytes += clause.memory_bytes();

            // Notify indices about pending clause
            index_registry.on_clause_pending(clause_idx, &oriented);

            clauses.push(clause);
            new.push(clause_idx);
            clause_idx += 1;
        }

        // Reset clause selector state
        let mut clause_selector = clause_selector;
        clause_selector.reset();

        // Initialize generating rules
        let generating_inferences: Vec<Box<dyn GeneratingInference>> = vec![
            Box::new(FactoringRule::new()),
            Box::new(EqualityResolutionRule::new()),
            Box::new(EqualityFactoringRule::new()),
            Box::new(ResolutionRule::new()),
            Box::new(SuperpositionRule::new()),
        ];

        // Build initial event log with input clauses
        let mut event_log = Vec::new();
        for idx in 0..clause_idx {
            event_log.push(StateChange::Add(
                clauses[idx].clone(),
                "Input".into(),
                vec![],
            ));
        }

        let profile = if config.enable_profiling {
            Some(SaturationProfile::default())
        } else {
            None
        };

        let state = SaturationState {
            clauses,
            processed: indexmap::IndexSet::new(),
            unprocessed: indexmap::IndexSet::new(),
            new,
            event_log,
            clause_memory_bytes,
            current_iteration: 0,
            initial_clause_count,
        };

        ProofAtlas {
            config,
            clause_manager,
            state,
            simplifying_inferences,
            generating_inferences,
            index_registry,
            clause_selector,
            profile,
            start_time: None,
        }
    }

    /// Run saturation to completion, consuming the prover.
    ///
    /// Returns the proof result, optional profiling data, event log, and interner.
    pub fn prove(mut self) -> (ProofResult, Option<SaturationProfile>, EventLog, Interner) {
        let start_time = Instant::now();
        self.start_time = Some(start_time);

        let result = loop {
            if let Some(result) = self.step() {
                break result;
            }
        };

        // Finalize profile
        if let Some(p) = self.profile.as_mut() {
            p.total_time = start_time.elapsed();
            p.selector_name = self.clause_selector.name().to_string();
            if let Some(stats) = self.clause_selector.stats() {
                p.selector_cache_hits = stats.cache_hits;
                p.selector_cache_misses = stats.cache_misses;
                p.selector_embed_time = stats.embed_time;
                p.selector_score_time = stats.score_time;
            }
        }

        (
            result,
            self.profile,
            self.state.event_log,
            self.clause_manager.interner,
        )
    }

    /// Execute one step of the saturation loop.
    ///
    /// Returns `Some(result)` if the proof search is complete, `None` to continue.
    /// Each step processes all new clauses, then selects one given clause and
    /// generates inferences with it.
    pub fn step(&mut self) -> Option<ProofResult> {
        let start_time = *self.start_time.get_or_insert_with(Instant::now);

        // === Step 1: Process new clauses ===
        while let Some(&clause_idx) = self.state.new.last() {
            // 1a: Check empty clause immediately
            if self.state.clauses[clause_idx].is_empty() {
                let proof = self.state.extract_proof(clause_idx);
                return Some(ProofResult::Proof(proof));
            }

            // 1b: Apply forward simplification rules
            let mut forward_deleted = false;
            let mut collected_changes: Vec<StateChange> = Vec::new();

            for rule in self.simplifying_inferences.iter() {
                let rule_name = rule.name();
                let t_rule = self.profile.as_ref().map(|_| Instant::now());

                let changes = rule.simplify_forward(
                    clause_idx,
                    &self.state,
                    &self.clause_manager,
                    &self.index_registry,
                );

                let success = !changes.is_empty();

                if let (Some(p), Some(t)) = (self.profile.as_mut(), t_rule) {
                    p.record_simplification_forward_attempt(rule_name, success, t.elapsed());
                    p.forward_simplify_time += t.elapsed();
                }

                if success {
                    collected_changes = changes;
                    forward_deleted = true;
                    break;
                }
            }

            for change in collected_changes {
                self.apply_change(change);
            }

            if forward_deleted {
                continue;
            }

            // 1c: Clause survives - activate it in indices
            self.index_registry
                .on_clause_activated(clause_idx, &self.state.clauses[clause_idx]);

            // 1d: Apply backward simplification rules
            let mut all_backward_changes: Vec<StateChange> = Vec::new();

            for rule in self.simplifying_inferences.iter() {
                let rule_name = rule.name();
                let t_rule = self.profile.as_ref().map(|_| Instant::now());

                let changes = rule.simplify_backward(
                    clause_idx,
                    &self.state,
                    &self.clause_manager,
                    &self.index_registry,
                );

                let count = changes
                    .iter()
                    .filter(|c| {
                        matches!(c, StateChange::Delete(..) | StateChange::Add(..))
                    })
                    .count();

                if let (Some(p), Some(t)) = (self.profile.as_mut(), t_rule) {
                    p.record_simplification_backward_attempt(rule_name, count, t.elapsed());
                    p.backward_simplify_time += t.elapsed();
                }

                if !changes.is_empty() {
                    all_backward_changes.extend(changes);
                }
            }

            for change in all_backward_changes {
                self.apply_change(change);
            }

            // 1e: Transfer N → U
            self.apply_change(StateChange::Transfer(clause_idx));
        }

        // === Step 2: Check saturation ===
        if self.state.unprocessed.is_empty() {
            let steps = self.state.build_proof_steps();
            let clauses = self.state.clauses.clone();
            return Some(ProofResult::Saturated(steps, clauses));
        }

        // Check resource limits
        if let Some(result) = self.check_limits(start_time) {
            return Some(result);
        }

        self.state.current_iteration += 1;

        // Track max sizes
        if let Some(p) = self.profile.as_mut() {
            p.iterations = self.state.current_iteration;
            let up_size = self.state.unprocessed.len();
            let pr_size = self.state.processed.len();
            if up_size > p.max_unprocessed_size {
                p.max_unprocessed_size = up_size;
            }
            if pr_size > p.max_processed_size {
                p.max_processed_size = pr_size;
            }
        }

        // === Step 3: Select given clause ===
        let t0 = self.profile.as_ref().map(|_| Instant::now());
        let given_idx = match self.select_given_clause() {
            Some(idx) => idx,
            None => {
                let steps = self.state.build_proof_steps();
                let clauses = self.state.clauses.clone();
                return Some(ProofResult::Saturated(steps, clauses));
            }
        };
        if let (Some(p), Some(t)) = (self.profile.as_mut(), t0) {
            p.select_given_time += t.elapsed();
        }

        // === Step 4: Activate given clause (transfer from U to P) ===
        self.apply_change(StateChange::Activate(given_idx));

        // === Step 5: Generate inferences ===
        let t0 = self.profile.as_ref().map(|_| Instant::now());
        let new_changes = self.generate_inferences(given_idx);
        if let (Some(p), Some(t)) = (self.profile.as_mut(), t0) {
            p.generate_inferences_time += t.elapsed();
            p.clauses_generated += new_changes.len();
        }

        // Add new inferences
        let t0 = self.profile.as_ref().map(|_| Instant::now());
        for change in new_changes {
            if let StateChange::Add(clause, rule_name, premises) = change {
                let mut oriented = clause.clone();
                self.clause_manager.orient_equalities(&mut oriented);
                self.apply_change(StateChange::Add(oriented, rule_name, premises));
            }
        }
        if let (Some(p), Some(t)) = (self.profile.as_mut(), t0) {
            p.add_inferences_time += t.elapsed();
        }

        None // Continue
    }

    // =========================================================================
    // Private helper methods
    // =========================================================================

    /// Apply a single StateChange, update internal state, and record to event log
    fn apply_change(&mut self, change: StateChange) {
        match &change {
            StateChange::Add(clause, _rule_name, _premises) => {
                if clause.literals.len() > self.config.max_clause_size {
                    return;
                }

                let new_idx = self.state.clauses.len();
                let mut clause_with_id = clause.clone();
                clause_with_id.id = Some(new_idx);
                clause_with_id.age = self.state.current_iteration;
                clause_with_id.role = crate::logic::ClauseRole::Derived;

                let mut oriented = clause.clone();
                self.clause_manager.orient_equalities(&mut oriented);
                self.index_registry.on_clause_pending(new_idx, &oriented);

                self.state.clause_memory_bytes += clause_with_id.memory_bytes();
                self.state.clauses.push(clause_with_id.clone());
                self.state.new.push(new_idx);

                // Re-construct with updated clause (includes id)
                if let StateChange::Add(_, rule_name, premises) = change {
                    self.state.event_log.push(StateChange::Add(
                        clause_with_id,
                        rule_name,
                        premises,
                    ));
                }

                if let Some(p) = self.profile.as_mut() {
                    p.clauses_added += 1;
                }
            }
            StateChange::Delete(clause_idx, _rule_name, _justification) => {
                let clause_idx = *clause_idx;
                // Remove from whichever set contains the clause
                if self.state.new.last() == Some(&clause_idx) {
                    self.state.new.pop();
                } else if self.state.unprocessed.shift_remove(&clause_idx) {
                    let clause = &self.state.clauses[clause_idx];
                    self.index_registry.on_clause_removed(clause_idx, clause);
                } else if self.state.processed.shift_remove(&clause_idx) {
                    let clause = &self.state.clauses[clause_idx];
                    self.index_registry.on_clause_removed(clause_idx, clause);
                }
                self.state.event_log.push(change);
            }
            StateChange::Transfer(clause_idx) => {
                let clause_idx = *clause_idx;
                // N → U
                if self.state.new.last() == Some(&clause_idx) {
                    self.state.new.pop();
                }
                self.state.unprocessed.insert(clause_idx);
                self.state.event_log.push(change);
            }
            StateChange::Activate(clause_idx) => {
                let clause_idx = *clause_idx;
                // U → P (selector already removed from U)
                self.state.unprocessed.shift_remove(&clause_idx);
                self.state.processed.insert(clause_idx);
                self.index_registry.on_clause_processed(clause_idx, &self.state.clauses[clause_idx]);
                self.state.event_log.push(change);
            }
        }
    }

    /// Check resource limits and return termination result if exceeded
    fn check_limits(&self, start_time: Instant) -> Option<ProofResult> {
        if let Some(limit_mb) = self.config.max_clause_memory_mb {
            if self.state.clause_memory_bytes >= limit_mb * 1024 * 1024 {
                return Some(ProofResult::ResourceLimit(
                    self.state.build_proof_steps(),
                    self.state.clauses.clone(),
                ));
            }
        }
        if self.config.max_iterations > 0
            && self.state.current_iteration >= self.config.max_iterations
        {
            return Some(ProofResult::ResourceLimit(
                self.state.build_proof_steps(),
                self.state.clauses.clone(),
            ));
        }
        if self.config.max_clauses > 0 && self.state.clauses.len() >= self.config.max_clauses {
            return Some(ProofResult::ResourceLimit(
                self.state.build_proof_steps(),
                self.state.clauses.clone(),
            ));
        }
        if start_time.elapsed() > self.config.timeout {
            return Some(ProofResult::Timeout(
                self.state.build_proof_steps(),
                self.state.clauses.clone(),
            ));
        }
        None
    }

    /// Select the next given clause using the configured selector
    fn select_given_clause(&mut self) -> Option<usize> {
        self.clause_selector
            .select(&mut self.state.unprocessed, &self.state.clauses)
    }

    /// Generate all inferences with the given clause
    fn generate_inferences(&mut self, given_idx: usize) -> Vec<StateChange> {
        let mut results = Vec::new();

        // Take generating_inferences out of self to avoid borrow conflict
        // (we need &self.state and &mut self.clause_manager simultaneously)
        let generating_inferences = std::mem::take(&mut self.generating_inferences);

        for rule in &generating_inferences {
            let rule_name = rule.name();
            let t0 = self.profile.as_ref().map(|_| Instant::now());
            let before = results.len();

            let changes = rule.generate(
                given_idx,
                &self.state,
                &mut self.clause_manager,
                &self.index_registry,
            );

            results.extend(changes);

            if let (Some(p), Some(t)) = (self.profile.as_mut(), t0) {
                let count = results.len() - before;
                p.record_generating_rule(rule_name, count, t.elapsed());
            }
        }

        // Put generating_inferences back
        self.generating_inferences = generating_inferences;

        results
    }
}

/// Run saturation on a CNF formula
pub fn saturate(
    formula: CNFFormula,
    config: ProverConfig,
    clause_selector: Box<dyn ClauseSelector>,
    interner: Interner,
) -> (ProofResult, Option<SaturationProfile>, EventLog, Interner) {
    let prover = ProofAtlas::new(formula.clauses, config, clause_selector, interner);
    prover.prove()
}
