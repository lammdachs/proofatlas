//! ProofAtlas prover: orchestrates saturation-based theorem proving.
//!
//! The `ProofAtlas` struct combines clause management, inference rules, and
//! selection strategies to implement the given-clause saturation algorithm.
//!
//! Use `prove()` to run to completion, or `step()` for incremental execution.

use crate::logic::clause_manager::ClauseManager;
use crate::logic::{CNFFormula, Clause, ClauseKey, Interner};
use crate::state::{
    ClauseNotification, ClauseSet, Derivation, EventLog, GeneratingInference,
    InferenceResult, Proof, ProofResult, ProofStep, SaturationState, SimplifyingInference,
    StateChange,
};
use crate::config::{LiteralSelectionStrategy, ProverConfig};
use crate::profile::SaturationProfile;
use crate::index::{IndexKind, IndexRegistry};
use crate::simplifying::{TautologyRule, SubsumptionRule, DemodulationRule};
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

        // Initialize simplification rules
        let mut simplifying_inferences: Vec<Box<dyn SimplifyingInference>> = vec![
            Box::new(TautologyRule::new(&interner)),
            Box::new(DemodulationRule::new(&interner)),
            Box::new(SubsumptionRule::new(&interner)),
        ];

        // Initialize all simplification rules with input clauses
        for rule in &mut simplifying_inferences {
            rule.initialize(&initial_clauses);
        }

        // Create IndexRegistry
        let required_indices: HashSet<IndexKind> = HashSet::new();
        let mut index_registry = IndexRegistry::new(&required_indices, &interner);
        index_registry.initialize(&initial_clauses);

        // Create literal selector based on configuration
        let literal_selector: Box<dyn crate::selection::LiteralSelector> =
            match config.literal_selection {
                LiteralSelectionStrategy::Sel0 => Box::new(SelectAll),
                LiteralSelectionStrategy::Sel20 => Box::new(SelectMaximal::new()),
                LiteralSelectionStrategy::Sel21 => {
                    Box::new(SelectUniqueMaximalOrNegOrMaximal::new())
                }
                LiteralSelectionStrategy::Sel22 => Box::new(SelectNegMaxWeightOrMaximal::new()),
            };

        // Create clause manager
        let clause_manager = ClauseManager::new(interner, literal_selector);

        // Build clause storage and N set
        let mut clauses = Vec::new();
        let mut new = std::collections::VecDeque::new();
        let mut clause_memory_bytes = 0usize;

        let mut clause_idx = 0;
        for mut clause in initial_clauses.into_iter() {
            // Orient equalities before adding
            let mut oriented = clause.clone();
            clause_manager.orient_equalities(&mut oriented);

            clause.id = Some(clause_idx);

            // Track clause memory
            clause_memory_bytes += clause.memory_bytes();

            // Notify rules about pending clause
            for rule in &mut simplifying_inferences {
                rule.on_clause_pending(clause_idx, &oriented);
            }
            index_registry.on_clause_pending(clause_idx, &oriented);

            clauses.push(clause);
            new.push_back(clause_idx);
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
            event_log.push(StateChange::Add {
                clause: clauses[idx].clone(),
                derivation: Derivation::input(),
            });
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
        while let Some(clause_idx) = self.state.new.pop_front() {
            // 1a: Check empty clause immediately
            if self.state.clauses[clause_idx].is_empty() {
                let proof = self.extract_proof(clause_idx);
                return Some(ProofResult::Proof(proof));
            }

            // 1b: Apply forward simplification rules
            let clause = &self.state.clauses[clause_idx];
            let mut forward_deleted = false;
            let mut collected_changes: Vec<StateChange> = Vec::new();

            for rule in self.simplifying_inferences.iter() {
                let rule_name = rule.name();
                let t_rule = self.profile.as_ref().map(|_| Instant::now());

                let changes = rule.simplify_forward(
                    clause_idx,
                    clause,
                    &self.state.clauses,
                    &self.clause_manager.interner,
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

            // 1c: Clause survives - activate it
            for rule in &mut self.simplifying_inferences {
                rule.on_clause_activated(clause_idx, &self.state.clauses[clause_idx]);
            }
            self.index_registry
                .on_clause_activated(clause_idx, &self.state.clauses[clause_idx]);

            // 1d: Apply backward simplification rules
            let mut all_backward_changes: Vec<StateChange> = Vec::new();

            {
                let clause = &self.state.clauses[clause_idx];

                for rule in self.simplifying_inferences.iter() {
                    let rule_name = rule.name();
                    let t_rule = self.profile.as_ref().map(|_| Instant::now());

                    let changes = rule.simplify_backward(
                        clause_idx,
                        clause,
                        &self.state.clauses,
                        &self.state.unprocessed,
                        &self.state.processed,
                        &self.clause_manager.interner,
                    );

                    let count = changes
                        .iter()
                        .filter(|c| {
                            matches!(c, StateChange::Delete { .. } | StateChange::Add { .. })
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
            }

            for change in all_backward_changes {
                self.apply_change(change);
            }

            // Transfer: move to unprocessed (U)
            self.state.unprocessed.insert(clause_idx);
            self.notify_rules(ClauseSet::Unprocessed, clause_idx, true);
            self.state
                .event_log
                .push(StateChange::Transfer { clause_idx });
        }

        // === Step 2: Check saturation ===
        if self.state.unprocessed.is_empty() {
            let steps = self.build_proof_steps();
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
                let steps = self.build_proof_steps();
                let clauses = self.state.clauses.clone();
                return Some(ProofResult::Saturated(steps, clauses));
            }
        };
        if let (Some(p), Some(t)) = (self.profile.as_mut(), t0) {
            p.select_given_time += t.elapsed();
        }

        // === Step 4: Activate given clause (transfer from U to P) ===
        self.notify_rules(ClauseSet::Unprocessed, given_idx, false);
        self.state.processed.insert(given_idx);
        self.notify_rules(ClauseSet::Processed, given_idx, true);
        self.state
            .event_log
            .push(StateChange::Activate { clause_idx: given_idx });

        // === Step 5: Generate inferences ===
        let t0 = self.profile.as_ref().map(|_| Instant::now());
        let new_inferences = self.generate_inferences(given_idx);
        if let (Some(p), Some(t)) = (self.profile.as_mut(), t0) {
            p.generate_inferences_time += t.elapsed();
            p.clauses_generated += new_inferences.len();
        }

        // Add new inferences (deduplicate within the batch first)
        let t0 = self.profile.as_ref().map(|_| Instant::now());
        let mut seen_in_batch: HashSet<ClauseKey> = HashSet::new();
        for inference in new_inferences {
            let mut oriented = inference.conclusion.clone();
            self.clause_manager.orient_equalities(&mut oriented);
            let clause_key = ClauseKey::from_clause(&oriented);
            if seen_in_batch.insert(clause_key) {
                self.apply_change(StateChange::Add {
                    clause: oriented,
                    derivation: inference.derivation,
                });
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

    /// Notify all rules about a clause being added to or removed from a set
    fn notify_rules(&mut self, set: ClauseSet, clause_idx: usize, is_add: bool) {
        let clause = &self.state.clauses[clause_idx];
        let notif = if is_add {
            ClauseNotification::Added { clause_idx, clause }
        } else {
            ClauseNotification::Removed { clause_idx, clause }
        };
        for rule in &mut self.simplifying_inferences {
            rule.notify(set, notif.clone());
        }
        if set == ClauseSet::Processed {
            for rule in &mut self.generating_inferences {
                rule.notify(notif.clone());
            }
        }
    }

    /// Apply a single StateChange, update internal state, and record to event log
    fn apply_change(&mut self, change: StateChange) {
        match &change {
            StateChange::Add { clause, derivation } => {
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
                for rule in &mut self.simplifying_inferences {
                    rule.on_clause_pending(new_idx, &oriented);
                }
                self.index_registry.on_clause_pending(new_idx, &oriented);

                self.state.clause_memory_bytes += clause_with_id.memory_bytes();
                self.state.clauses.push(clause_with_id.clone());
                self.state.new.push_back(new_idx);

                self.state.event_log.push(StateChange::Add {
                    clause: clause_with_id,
                    derivation: derivation.clone(),
                });

                if let Some(p) = self.profile.as_mut() {
                    p.clauses_added += 1;
                }
            }
            StateChange::Delete {
                clause_idx,
                rule_name: _,
            } => {
                if self.state.new.contains(clause_idx) {
                    // Clause is in N - just record event
                } else if self.state.unprocessed.shift_remove(clause_idx) {
                    self.notify_rules(ClauseSet::Unprocessed, *clause_idx, false);
                    let clause = &self.state.clauses[*clause_idx];
                    self.index_registry.on_clause_removed(*clause_idx, clause);
                } else if self.state.processed.shift_remove(clause_idx) {
                    self.notify_rules(ClauseSet::Processed, *clause_idx, false);
                    let clause = &self.state.clauses[*clause_idx];
                    self.index_registry.on_clause_removed(*clause_idx, clause);
                }
                self.state.event_log.push(change);
            }
            StateChange::Transfer { clause_idx } => {
                self.state.unprocessed.insert(*clause_idx);
                self.notify_rules(ClauseSet::Unprocessed, *clause_idx, true);
                self.state.event_log.push(change);
            }
            StateChange::Activate { clause_idx } => {
                self.state.processed.insert(*clause_idx);
                self.notify_rules(ClauseSet::Processed, *clause_idx, true);
                self.state.event_log.push(change);
            }
        }
    }

    /// Check resource limits and return termination result if exceeded
    fn check_limits(&self, start_time: Instant) -> Option<ProofResult> {
        if let Some(limit_mb) = self.config.max_clause_memory_mb {
            if self.state.clause_memory_bytes >= limit_mb * 1024 * 1024 {
                return Some(ProofResult::ResourceLimit(
                    self.build_proof_steps(),
                    self.state.clauses.clone(),
                ));
            }
        }
        if self.config.max_iterations > 0
            && self.state.current_iteration >= self.config.max_iterations
        {
            return Some(ProofResult::ResourceLimit(
                self.build_proof_steps(),
                self.state.clauses.clone(),
            ));
        }
        if self.config.max_clauses > 0 && self.state.clauses.len() >= self.config.max_clauses {
            return Some(ProofResult::ResourceLimit(
                self.build_proof_steps(),
                self.state.clauses.clone(),
            ));
        }
        if start_time.elapsed() > self.config.timeout {
            return Some(ProofResult::Timeout(
                self.build_proof_steps(),
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
    fn generate_inferences(&mut self, given_idx: usize) -> Vec<InferenceResult> {
        let mut results = Vec::new();
        let given_clause = &self.state.clauses[given_idx];
        let selector = self.clause_manager.literal_selector.as_ref();

        for rule in &self.generating_inferences {
            let rule_name = rule.name();
            let t0 = self.profile.as_ref().map(|_| Instant::now());
            let before = results.len();

            let changes = rule.generate(
                given_idx,
                given_clause,
                &self.state.clauses,
                &self.state.processed,
                selector,
                &mut self.clause_manager.interner,
            );

            for change in changes {
                if let StateChange::Add { clause, derivation } = change {
                    results.push(InferenceResult {
                        conclusion: clause,
                        derivation,
                    });
                }
            }

            if let (Some(p), Some(t)) = (self.profile.as_mut(), t0) {
                let count = results.len() - before;
                p.record_generating_rule(rule_name, count, t.elapsed());
            }
        }

        results
    }

    /// Extract a proof by backward traversal from the empty clause.
    fn extract_proof(&self, empty_clause_idx: usize) -> Proof {
        let mut derivation_map = std::collections::HashMap::new();
        for event in &self.state.event_log {
            if let StateChange::Add { clause, derivation } = event {
                if let Some(idx) = clause.id {
                    derivation_map.insert(idx, derivation.clone());
                }
            }
        }

        let mut proof_clause_indices = Vec::new();
        let mut visited = HashSet::new();
        let mut to_visit = vec![empty_clause_idx];

        while let Some(idx) = to_visit.pop() {
            if !visited.insert(idx) {
                continue;
            }
            proof_clause_indices.push(idx);
            if let Some(derivation) = derivation_map.get(&idx) {
                to_visit.extend(derivation.clause_indices());
            }
        }

        proof_clause_indices.sort();

        let steps = proof_clause_indices
            .iter()
            .map(|&idx| ProofStep {
                clause_idx: idx,
                derivation: derivation_map
                    .get(&idx)
                    .cloned()
                    .unwrap_or_else(Derivation::input),
                conclusion: self.state.clauses[idx].clone(),
            })
            .collect();

        Proof {
            steps,
            empty_clause_idx,
            all_clauses: self.state.clauses.clone(),
        }
    }

    /// Build proof steps from event log.
    fn build_proof_steps(&self) -> Vec<ProofStep> {
        self.state
            .event_log
            .iter()
            .filter_map(|event| {
                if let StateChange::Add { clause, derivation } = event {
                    clause.id.map(|idx| ProofStep {
                        clause_idx: idx,
                        derivation: derivation.clone(),
                        conclusion: clause.clone(),
                    })
                } else {
                    None
                }
            })
            .collect()
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
