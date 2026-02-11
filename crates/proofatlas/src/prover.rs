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
    /// Initial clauses to be added during init()
    initial_clauses: Vec<Clause>,
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

        // Reset clause selector state and provide interner for symbol name resolution
        let mut clause_selector = clause_selector;
        clause_selector.reset();
        clause_selector.set_interner(Arc::new(clause_manager.interner.clone()));

        // Initialize generating rules
        let generating_inferences: Vec<Box<dyn GeneratingInference>> = vec![
            Box::new(FactoringRule::new()),
            Box::new(EqualityResolutionRule::new()),
            Box::new(EqualityFactoringRule::new()),
            Box::new(ResolutionRule::new()),
            Box::new(SuperpositionRule::new()),
        ];

        let profile = if config.enable_profiling {
            Some(SaturationProfile::default())
        } else {
            None
        };

        let state = SaturationState {
            clauses: Vec::new(),
            processed: indexmap::IndexSet::new(),
            unprocessed: indexmap::IndexSet::new(),
            new: Vec::new(),
            event_log: Vec::new(),
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
            initial_clauses,
        }
    }

    /// Run saturation to completion, consuming the prover.
    ///
    /// Returns the proof result, optional profiling data, event log, and interner.
    pub fn prove(mut self) -> (ProofResult, Option<SaturationProfile>, EventLog, Interner) {
        let start_time = Instant::now();
        self.start_time = Some(start_time);

        if let Some(result) = self.init() {
            return (
                result,
                self.profile,
                self.state.event_log,
                self.clause_manager.interner,
            );
        }

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
        self.start_time.get_or_insert_with(Instant::now);

        // === Step 1: Process new clauses ===
        'simplify: while let Some(&clause_idx) = self.state.new.last() {
            // 1a: Apply forward simplification rules
            let mut forward_change: Option<StateChange> = None;
            for rule in self.simplifying_inferences.iter() {
                let rule_name = rule.name();
                let t_rule = self.profile.as_ref().map(|_| Instant::now());

                let change = rule.simplify_forward(
                    clause_idx,
                    &self.state,
                    &self.clause_manager,
                    &self.index_registry,
                );

                if let (Some(p), Some(t)) = (self.profile.as_mut(), t_rule) {
                    p.record_simplification_forward_attempt(rule_name, change.is_some(), t.elapsed());
                    p.forward_simplify_time += t.elapsed();
                }

                if change.is_some() {
                    forward_change = change;
                    break;
                }
            }

            if let Some(change) = forward_change {
                if let Some(result) = self.apply_change(change) {
                    return Some(result);
                }
                continue 'simplify;
            }

            // 1c: Transfer N → U (activates clause in indices via on_transfer)
            if let Some(result) = self.apply_change(StateChange::Transfer(clause_idx)) {
                return Some(result);
            }

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

                let count = changes.len();

                if let (Some(p), Some(t)) = (self.profile.as_mut(), t_rule) {
                    p.record_simplification_backward_attempt(rule_name, count, t.elapsed());
                    p.backward_simplify_time += t.elapsed();
                }

                if !changes.is_empty() {
                    all_backward_changes.extend(changes);
                }
            }

            for change in all_backward_changes {
                if let Some(result) = self.apply_change(change) {
                    return Some(result);
                }
            }
        }

        // === Step 2: Check saturation ===
        if self.state.unprocessed.is_empty() {
            let steps = self.state.build_proof_steps();
            let clauses = self.state.clauses.clone();
            return Some(ProofResult::Saturated(steps, clauses));
        }

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

        // === Step 2b: Check timeout before selection ===
        // When using a remote scoring server, select() can block for seconds
        // waiting on server mutex contention.  Check wall-clock timeout here
        // so the prover exits promptly after a long-blocking select().
        if let Some(start) = self.start_time {
            if start.elapsed() > self.config.timeout {
                return Some(ProofResult::ResourceLimit(
                    self.state.build_proof_steps(),
                    self.state.clauses.clone(),
                ));
            }
        }

        // === Step 3: Select given clause ===
        let t0 = self.profile.as_ref().map(|_| Instant::now());
        let given_idx = match self.select_given_clause() {
            Some(idx) => idx,
            None if self.state.unprocessed.is_empty() => {
                let steps = self.state.build_proof_steps();
                let clauses = self.state.clauses.clone();
                return Some(ProofResult::Saturated(steps, clauses));
            }
            None => {
                // Selector returned None with clauses still available — selector error
                // (e.g., scoring server died). Report as resource limit.
                let steps = self.state.build_proof_steps();
                let clauses = self.state.clauses.clone();
                return Some(ProofResult::ResourceLimit(steps, clauses));
            }
        };
        if let (Some(p), Some(t)) = (self.profile.as_mut(), t0) {
            p.select_given_time += t.elapsed();
        }

        // === Step 4: Activate given clause (transfer from U to P) ===
        self.clause_selector.on_clause_processed(given_idx);
        if let Some(result) = self.apply_change(StateChange::Activate(given_idx)) {
            return Some(result);
        }

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
                if let Some(result) = self.apply_change(StateChange::Add(oriented, rule_name, premises)) {
                    return Some(result);
                }
            }
        }
        if let (Some(p), Some(t)) = (self.profile.as_mut(), t0) {
            p.add_inferences_time += t.elapsed();
        }

        None // Continue
    }

    /// Add initial clauses to the N set via `apply_change`.
    ///
    /// Returns `Some(result)` if an empty clause is found in the input.
    /// Must be called exactly once before the first `step()`.
    pub fn init(&mut self) -> Option<ProofResult> {
        let initial_clauses = std::mem::take(&mut self.initial_clauses);
        for clause in initial_clauses {
            if let Some(result) = self.apply_change(StateChange::Add(clause, "Input".into(), vec![])) {
                return Some(result);
            }
        }
        None
    }

    // =========================================================================
    // Private helper methods
    // =========================================================================

    /// Apply a single StateChange, update internal state, record to event log,
    /// and check resource limits.
    ///
    /// Returns `Some(ProofResult::ResourceLimit(..))` if a limit is exceeded.
    fn apply_change(&mut self, change: StateChange) -> Option<ProofResult> {
        match &change {
            StateChange::Add(clause, _rule_name, _premises) => {
                let is_input = matches!(&_rule_name as &str, "Input");

                // Never silently drop input clauses
                if !is_input && clause.literals.len() > self.config.max_clause_size {
                    return None;
                }

                let new_idx = self.state.clauses.len();
                let mut clause_with_id = clause.clone();
                clause_with_id.id = Some(new_idx);
                if !is_input {
                    clause_with_id.age = self.state.current_iteration;
                    clause_with_id.role = crate::logic::ClauseRole::Derived;
                }

                let mut oriented = clause.clone();
                self.clause_manager.orient_equalities(&mut oriented);
                self.index_registry.on_add(new_idx, &oriented);

                let is_empty = clause_with_id.is_empty();

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

                // Empty clause → proof found
                if is_empty {
                    return Some(ProofResult::Proof(self.state.extract_proof(new_idx)));
                }

                if let Some(p) = self.profile.as_mut() {
                    p.clauses_added += 1;
                }

                // Skip resource limit checks for input clauses
                if !is_input {
                    // Check limits after every Add
                    let num_clauses = self.state.clauses.len();

                    // max_clauses: every Add (integer comparison, free)
                    if self.config.max_clauses > 0 && num_clauses >= self.config.max_clauses {
                        return Some(ProofResult::ResourceLimit(
                            self.state.build_proof_steps(),
                            self.state.clauses.clone(),
                        ));
                    }

                    // timeout + memory: every 100th Add (amortize syscall cost)
                    if num_clauses % 100 == 0 {
                        if let Some(start) = self.start_time {
                            if start.elapsed() > self.config.timeout {
                                return Some(ProofResult::ResourceLimit(
                                    self.state.build_proof_steps(),
                                    self.state.clauses.clone(),
                                ));
                            }
                        }

                        if let Some(limit_mb) = self.config.memory_limit {
                            if let Some(rss) = crate::config::process_memory_mb() {
                                if rss >= limit_mb {
                                    return Some(ProofResult::ResourceLimit(
                                        self.state.build_proof_steps(),
                                        self.state.clauses.clone(),
                                    ));
                                }
                            }
                        }
                    }
                }
            }
            StateChange::Simplify(clause_idx, ref replacement, ref _rule_name, ref _premises) => {
                let clause_idx = *clause_idx;
                // Remove the simplified clause from whichever set contains it
                if self.state.new.last() == Some(&clause_idx) {
                    self.state.new.pop();
                } else if self.state.unprocessed.shift_remove(&clause_idx) {
                    let clause = &self.state.clauses[clause_idx];
                    self.index_registry.on_delete(clause_idx, clause);
                } else if self.state.processed.shift_remove(&clause_idx) {
                    let clause = &self.state.clauses[clause_idx];
                    self.index_registry.on_delete(clause_idx, clause);
                }

                // If there's a replacement clause, add it to N
                let logged_change = if let Some(repl) = replacement {
                    if repl.literals.len() > self.config.max_clause_size {
                        self.state.event_log.push(change);
                        return None;
                    }

                    let new_idx = self.state.clauses.len();
                    let mut clause_with_id = repl.clone();
                    clause_with_id.id = Some(new_idx);
                    clause_with_id.age = self.state.current_iteration;
                    clause_with_id.role = crate::logic::ClauseRole::Derived;

                    let mut oriented = repl.clone();
                    self.clause_manager.orient_equalities(&mut oriented);
                    self.index_registry.on_add(new_idx, &oriented);

                    let is_empty = clause_with_id.is_empty();

                    self.state.clauses.push(clause_with_id.clone());
                    self.state.new.push(new_idx);

                    if let Some(p) = self.profile.as_mut() {
                        p.clauses_added += 1;
                    }

                    // Log with the clause that has an id assigned
                    let StateChange::Simplify(idx, _, rule_name, premises) = change else {
                        unreachable!()
                    };
                    let logged = StateChange::Simplify(idx, Some(clause_with_id), rule_name, premises);
                    self.state.event_log.push(logged);

                    // Empty replacement clause → proof found
                    if is_empty {
                        return Some(ProofResult::Proof(self.state.extract_proof(new_idx)));
                    }

                    // Check limits after adding replacement
                    let num_clauses = self.state.clauses.len();
                    if self.config.max_clauses > 0 && num_clauses >= self.config.max_clauses {
                        return Some(ProofResult::ResourceLimit(
                            self.state.build_proof_steps(),
                            self.state.clauses.clone(),
                        ));
                    }
                    if num_clauses % 100 == 0 {
                        if let Some(start) = self.start_time {
                            if start.elapsed() > self.config.timeout {
                                return Some(ProofResult::ResourceLimit(
                                    self.state.build_proof_steps(),
                                    self.state.clauses.clone(),
                                ));
                            }
                        }
                        if let Some(limit_mb) = self.config.memory_limit {
                            if let Some(rss) = crate::config::process_memory_mb() {
                                if rss >= limit_mb {
                                    return Some(ProofResult::ResourceLimit(
                                        self.state.build_proof_steps(),
                                        self.state.clauses.clone(),
                                    ));
                                }
                            }
                        }
                    }

                    return None;
                } else {
                    change
                };
                self.state.event_log.push(logged_change);
            }
            StateChange::Transfer(clause_idx) => {
                let clause_idx = *clause_idx;
                // N → U
                if self.state.new.last() == Some(&clause_idx) {
                    self.state.new.pop();
                }
                self.state.unprocessed.insert(clause_idx);
                self.index_registry.on_transfer(clause_idx, &self.state.clauses[clause_idx]);
                self.state.event_log.push(change);
            }
            StateChange::Activate(clause_idx) => {
                let clause_idx = *clause_idx;
                // U → P (selector already removed from U)
                self.state.unprocessed.shift_remove(&clause_idx);
                self.state.processed.insert(clause_idx);
                self.index_registry.on_activate(clause_idx, &self.state.clauses[clause_idx]);
                self.state.event_log.push(change);

                // Increment iteration count at activation (given clause selection)
                self.state.current_iteration += 1;

                // Check max_iterations
                if self.config.max_iterations > 0
                    && self.state.current_iteration >= self.config.max_iterations
                {
                    return Some(ProofResult::ResourceLimit(
                        self.state.build_proof_steps(),
                        self.state.clauses.clone(),
                    ));
                }
            }
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
