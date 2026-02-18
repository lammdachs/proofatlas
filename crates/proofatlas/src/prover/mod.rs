//! Prover: orchestrates saturation-based theorem proving.
//!
//! The `Prover` struct combines clause management, inference rules, and
//! selection strategies to implement the given-clause saturation algorithm.
//!
//! Use `prove()` to run to completion, or `step()` for incremental execution.

pub mod profile;

use crate::logic::clause_manager::ClauseManager;
use crate::logic::{CNFFormula, Clause, Interner};
use crate::state::{
    GeneratingInference,
    ProofResult, SaturationState, SimplifyingInference,
    StateChange, ProofStep,
};
use crate::config::{LiteralSelectionStrategy, ProverConfig};
use self::profile::SaturationProfile;
use crate::index::{IndexKind, IndexRegistry, SelectedLiteralIndex};
use crate::simplifying::{TautologyRule, SubsumptionRule, DemodulationRule};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use crate::generating::{
    ResolutionRule, SuperpositionRule, FactoringRule,
    EqualityResolutionRule, EqualityFactoringRule,
};
use crate::selection::{
    SelectAll, SelectMaximal, SelectNegMaxWeightOrMaximal,
    SelectUniqueMaximalOrNegOrMaximal,
};
use crate::selection::clause::ProverSink;
use crate::logic::time_compat::Instant;
use std::collections::HashSet;

/// Per-problem saturation engine.
///
/// Combines clause management, simplifying/generating inference rules,
/// index registry, and clause selection into the given-clause algorithm.
pub struct Prover {
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
    /// Clause selection sink (signal-based interface)
    sink: Box<dyn ProverSink>,
    /// Profiling data (None if profiling disabled)
    profile: Option<SaturationProfile>,
    /// Start time of the proof search
    start_time: Option<Instant>,
    /// Initial clauses to be added during init()
    initial_clauses: Vec<Clause>,
    /// Cancellation flag — set to `true` to stop the saturation loop.
    pub cancel: Arc<AtomicBool>,
}

impl Prover {
    /// Create a new Prover from initial clauses.
    ///
    /// # Arguments
    /// * `initial_clauses` - The initial clause set
    /// * `config` - Prover configuration
    /// * `sink` - Clause selection sink (signal-based interface)
    /// * `interner` - Symbol interner for resolving symbol names
    pub fn new(
        initial_clauses: Vec<Clause>,
        config: ProverConfig,
        sink: Box<dyn ProverSink>,
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
            IndexKind::DiscriminationTree,
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
        let mut clause_manager = ClauseManager::new(interner, literal_selector);
        clause_manager.memory_limit = config.memory_limit;
        clause_manager.baseline_rss_mb = crate::config::process_memory_mb().unwrap_or(0);

        // Reset sink state
        let mut sink = sink;
        sink.reset();

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

        let cancel = Arc::new(AtomicBool::new(false));
        clause_manager.cancel = cancel.clone();

        Prover {
            config,
            clause_manager,
            state,
            simplifying_inferences,
            generating_inferences,
            index_registry,
            sink,
            profile,
            start_time: None,
            initial_clauses,
            cancel,
        }
    }

    /// Run saturation to completion.
    ///
    /// After completion, all data (clauses, event log, profile, interner) remains
    /// accessible on the prover via accessor methods.
    pub fn prove(&mut self) -> ProofResult {
        let start_time = Instant::now();
        self.start_time = Some(start_time);
        self.clause_manager.start_time = Some(start_time);
        self.clause_manager.timeout = self.config.timeout;

        let t_init = self.profile.as_ref().map(|_| Instant::now());
        if let Some(result) = self.init() {
            if let (Some(p), Some(t)) = (self.profile.as_mut(), t_init) {
                p.init_time = t.elapsed();
                p.total_time = start_time.elapsed();
            }
            return result;
        }
        if let (Some(p), Some(t)) = (self.profile.as_mut(), t_init) {
            p.init_time = t.elapsed();
        }

        let result = loop {
            if let Some(result) = self.step() {
                break result;
            }
        };

        // Finalize profile
        if let Some(p) = self.profile.as_mut() {
            p.total_time = start_time.elapsed();
            p.selector_name = self.sink.name().to_string();
            if let Some(stats) = self.sink.stats() {
                p.selector_cache_hits = stats.cache_hits;
                p.selector_cache_misses = stats.cache_misses;
                p.selector_embed_time = stats.embed_time;
                p.selector_score_time = stats.score_time;
            }
        }

        result
    }

    // =========================================================================
    // Public accessors (available after prove())
    // =========================================================================

    /// Get the interner (symbol name resolution).
    pub fn interner(&self) -> &Interner { &self.clause_manager.interner }

    /// Get the event log (all state changes during saturation).
    pub fn event_log(&self) -> &[StateChange] { &self.state.event_log }

    /// Get all clauses generated during saturation.
    pub fn clauses(&self) -> &[Arc<Clause>] { &self.state.clauses }

    /// Get profiling data (None if profiling was not enabled).
    pub fn profile(&self) -> Option<&SaturationProfile> { self.profile.as_ref() }

    /// Extract proof steps by backward traversal from the given clause index.
    pub fn extract_proof(&self, clause_idx: usize) -> Vec<ProofStep> {
        self.state.extract_proof(clause_idx)
    }

    /// Verify every step in a proof is correct.
    ///
    /// Extracts the proof from the given empty clause index, then dispatches
    /// each step to the appropriate rule's `verify()` method.
    pub fn verify_proof(&self, empty_clause_idx: usize) -> Result<(), crate::state::VerificationError> {
        use crate::state::VerificationError;

        let steps = self.state.extract_proof(empty_clause_idx);

        for (step_num, step) in steps.iter().enumerate() {
            // Check that all premise clause indices are valid
            for pos in &step.premises {
                if pos.clause >= self.state.clauses.len() {
                    return Err(VerificationError::InvalidPremise {
                        step_idx: step_num,
                        premise_idx: pos.clause,
                    });
                }
            }

            match step.rule_name.as_str() {
                "Input" => {
                    // Input steps: verify the clause index is within the initial clause count
                    if step.clause_idx >= self.state.initial_clause_count {
                        return Err(VerificationError::InputNotFound {
                            step_idx: step_num,
                            clause_idx: step.clause_idx,
                        });
                    }
                }
                rule_name => {
                    // Try generating rules first
                    let gen_result = self.generating_inferences.iter()
                        .find(|r| r.name() == rule_name)
                        .map(|r| r.verify(&step.conclusion, &step.premises, &self.state, &self.clause_manager));

                    if let Some(result) = gen_result {
                        if let Err(mut e) = result {
                            // Patch in the correct step index
                            match &mut e {
                                VerificationError::InvalidConclusion { step_idx, .. } => *step_idx = step_num,
                                VerificationError::InvalidPremise { step_idx, .. } => *step_idx = step_num,
                                VerificationError::UnknownRule { step_idx, .. } => *step_idx = step_num,
                                VerificationError::InputNotFound { step_idx, .. } => *step_idx = step_num,
                            }
                            return Err(e);
                        }
                        continue;
                    }

                    // Try simplifying rules
                    let simp_result = self.simplifying_inferences.iter()
                        .find(|r| r.name() == rule_name)
                        .map(|r| r.verify(
                            step.clause_idx,
                            // For simplifying inferences, the "conclusion" in the proof step
                            // is the replacement clause (if any). We need to determine if
                            // this is a deletion or a replacement.
                            if step.conclusion.is_empty() && rule_name != "EqualityResolution" {
                                None
                            } else {
                                Some(step.conclusion.as_ref())
                            },
                            &step.premises,
                            &self.state,
                            &self.clause_manager,
                        ));

                    if let Some(result) = simp_result {
                        if let Err(mut e) = result {
                            match &mut e {
                                VerificationError::InvalidConclusion { step_idx, .. } => *step_idx = step_num,
                                VerificationError::InvalidPremise { step_idx, .. } => *step_idx = step_num,
                                VerificationError::UnknownRule { step_idx, .. } => *step_idx = step_num,
                                VerificationError::InputNotFound { step_idx, .. } => *step_idx = step_num,
                            }
                            return Err(e);
                        }
                        continue;
                    }

                    return Err(VerificationError::UnknownRule {
                        step_idx: step_num,
                        rule: rule_name.to_string(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Execute one step of the saturation loop.
    ///
    /// Returns `Some(result)` if the proof search is complete, `None` to continue.
    /// Each step processes all new clauses, then selects one given clause and
    /// generates inferences with it.
    pub fn step(&mut self) -> Option<ProofResult> {
        let now = *self.start_time.get_or_insert_with(Instant::now);
        if self.clause_manager.start_time.is_none() {
            self.clause_manager.start_time = Some(now);
            self.clause_manager.timeout = self.config.timeout;
        }

        // === Step 1: Process new clauses ===
        let t_process_new = self.profile.as_ref().map(|_| Instant::now());
        'simplify: while let Some(&clause_idx) = self.state.new.last() {
            // Check timeout/cancel between processing each new clause
            if self.clause_manager.is_cancelled() {
                return Some(ProofResult::ResourceLimit);
            }

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

        if let (Some(p), Some(t)) = (self.profile.as_mut(), t_process_new) {
            p.process_new_time += t.elapsed();
        }

        // === Step 2: Check saturation ===
        if self.state.unprocessed.is_empty() {
            return Some(ProofResult::Saturated);
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

        // === Step 2b: Check timeout/cancel before selection ===
        // When using a remote scoring server, select() can block for seconds
        // waiting on server mutex contention.  Check wall-clock timeout here
        // so the prover exits promptly after a long-blocking select().
        if self.cancel.load(Ordering::Relaxed) {
            return Some(ProofResult::ResourceLimit);
        }
        if let Some(start) = self.start_time {
            if start.elapsed() > self.config.timeout {
                return Some(ProofResult::ResourceLimit);
            }
        }

        // === Step 3: Select given clause ===
        let t0 = self.profile.as_ref().map(|_| Instant::now());
        let given_idx = match self.select_given_clause() {
            Some(idx) => idx,
            None if self.state.unprocessed.is_empty() => {
                return Some(ProofResult::Saturated);
            }
            None => {
                // Selector returned None with clauses still available — selector error
                // (e.g., scoring server died). Report as resource limit.
                return Some(ProofResult::ResourceLimit);
            }
        };
        if let (Some(p), Some(t)) = (self.profile.as_mut(), t0) {
            p.select_given_time += t.elapsed();
        }

        // === Step 4: Activate given clause (transfer from U to P) ===
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

        // Add new inferences (orientation handled by apply_change)
        let t0 = self.profile.as_ref().map(|_| Instant::now());
        for change in new_changes {
            if let Some(result) = self.apply_change(change) {
                return Some(result);
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
        for mut clause in initial_clauses {
            self.clause_manager.orient_equalities(&mut clause);
            if let Some(result) = self.apply_change(StateChange::Add(Arc::new(clause), "Input".into(), vec![])) {
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
    /// Returns `Some(ProofResult::ResourceLimit)` if a limit is exceeded.
    fn apply_change(&mut self, change: StateChange) -> Option<ProofResult> {
        match change {
            StateChange::Add(mut arc_clause, rule_name, premises) => {
                if arc_clause.literals.len() > self.config.max_clause_size {
                    return None;
                }

                let new_idx = self.state.clauses.len();

                // Arc has refcount=1 (just created by caller), so get_mut succeeds.
                // Clause is already oriented at creation — only set metadata here.
                {
                    let clause = Arc::get_mut(&mut arc_clause)
                        .expect("Arc refcount must be 1 in apply_change/Add");
                    clause.id = Some(new_idx);
                    clause.age = self.state.current_iteration;
                    clause.role = crate::logic::ClauseRole::Derived;
                    clause.derivation_rule = Clause::rule_name_to_id(&rule_name);
                }

                self.index_registry.on_add(new_idx, &arc_clause);
                let is_empty = arc_clause.is_empty();

                self.state.clauses.push(Arc::clone(&arc_clause));
                self.state.new.push(new_idx);
                self.state.event_log.push(StateChange::Add(arc_clause, rule_name, premises));

                // Empty clause → proof found
                if is_empty {
                    return Some(ProofResult::Proof { empty_clause_idx: new_idx });
                }

                if let Some(p) = self.profile.as_mut() {
                    p.clauses_added += 1;
                }

                // Check limits after every Add
                let num_clauses = self.state.clauses.len();

                // max_clauses: every Add (integer comparison, free)
                if self.config.max_clauses > 0 && num_clauses >= self.config.max_clauses {
                    return Some(ProofResult::ResourceLimit);
                }

                // memory + timeout + cancel: every 10th Add (amortize /proc read)
                if num_clauses % 10 == 0 {
                    if let Some(limit_mb) = self.config.memory_limit {
                        if let Some(rss) = crate::config::process_memory_mb() {
                            let baseline = self.clause_manager.baseline_rss_mb;
                            if rss.saturating_sub(baseline) >= limit_mb {
                                return Some(ProofResult::ResourceLimit);
                            }
                        }
                    }

                    if self.cancel.load(Ordering::Relaxed) {
                        return Some(ProofResult::ResourceLimit);
                    }

                    if let Some(start) = self.start_time {
                        if start.elapsed() > self.config.timeout {
                            return Some(ProofResult::ResourceLimit);
                        }
                    }
                }
            }
            StateChange::Simplify(clause_idx, replacement, rule_name, premises) => {
                // Remove the simplified clause from whichever set contains it
                if self.state.new.last() == Some(&clause_idx) {
                    self.state.new.pop();
                } else if self.state.unprocessed.shift_remove(&clause_idx) {
                    let clause = &self.state.clauses[clause_idx];
                    self.index_registry.on_delete(clause_idx, clause);
                    self.sink.on_simplify(clause_idx);
                } else if self.state.processed.shift_remove(&clause_idx) {
                    let clause = &self.state.clauses[clause_idx];
                    self.index_registry.on_delete(clause_idx, clause);
                    self.sink.on_simplify(clause_idx);
                }

                // If there's a replacement clause, add it to N
                if let Some(mut repl) = replacement {
                    if repl.literals.len() > self.config.max_clause_size {
                        self.state.event_log.push(StateChange::Simplify(clause_idx, Some(repl), rule_name, premises));
                        return None;
                    }

                    let new_idx = self.state.clauses.len();

                    // Arc has refcount=1, so get_mut succeeds.
                    // Clause is already oriented at creation — only set metadata here.
                    {
                        let clause = Arc::get_mut(&mut repl)
                            .expect("Arc refcount must be 1 in apply_change/Simplify");
                        clause.id = Some(new_idx);
                        clause.age = self.state.current_iteration;
                        clause.role = crate::logic::ClauseRole::Derived;
                        clause.derivation_rule = Clause::rule_name_to_id(&rule_name);
                    }

                    self.index_registry.on_add(new_idx, &repl);
                    let is_empty = repl.is_empty();

                    self.state.clauses.push(Arc::clone(&repl));
                    self.state.new.push(new_idx);

                    if let Some(p) = self.profile.as_mut() {
                        p.clauses_added += 1;
                    }

                    self.state.event_log.push(StateChange::Simplify(clause_idx, Some(repl), rule_name, premises));

                    // Empty replacement clause → proof found
                    if is_empty {
                        return Some(ProofResult::Proof { empty_clause_idx: new_idx });
                    }

                    // Check limits after adding replacement
                    let num_clauses = self.state.clauses.len();
                    if self.config.max_clauses > 0 && num_clauses >= self.config.max_clauses {
                        return Some(ProofResult::ResourceLimit);
                    }
                    if num_clauses % 10 == 0 {
                        if let Some(limit_mb) = self.config.memory_limit {
                            if let Some(rss) = crate::config::process_memory_mb() {
                                let baseline = self.clause_manager.baseline_rss_mb;
                                if rss.saturating_sub(baseline) >= limit_mb {
                                    return Some(ProofResult::ResourceLimit);
                                }
                            }
                        }
                        if self.cancel.load(Ordering::Relaxed) {
                            return Some(ProofResult::ResourceLimit);
                        }
                        if let Some(start) = self.start_time {
                            if start.elapsed() > self.config.timeout {
                                return Some(ProofResult::ResourceLimit);
                            }
                        }
                    }

                    return None;
                } else {
                    self.state.event_log.push(StateChange::Simplify(clause_idx, None, rule_name, premises));
                }
            }
            StateChange::Transfer(clause_idx) => {
                // N → U
                if self.state.new.last() == Some(&clause_idx) {
                    self.state.new.pop();
                }
                self.state.unprocessed.insert(clause_idx);
                self.index_registry.on_transfer(clause_idx, &self.state.clauses[clause_idx]);
                self.sink.on_transfer(clause_idx, &self.state.clauses[clause_idx]);
                self.state.event_log.push(StateChange::Transfer(clause_idx));
            }
            StateChange::Activate(clause_idx) => {
                // U → P
                self.state.unprocessed.shift_remove(&clause_idx);
                self.state.processed.insert(clause_idx);
                self.index_registry.on_activate(clause_idx, &self.state.clauses[clause_idx]);
                self.sink.on_activate(clause_idx);
                self.state.event_log.push(StateChange::Activate(clause_idx));

                // Increment iteration count at activation (given clause selection)
                self.state.current_iteration += 1;

                // Check max_iterations
                if self.config.max_iterations > 0
                    && self.state.current_iteration >= self.config.max_iterations
                {
                    return Some(ProofResult::ResourceLimit);
                }
            }
        }
        None
    }

    /// Select the next given clause using the configured sink
    fn select_given_clause(&mut self) -> Option<usize> {
        self.sink.select()
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

/// Run saturation on a CNF formula, returning the prover with all state intact.
pub fn saturate(
    formula: CNFFormula,
    config: ProverConfig,
    sink: Box<dyn ProverSink>,
    interner: Interner,
) -> (ProofResult, Prover) {
    let mut prover = Prover::new(formula.clauses, config, sink, interner);
    let result = prover.prove();
    (result, prover)
}
