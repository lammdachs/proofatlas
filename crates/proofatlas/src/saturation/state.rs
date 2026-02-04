//! Main saturation state and algorithm
//!
//! This module implements a three-set saturation loop based on the framework
//! described by Waldmann et al. in "A Comprehensive Framework for Saturation
//! Theorem Proving" (IJCAR 2020), specifically Example 63.
//!
//! ## Clause Sets
//!
//! The prover maintains three clause sets:
//!
//! - **New (N)**: Freshly derived clauses awaiting forward simplification
//! - **Unprocessed (U)**: Clauses that have survived simplification, awaiting selection
//! - **Processed (P)**: Clauses that have been selected and used for generating inferences
//!
//! ## Algorithm
//!
//! Each iteration of the main loop:
//!
//! 1. **Process New Clauses**:
//!    - For each clause C in N:
//!      - **Forward Simplification**: Apply simplification rules (tautology, demodulation, subsumption)
//!      - If C survives:
//!        - **Backward Simplification**: Apply backward simplification rules
//!        - **Transfer**: Move C from N to U
//!
//! 2. **Select Given Clause**: Choose a clause G from U using the clause selector
//!
//! 3. **Generate Inferences**: Apply generating inference rules between G and clauses in P
//!
//! 4. **Activate**: Move G from U to P
//!
//! ## Polymorphic Rule Architecture
//!
//! The saturation loop uses registered rules implementing two traits:
//! - **SimplificationRule**: Tautology deletion, subsumption, demodulation
//! - **GeneratingInferenceRule**: Resolution, superposition, factoring, etc.
//!
//! All rules return `Vec<ProofStateChange>` for atomic state modifications.

use super::profile::SaturationProfile;
use super::rule::{
    ClauseNotification, ClauseView, DemodulationRule, EqualityFactoringRule, EqualityResolutionRule,
    FactoringRule, GeneratingInferenceRule, ProofStateChange, ResolutionRule, SimplificationRule,
    SubsumptionRule, SuperpositionRule, TautologyRule,
};
use super::trace::{
    BackwardSimplification, ClauseSimplification, ForwardSimplification, GeneratingInference,
    SaturationStep, SaturationTrace, SimplificationOutcome,
};
use crate::fol::Clause;
use crate::inference::{Derivation, InferenceResult, Proof, ProofStep};
use crate::parser::orient_equalities::orient_clause_equalities;
use crate::selection::{
    ClauseSelector, LiteralSelector, SelectAll, SelectMaximal, SelectNegMaxWeightOrMaximal,
    SelectUniqueMaximalOrNegOrMaximal,
};
use crate::time_compat::Instant;
use std::collections::{HashSet, VecDeque};
use std::time::Duration;

/// Configuration for the saturation loop
#[derive(Debug, Clone)]
pub struct SaturationConfig {
    pub max_clauses: usize,
    pub max_iterations: usize,
    pub max_clause_size: usize,
    pub timeout: Duration,
    pub literal_selection: LiteralSelectionStrategy,
    /// Memory limit for clause storage in MB (directly comparable across provers)
    pub max_clause_memory_mb: Option<usize>,
    /// Enable structured profiling (zero overhead when false)
    pub enable_profiling: bool,
}

/// Literal selection strategies (numbers match Vampire's --selection option)
///
/// From Hoder et al. "Selecting the selection" (2016):
/// - Sel0: Select all literals
/// - Sel20: Select all maximal literals
/// - Sel21: Select unique maximal, else max-weight negative, else all maximal
/// - Sel22: Select max-weight negative literal, else all maximal
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiteralSelectionStrategy {
    /// Selection 0: Select all literals (no selection)
    Sel0,
    /// Selection 20: Select all maximal literals
    Sel20,
    /// Selection 21: Unique maximal, else max-weight negative, else all maximal
    Sel21,
    /// Selection 22: Max-weight negative literal, else all maximal
    Sel22,
}

impl Default for SaturationConfig {
    fn default() -> Self {
        SaturationConfig {
            max_clauses: 0,      // 0 means no limit
            max_iterations: 0,   // 0 means no limit
            max_clause_size: 100,
            timeout: Duration::from_secs(60),
            literal_selection: LiteralSelectionStrategy::Sel21,
            max_clause_memory_mb: None,
            enable_profiling: false,
        }
    }
}

/// Result of saturation
#[derive(Debug, Clone)]
pub enum SaturationResult {
    /// Empty clause derived - proof found
    Proof(Proof),
    /// Saturated without finding empty clause
    Saturated(Vec<ProofStep>, Vec<Clause>),
    /// Resource limit reached
    ResourceLimit(Vec<ProofStep>, Vec<Clause>),
    /// Timeout reached
    Timeout(Vec<ProofStep>, Vec<Clause>),
}

impl SaturationResult {
    /// Convert to JSON representation
    pub fn to_json(&self, time_seconds: f64) -> crate::json::SaturationResultJson {
        use crate::json::SaturationResultJson;

        match self {
            SaturationResult::Proof(proof) => SaturationResultJson::Proof {
                proof: proof.into(),
                time_seconds,
            },
            SaturationResult::Saturated(steps, clauses) => SaturationResultJson::Saturated {
                final_clauses: clauses.iter().map(|c| c.into()).collect(),
                proof_steps: steps.iter().map(|s| s.into()).collect(),
                time_seconds,
            },
            SaturationResult::ResourceLimit(steps, clauses) => {
                SaturationResultJson::ResourceLimit {
                    reason: "Clause or iteration limit exceeded".to_string(),
                    final_clauses: clauses.iter().map(|c| c.into()).collect(),
                    proof_steps: steps.iter().map(|s| s.into()).collect(),
                    time_seconds,
                }
            }
            SaturationResult::Timeout(steps, clauses) => SaturationResultJson::Timeout {
                final_clauses: clauses.iter().map(|c| c.into()).collect(),
                proof_steps: steps.iter().map(|s| s.into()).collect(),
                time_seconds,
            },
        }
    }
}

/// Main saturation state implementing the three-set given-clause algorithm.
///
/// See module documentation for algorithm details.
pub struct SaturationState {
    /// Storage for all clauses, indexed by clause ID
    clauses: Vec<Clause>,
    /// Derivations parallel to clauses (how each clause was derived)
    derivations: Vec<Option<Derivation>>,
    /// Set P: Processed/active clauses (used for generating inferences)
    processed: HashSet<usize>,
    /// Set U: Unprocessed/passive clauses (awaiting selection)
    unprocessed: VecDeque<usize>,
    /// Set N: New clauses (awaiting forward simplification)
    new: VecDeque<usize>,
    /// Configuration
    config: SaturationConfig,
    /// Clause selector
    clause_selector: Box<dyn ClauseSelector>,
    /// Literal selector
    literal_selector: Box<dyn LiteralSelector>,
    /// Tracked clause memory usage in bytes
    clause_memory_bytes: usize,
    /// Current iteration (used for clause age)
    current_iteration: usize,
    /// Number of initial input clauses
    initial_clause_count: usize,
    /// Completed saturation iterations (for trace)
    iterations: Vec<SaturationStep>,
    /// Per-iteration accumulators (reset each iteration)
    current_simplifications: Vec<ClauseSimplification>,
    current_generation: Vec<GeneratingInference>,

    // === Polymorphic Rules ===
    /// Simplification rules (tautology, demodulation, subsumption)
    simplification_rules: Vec<Box<dyn SimplificationRule>>,
    /// Generating inference rules (resolution, superposition, factoring, etc.)
    generating_rules: Vec<Box<dyn GeneratingInferenceRule>>,

    // === Event Log ===
    /// Raw event log capturing all state changes
    event_log: Vec<ProofStateChange>,
}

impl SaturationState {
    /// Create new saturation state from initial clauses
    ///
    /// # Arguments
    /// * `initial_clauses` - The initial clause set
    /// * `config` - Saturation configuration
    /// * `clause_selector` - Clause selection strategy
    pub fn new(
        initial_clauses: Vec<Clause>,
        config: SaturationConfig,
        clause_selector: Box<dyn ClauseSelector>,
    ) -> Self {
        let initial_clause_count = initial_clauses.len();

        // Initialize simplification rules
        let mut simplification_rules: Vec<Box<dyn SimplificationRule>> = vec![
            Box::new(TautologyRule::new()),
            Box::new(DemodulationRule::new()),
            Box::new(SubsumptionRule::new()),
        ];

        // Initialize all simplification rules with input clauses
        for rule in &mut simplification_rules {
            rule.initialize(&initial_clauses);
        }

        let mut clauses = Vec::new();
        let mut derivations = Vec::new();
        let mut new = VecDeque::new();
        let mut clause_memory_bytes = 0usize;

        // Add initial clauses with IDs
        let mut clause_idx = 0;
        for mut clause in initial_clauses.into_iter() {
            // Orient equalities before adding
            let mut oriented = clause.clone();
            orient_clause_equalities(&mut oriented);

            clause.id = Some(clause_idx);

            // Track clause memory
            clause_memory_bytes += clause.memory_bytes();

            // Notify rules about pending clause
            for rule in &mut simplification_rules {
                rule.on_clause_pending(clause_idx, &oriented);
            }

            // Record derivation for input clause
            derivations.push(Some(Derivation::input()));

            clauses.push(clause);
            new.push_back(clause_idx);
            clause_idx += 1;
        }

        // Create literal selector based on configuration
        let literal_selector: Box<dyn LiteralSelector> = match config.literal_selection {
            LiteralSelectionStrategy::Sel0 => Box::new(SelectAll),
            LiteralSelectionStrategy::Sel20 => Box::new(SelectMaximal::new()),
            LiteralSelectionStrategy::Sel21 => Box::new(SelectUniqueMaximalOrNegOrMaximal::new()),
            LiteralSelectionStrategy::Sel22 => Box::new(SelectNegMaxWeightOrMaximal::new()),
        };

        // Reset clause selector state (clear any caches from previous problems)
        let mut clause_selector = clause_selector;
        clause_selector.reset();

        // Initialize generating rules
        let generating_rules: Vec<Box<dyn GeneratingInferenceRule>> = vec![
            Box::new(FactoringRule::new()),
            Box::new(EqualityResolutionRule::new()),
            Box::new(EqualityFactoringRule::new()),
            Box::new(ResolutionRule::new()),
            Box::new(SuperpositionRule::new()),
        ];

        // Build initial event log with input clauses
        let mut event_log = Vec::new();
        for idx in 0..clause_idx {
            event_log.push(ProofStateChange::AddN {
                clause: clauses[idx].clone(),
                derivation: Derivation::input(),
            });
        }

        SaturationState {
            clauses,
            derivations,
            processed: HashSet::new(),
            unprocessed: VecDeque::new(),
            new,
            config,
            clause_selector,
            literal_selector,
            clause_memory_bytes,
            current_iteration: 0,
            initial_clause_count,
            iterations: Vec::new(),
            current_simplifications: Vec::new(),
            current_generation: Vec::new(),
            simplification_rules,
            generating_rules,
            event_log,
        }
    }

    /// Set the literal selector
    pub fn set_literal_selector(&mut self, selector: Box<dyn LiteralSelector>) {
        self.literal_selector = selector;
    }

    /// Set the clause selector
    pub fn set_clause_selector(&mut self, selector: Box<dyn ClauseSelector>) {
        self.clause_selector = selector;
    }

    /// Get all clauses (for JSON export)
    pub fn get_clauses(&self) -> &[Clause] {
        &self.clauses
    }

    /// Extract a proof by backward traversal from the empty clause.
    pub fn extract_proof(&self, empty_clause_idx: usize) -> Proof {
        let mut proof_clause_indices = Vec::new();
        let mut visited = HashSet::new();
        let mut to_visit = vec![empty_clause_idx];

        while let Some(idx) = to_visit.pop() {
            if !visited.insert(idx) {
                continue;
            }
            proof_clause_indices.push(idx);
            if let Some(Some(derivation)) = self.derivations.get(idx) {
                to_visit.extend(derivation.premises());
            }
        }

        // Sort topologically (by index, since parents always have lower indices)
        proof_clause_indices.sort();

        let steps = proof_clause_indices
            .iter()
            .map(|&idx| ProofStep {
                clause_idx: idx,
                derivation: self.derivations[idx].clone().unwrap(),
                conclusion: self.clauses[idx].clone(),
            })
            .collect();

        Proof {
            steps,
            empty_clause_idx,
            all_clauses: self.clauses.clone(),
        }
    }

    /// Convert the saturation state into a structured trace.
    pub fn into_trace(self) -> SaturationTrace {
        SaturationTrace {
            clauses: self.clauses.iter().map(|c| c.to_string()).collect(),
            initial_clause_count: self.initial_clause_count,
            iterations: self.iterations,
        }
    }

    /// Get the derivations (for external proof extraction)
    pub fn derivations(&self) -> &[Option<Derivation>] {
        &self.derivations
    }

    /// Get the iterations (for external trace building)
    pub fn iterations(&self) -> &[SaturationStep] {
        &self.iterations
    }

    /// Get the event log (raw state changes)
    pub fn event_log(&self) -> &[ProofStateChange] {
        &self.event_log
    }

    /// Flush accumulated simplification/generation events into a completed iteration.
    fn flush_iteration(&mut self, given_clause: Option<usize>) {
        self.iterations.push(SaturationStep {
            simplifications: std::mem::take(&mut self.current_simplifications),
            given_clause,
            generating_inferences: std::mem::take(&mut self.current_generation),
        });
    }

    /// Build proof steps from derivations (for backward-compatible SaturationResult).
    fn build_proof_steps(&self) -> Vec<ProofStep> {
        self.derivations
            .iter()
            .enumerate()
            .filter_map(|(idx, d)| {
                d.as_ref().map(|derivation| ProofStep {
                    clause_idx: idx,
                    derivation: derivation.clone(),
                    conclusion: self.clauses[idx].clone(),
                })
            })
            .collect()
    }

    /// Build clause views for U and P
    fn build_views(&self) -> (Vec<usize>, Vec<usize>) {
        let u_indices: Vec<usize> = self.unprocessed.iter().copied().collect();
        let p_indices: Vec<usize> = self.processed.iter().copied().collect();
        (u_indices, p_indices)
    }

    /// Notify all rules about a clause being added to U
    fn notify_add_unprocessed(&mut self, clause_idx: usize) {
        let clause = &self.clauses[clause_idx];
        let notif = ClauseNotification::Added { clause_idx, clause };
        for rule in &mut self.simplification_rules {
            rule.notify_unprocessed(notif.clone());
        }
    }

    /// Notify all rules about a clause being removed from U
    fn notify_remove_unprocessed(&mut self, clause_idx: usize) {
        let clause = &self.clauses[clause_idx];
        let notif = ClauseNotification::Removed { clause_idx, clause };
        for rule in &mut self.simplification_rules {
            rule.notify_unprocessed(notif.clone());
        }
    }

    /// Notify all rules about a clause being added to P
    fn notify_add_processed(&mut self, clause_idx: usize) {
        let clause = &self.clauses[clause_idx];
        let notif = ClauseNotification::Added { clause_idx, clause };
        for rule in &mut self.simplification_rules {
            rule.notify_processed(notif.clone());
        }
        for rule in &mut self.generating_rules {
            rule.notify(notif.clone());
        }
    }

    /// Notify all rules about a clause being removed from P
    fn notify_remove_processed(&mut self, clause_idx: usize) {
        let clause = &self.clauses[clause_idx];
        let notif = ClauseNotification::Removed { clause_idx, clause };
        for rule in &mut self.simplification_rules {
            rule.notify_processed(notif.clone());
        }
        for rule in &mut self.generating_rules {
            rule.notify(notif.clone());
        }
    }

    /// Apply a single ProofStateChange, update internal state, and record to event log
    fn apply_change(&mut self, change: ProofStateChange, profile: &mut Option<SaturationProfile>) {
        match &change {
            ProofStateChange::AddN { clause, derivation } => {
                // Check clause size limit
                if clause.literals.len() > self.config.max_clause_size {
                    return;
                }

                let new_idx = self.clauses.len();
                let mut clause_with_id = clause.clone();
                clause_with_id.id = Some(new_idx);
                clause_with_id.age = self.current_iteration;
                clause_with_id.role = crate::fol::ClauseRole::Derived;

                // Notify rules about pending clause
                let mut oriented = clause.clone();
                orient_clause_equalities(&mut oriented);
                for rule in &mut self.simplification_rules {
                    rule.on_clause_pending(new_idx, &oriented);
                }

                self.clause_memory_bytes += clause_with_id.memory_bytes();
                self.clauses.push(clause_with_id.clone());
                self.derivations.push(Some(derivation.clone()));
                self.new.push_back(new_idx);

                // Record event with the actual stored clause
                self.event_log.push(ProofStateChange::AddN {
                    clause: clause_with_id,
                    derivation: derivation.clone(),
                });

                if let Some(p) = profile.as_mut() {
                    p.clauses_added += 1;
                }
            }
            ProofStateChange::RemoveN { clause_idx: _, rule_name: _ } => {
                // Clause is just not transferred - no action needed for state
                // Record event
                self.event_log.push(change);
            }
            ProofStateChange::AddU { clause_idx } => {
                self.unprocessed.push_back(*clause_idx);
                self.notify_add_unprocessed(*clause_idx);
                self.event_log.push(change);
            }
            ProofStateChange::RemoveU { clause_idx, rule_name: _ } => {
                if self.unprocessed.iter().any(|&x| x == *clause_idx) {
                    self.unprocessed.retain(|&x| x != *clause_idx);
                    self.notify_remove_unprocessed(*clause_idx);
                    self.event_log.push(change);
                }
            }
            ProofStateChange::AddP { clause_idx } => {
                self.processed.insert(*clause_idx);
                self.notify_add_processed(*clause_idx);
                self.event_log.push(change);
            }
            ProofStateChange::RemoveP { clause_idx, rule_name: _ } => {
                if self.processed.remove(clause_idx) {
                    self.notify_remove_processed(*clause_idx);
                    self.event_log.push(change);
                }
            }
        }
    }

    pub fn saturate(mut self) -> (SaturationResult, Option<SaturationProfile>, SaturationTrace) {
        let mut profile = if self.config.enable_profiling {
            Some(SaturationProfile::default())
        } else {
            None
        };

        let start_time = Instant::now();

        let result = 'outer: loop {
            // === Step 1: Process new clauses ===
            let t0 = profile.as_ref().map(|_| Instant::now());
            'process_n: while let Some(clause_idx) = self.new.pop_front() {
                // 1a: Check empty clause immediately
                if self.clauses[clause_idx].is_empty() {
                    if let (Some(p), Some(t)) = (profile.as_mut(), t0) {
                        p.forward_simplify_time += t.elapsed();
                    }
                    // Record simplification event for empty clause
                    self.current_simplifications.push(ClauseSimplification {
                        clause_idx,
                        outcome: None, // Empty clause found
                    });
                    // Flush the final partial iteration
                    self.flush_iteration(None);
                    let proof = self.extract_proof(clause_idx);
                    break 'outer SaturationResult::Proof(proof);
                }

                // Build views for simplification rules
                let (u_indices, p_indices) = self.build_views();
                let u_view = ClauseView::new(&u_indices, &self.clauses);
                let p_view = ClauseView::new(&p_indices, &self.clauses);

                // 1b: Apply forward simplification rules
                // Collect changes first to avoid borrow conflict, then apply after loop
                let clause = &self.clauses[clause_idx];
                let mut forward_deleted = false;
                let mut forward_outcome = None;
                let mut collected_changes: Vec<ProofStateChange> = Vec::new();
                let mut is_demodulation_rule = false;

                for rule in self.simplification_rules.iter() {
                    let rule_name = rule.name();
                    let t_rule = profile.as_ref().map(|_| Instant::now());

                    let changes = rule.simplify_forward(clause_idx, clause, &u_view, &p_view, &self.clauses);

                    if !changes.is_empty() {
                        // Update profiling stats
                        if let (Some(p), Some(t)) = (profile.as_mut(), t_rule) {
                            p.record_simplification_forward(rule_name, 1, t.elapsed());
                        }

                        // Determine the outcome for tracing
                        forward_outcome = Some(match rule_name {
                            "Tautology" => SimplificationOutcome::Forward(ForwardSimplification::Tautology),
                            "Demodulation" => {
                                SimplificationOutcome::Forward(ForwardSimplification::Demodulation {
                                    demodulator: 0, // Will be set properly below
                                    result: 0,      // Will be set properly below
                                })
                            }
                            "Subsumption" => SimplificationOutcome::Forward(ForwardSimplification::Subsumption {
                                subsumer: 0, // Could track this if needed
                            }),
                            _ => SimplificationOutcome::Forward(ForwardSimplification::Tautology),
                        });

                        collected_changes = changes;
                        is_demodulation_rule = rule_name == "Demodulation";
                        forward_deleted = true;
                        break;
                    }
                }

                // Apply collected changes after the borrow ends
                for change in collected_changes {
                    if let ProofStateChange::AddN { clause, derivation } = change {
                        // For demodulation, update the outcome with correct indices
                        if is_demodulation_rule {
                            if let Some(demodulator) = derivation.demodulator() {
                                let new_idx = self.clauses.len();
                                forward_outcome = Some(SimplificationOutcome::Forward(
                                    ForwardSimplification::Demodulation {
                                        demodulator,
                                        result: new_idx,
                                    },
                                ));
                            }
                        }
                        self.apply_change(ProofStateChange::AddN { clause, derivation }, &mut profile);
                    }
                }

                if forward_deleted {
                    self.current_simplifications.push(ClauseSimplification {
                        clause_idx,
                        outcome: forward_outcome,
                    });
                    continue 'process_n;
                }

                // 1c: Clause survives - activate it
                for rule in &mut self.simplification_rules {
                    rule.on_clause_activated(clause_idx, &self.clauses[clause_idx]);
                }

                // 1d: Apply backward simplification rules
                // Collect all changes first to avoid borrow conflict, then apply after
                let mut all_backward_changes: Vec<ProofStateChange> = Vec::new();
                let mut backward_effects = Vec::new();

                {
                    let (u_indices, p_indices) = self.build_views();
                    let u_view = ClauseView::new(&u_indices, &self.clauses);
                    let p_view = ClauseView::new(&p_indices, &self.clauses);
                    let clause = &self.clauses[clause_idx];

                    for rule in self.simplification_rules.iter() {
                        let rule_name = rule.name();
                        let t_rule = profile.as_ref().map(|_| Instant::now());

                        let changes = rule.simplify_backward(clause_idx, clause, &u_view, &p_view);

                        if !changes.is_empty() {
                            // Update profiling stats
                            if let (Some(p), Some(t)) = (profile.as_mut(), t_rule) {
                                // Count affected clauses (removals for subsumption, additions for demodulation)
                                let count = changes.iter().filter(|c| {
                                    matches!(c, ProofStateChange::RemoveU { .. } | ProofStateChange::RemoveP { .. } | ProofStateChange::AddN { .. })
                                }).count();
                                p.record_simplification_backward(rule_name, count, t.elapsed());
                            }

                            // Track effects for trace (compute new_idx before collecting)
                            let mut current_new_idx = self.clauses.len();
                            for change in &changes {
                                match change {
                                    ProofStateChange::RemoveU { clause_idx: idx, .. } | ProofStateChange::RemoveP { clause_idx: idx, .. } => {
                                        backward_effects.push(BackwardSimplification::Subsumption {
                                            deleted_clause: *idx,
                                        });
                                    }
                                    ProofStateChange::AddN { derivation, .. } => {
                                        if let Some(target) = derivation.target() {
                                            backward_effects.push(BackwardSimplification::Demodulation {
                                                old_clause: target,
                                                result: current_new_idx,
                                            });
                                            current_new_idx += 1;
                                        }
                                    }
                                    _ => {}
                                }
                            }

                            all_backward_changes.extend(changes);
                        }
                    }
                }

                // Apply all collected changes after borrows end
                for change in all_backward_changes {
                    self.apply_change(change, &mut profile);
                }

                // Transfer: move to unprocessed (U)
                self.unprocessed.push_back(clause_idx);
                self.notify_add_unprocessed(clause_idx);
                self.event_log.push(ProofStateChange::AddU { clause_idx });

                // Record: clause survived and was transferred
                self.current_simplifications.push(ClauseSimplification {
                    clause_idx,
                    outcome: Some(SimplificationOutcome::Backward {
                        effects: backward_effects,
                    }),
                });
            }
            if let (Some(p), Some(t)) = (profile.as_mut(), t0) {
                p.forward_simplify_time += t.elapsed();
            }

            // === Step 2: Check saturation ===
            if self.unprocessed.is_empty() {
                self.flush_iteration(None);
                let steps = self.build_proof_steps();
                let clauses = self.clauses.clone();
                break SaturationResult::Saturated(steps, clauses);
            }

            // Check resource limits
            if let Some(limit_mb) = self.config.max_clause_memory_mb {
                if self.clause_memory_bytes >= limit_mb * 1024 * 1024 {
                    self.flush_iteration(None);
                    let steps = self.build_proof_steps();
                    let clauses = self.clauses.clone();
                    break SaturationResult::ResourceLimit(steps, clauses);
                }
            }
            if self.config.max_iterations > 0 && self.current_iteration >= self.config.max_iterations {
                self.flush_iteration(None);
                let steps = self.build_proof_steps();
                let clauses = self.clauses.clone();
                break SaturationResult::ResourceLimit(steps, clauses);
            }
            if self.config.max_clauses > 0 && self.clauses.len() >= self.config.max_clauses {
                self.flush_iteration(None);
                let steps = self.build_proof_steps();
                let clauses = self.clauses.clone();
                break SaturationResult::ResourceLimit(steps, clauses);
            }
            if start_time.elapsed() > self.config.timeout {
                self.flush_iteration(None);
                let steps = self.build_proof_steps();
                let clauses = self.clauses.clone();
                break SaturationResult::Timeout(steps, clauses);
            }

            self.current_iteration += 1;

            // Track max sizes
            if let Some(p) = profile.as_mut() {
                p.iterations = self.current_iteration;
                let up_size = self.unprocessed.len();
                let pr_size = self.processed.len();
                if up_size > p.max_unprocessed_size {
                    p.max_unprocessed_size = up_size;
                }
                if pr_size > p.max_processed_size {
                    p.max_processed_size = pr_size;
                }
            }

            // === Step 3: Select given clause ===
            let t0 = profile.as_ref().map(|_| Instant::now());
            let given_idx = match self.select_given_clause() {
                Some(idx) => idx,
                None => {
                    self.flush_iteration(None);
                    let steps = self.build_proof_steps();
                    let clauses = self.clauses.clone();
                    break SaturationResult::Saturated(steps, clauses);
                }
            };
            if let (Some(p), Some(t)) = (profile.as_mut(), t0) {
                p.select_given_time += t.elapsed();
            }

            // Flush the previous iteration's simplifications with the given clause selection
            self.flush_iteration(Some(given_idx));

            // === Step 4: Transfer given clause from U to P ===
            self.unprocessed.retain(|&x| x != given_idx);
            self.notify_remove_unprocessed(given_idx);
            self.event_log.push(ProofStateChange::RemoveU { clause_idx: given_idx, rule_name: "Selection".into() });
            self.processed.insert(given_idx);
            self.notify_add_processed(given_idx);
            self.event_log.push(ProofStateChange::AddP { clause_idx: given_idx });

            // === Step 5: Generate inferences using polymorphic rules ===
            let t0 = profile.as_ref().map(|_| Instant::now());
            let new_inferences = self.generate_inferences_polymorphic(given_idx, &mut profile);
            if let (Some(p), Some(t)) = (profile.as_mut(), t0) {
                p.generate_inferences_time += t.elapsed();
                p.clauses_generated += new_inferences.len();
            }

            // Add new inferences to new set (deduplicate within the batch first)
            let t0 = profile.as_ref().map(|_| Instant::now());
            let mut seen_in_batch = HashSet::new();
            for inference in new_inferences {
                // Orient the clause first to get canonical form
                let mut oriented = inference.conclusion.clone();
                orient_clause_equalities(&mut oriented);
                let clause_str = format!("{}", oriented);
                if seen_in_batch.insert(clause_str) {
                    let derivation = inference.derivation.clone();
                    if let Some(new_idx) = self.add_clause_to_new(inference) {
                        self.current_generation.push(
                            GeneratingInference::from_derivation(new_idx, &derivation),
                        );
                    }
                }
            }
            if let (Some(p), Some(t)) = (profile.as_mut(), t0) {
                p.add_inferences_time += t.elapsed();
            }
        };

        // Build structured trace from accumulated iterations
        let trace = SaturationTrace {
            clauses: self.clauses.iter().map(|c| c.to_string()).collect(),
            initial_clause_count: self.initial_clause_count,
            iterations: std::mem::take(&mut self.iterations),
        };

        // Finalize profile
        if let Some(p) = profile.as_mut() {
            p.total_time = start_time.elapsed();
            p.selector_name = self.clause_selector.name().to_string();
            if let Some(stats) = self.clause_selector.stats() {
                p.selector_cache_hits = stats.cache_hits;
                p.selector_cache_misses = stats.cache_misses;
                p.selector_embed_time = stats.embed_time;
                p.selector_score_time = stats.score_time;
            }
        }

        (result, profile, trace)
    }

    /// Select the next given clause using the configured selector
    fn select_given_clause(&mut self) -> Option<usize> {
        self.clause_selector
            .select(&mut self.unprocessed, &self.clauses)
    }

    /// Generate all inferences using polymorphic rules
    fn generate_inferences_polymorphic(
        &self,
        given_idx: usize,
        profile: &mut Option<SaturationProfile>,
    ) -> Vec<InferenceResult> {
        let mut results = Vec::new();
        let given_clause = &self.clauses[given_idx];
        let selector = self.literal_selector.as_ref();

        // Build view of processed clauses (excluding given, which was just added)
        let p_indices: Vec<usize> = self.processed.iter().copied().filter(|&idx| idx != given_idx).collect();
        let p_view = ClauseView::new(&p_indices, &self.clauses);

        // Apply each generating rule
        for rule in &self.generating_rules {
            let rule_name = rule.name();
            let t0 = profile.as_ref().map(|_| Instant::now());
            let before = results.len();

            let changes = rule.generate(given_idx, given_clause, &p_view, selector);

            // Convert ProofStateChange::AddN to InferenceResult
            for change in changes {
                if let ProofStateChange::AddN { clause, derivation } = change {
                    results.push(InferenceResult {
                        conclusion: clause,
                        derivation,
                    });
                }
            }

            // Update profile
            if let (Some(p), Some(t)) = (profile.as_mut(), t0) {
                let count = results.len() - before;
                p.record_generating_rule(rule_name, count, t.elapsed());
            }
        }

        results
    }

    /// Add a new clause to the new set (no forward simplification yet).
    fn add_clause_to_new(&mut self, inference: InferenceResult) -> Option<usize> {
        // Check clause size limit
        if inference.conclusion.literals.len() > self.config.max_clause_size {
            return None;
        }

        // Orient equalities first so we store the canonical form
        let mut current_clause = inference.conclusion.clone();
        orient_clause_equalities(&mut current_clause);

        // Add the clause to new set (pending, not active yet)
        let new_idx = self.clauses.len();
        let mut clause_with_id = current_clause.clone();
        clause_with_id.id = Some(new_idx);
        clause_with_id.age = self.current_iteration;
        clause_with_id.role = crate::fol::ClauseRole::Derived;

        // Notify rules about pending clause
        for rule in &mut self.simplification_rules {
            rule.on_clause_pending(new_idx, &current_clause);
        }

        // Track clause memory
        self.clause_memory_bytes += clause_with_id.memory_bytes();

        self.clauses.push(clause_with_id);
        self.derivations.push(Some(inference.derivation));
        self.new.push_back(new_idx);

        Some(new_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fol::{Atom, CNFFormula, Constant, Literal, PredicateSymbol, Term, Variable};
    use crate::selection::AgeWeightSelector;

    fn create_selector() -> Box<dyn ClauseSelector> {
        Box::new(AgeWeightSelector::default())
    }

    #[test]
    fn test_simple_proof() {
        // P(a)
        // ~P(X) ∨ Q(X)
        // ~Q(a)
        // Should derive empty clause

        let p = PredicateSymbol {
            name: "P".to_string(),
            arity: 1,
        };
        let q = PredicateSymbol {
            name: "Q".to_string(),
            arity: 1,
        };
        let a = Term::Constant(Constant {
            name: "a".to_string(),
        });
        let x = Term::Variable(Variable {
            name: "X".to_string(),
        });

        let clauses = vec![
            Clause::new(vec![Literal::positive(Atom {
                predicate: p.clone(),
                args: vec![a.clone()],
            })]),
            Clause::new(vec![
                Literal::negative(Atom {
                    predicate: p.clone(),
                    args: vec![x.clone()],
                }),
                Literal::positive(Atom {
                    predicate: q.clone(),
                    args: vec![x.clone()],
                }),
            ]),
            Clause::new(vec![Literal::negative(Atom {
                predicate: q.clone(),
                args: vec![a.clone()],
            })]),
        ];

        let formula = CNFFormula { clauses };
        let (result, profile, _) = crate::saturation::saturate(formula, SaturationConfig::default(), create_selector());

        match result {
            SaturationResult::Proof(_) => {} // Expected
            _ => panic!("Expected to find proof"),
        }
        assert!(profile.is_none(), "Profiling should be disabled by default");
    }

    #[test]
    fn test_profiling_enabled() {
        let p = PredicateSymbol {
            name: "P".to_string(),
            arity: 1,
        };
        let q = PredicateSymbol {
            name: "Q".to_string(),
            arity: 1,
        };
        let a = Term::Constant(Constant {
            name: "a".to_string(),
        });
        let x = Term::Variable(Variable {
            name: "X".to_string(),
        });

        let clauses = vec![
            Clause::new(vec![Literal::positive(Atom {
                predicate: p.clone(),
                args: vec![a.clone()],
            })]),
            Clause::new(vec![
                Literal::negative(Atom {
                    predicate: p.clone(),
                    args: vec![x.clone()],
                }),
                Literal::positive(Atom {
                    predicate: q.clone(),
                    args: vec![x.clone()],
                }),
            ]),
            Clause::new(vec![Literal::negative(Atom {
                predicate: q.clone(),
                args: vec![a.clone()],
            })]),
        ];

        let formula = CNFFormula { clauses };
        let mut config = SaturationConfig::default();
        config.enable_profiling = true;
        let (result, profile, _) = crate::saturation::saturate(formula, config, create_selector());

        match result {
            SaturationResult::Proof(_) => {}
            _ => panic!("Expected to find proof"),
        }

        let profile = profile.expect("Profile should be present when enabled");
        assert!(profile.total_time.as_nanos() > 0, "total_time should be non-zero");
        assert!(profile.iterations > 0, "iterations should be non-zero");
        assert!(
            profile.generating_rules.get("Resolution").map_or(0, |s| s.count) > 0,
            "should have resolution inferences"
        );
        assert_eq!(profile.selector_name, "AgeWeight");

        // Verify JSON serialization works
        let json = serde_json::to_string(&profile).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(value["total_time"].as_f64().unwrap() > 0.0);
        assert!(value["iterations"].as_u64().unwrap() > 0);
        assert_eq!(value["selector_name"].as_str().unwrap(), "AgeWeight");
        // Verify generating_rules structure
        assert!(value["generating_rules"]["Resolution"]["count"].as_u64().unwrap() > 0);
    }

    #[test]
    fn test_saturation_trace_populated() {
        let p = PredicateSymbol {
            name: "P".to_string(),
            arity: 1,
        };
        let q = PredicateSymbol {
            name: "Q".to_string(),
            arity: 1,
        };
        let a = Term::Constant(Constant {
            name: "a".to_string(),
        });
        let x = Term::Variable(Variable {
            name: "X".to_string(),
        });

        let clauses = vec![
            Clause::new(vec![Literal::positive(Atom {
                predicate: p.clone(),
                args: vec![a.clone()],
            })]),
            Clause::new(vec![
                Literal::negative(Atom {
                    predicate: p.clone(),
                    args: vec![x.clone()],
                }),
                Literal::positive(Atom {
                    predicate: q.clone(),
                    args: vec![x.clone()],
                }),
            ]),
            Clause::new(vec![Literal::negative(Atom {
                predicate: q.clone(),
                args: vec![a.clone()],
            })]),
        ];

        let formula = CNFFormula { clauses };
        let (result, _, trace) = crate::saturation::saturate(formula, SaturationConfig::default(), create_selector());

        assert!(matches!(result, SaturationResult::Proof(_)));

        // Trace should have content
        assert_eq!(trace.initial_clause_count, 3);
        assert!(trace.clauses.len() >= 3, "Should have at least the 3 input clauses");
        assert!(!trace.iterations.is_empty(), "Should have at least one iteration");

        // Verify serde serialization round-trips
        let json = serde_json::to_string(&trace).unwrap();
        let parsed: SaturationTrace = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.initial_clause_count, trace.initial_clause_count);
        assert_eq!(parsed.clauses.len(), trace.clauses.len());
        assert_eq!(parsed.iterations.len(), trace.iterations.len());
    }
}
