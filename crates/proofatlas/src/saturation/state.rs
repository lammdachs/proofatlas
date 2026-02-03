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
//! - **Unprocessed (P)**: Clauses that have survived simplification, awaiting selection
//! - **Processed (A)**: Clauses that have been selected and used for generating inferences
//!
//! ## Algorithm
//!
//! Each iteration of the main loop:
//!
//! 1. **Process New Clauses** (`forward_simplify_new`):
//!    - For each clause C in N:
//!      - **Forward Simplification**: Simplify C using unit equalities from A ∪ P
//!      - **Forward Deletion**: Check if C is subsumed by any clause in A ∪ P
//!      - If C survives:
//!        - **Backward Deletion**: Remove clauses in A ∪ P that are subsumed by C
//!        - **Backward Simplification**: If C is a unit equality, rewrite clauses in A ∪ P; results go to N
//!        - **Transfer**: Move C from N to P
//!
//! 2. **Select Given Clause**: Choose a clause G from P using the clause selector
//!
//! 3. **Generate Inferences**: Apply inference rules between G and clauses in A;
//!    results go to N
//!
//! 4. **Activate**: Move G from P to A
//!
//! ## Key Design Decisions
//!
//! - **Forward simplification uses A ∪ P**: Unlike simpler loops that only simplify
//!   by processed clauses, we also use unprocessed clauses. This catches more
//!   redundancies earlier.
//!
//! - **Backward simplification on transfer**: When a clause moves from N to P,
//!   we immediately use it to simplify existing clauses. This is more aggressive
//!   than waiting until the clause is selected.
//!
//! - **Pending vs Active clauses**: The subsumption checker tracks which clauses
//!   are "active" (in A ∪ P) vs "pending" (in N). Only active clauses participate
//!   in subsumption checks, preventing self-subsumption of new clauses.

use super::profile::SaturationProfile;
use super::subsumption::SubsumptionChecker;
use crate::core::{
    BackwardSimplification, Clause, ClauseSimplification, Derivation, ForwardSimplification,
    GeneratingInference, Proof, ProofStep, SaturationStep, SaturationTrace, SimplificationOutcome,
};
use crate::inference::{
    demodulation, equality_factoring, equality_resolution, factoring, resolution, superposition,
    InferenceResult,
};
use crate::parser::orient_equalities::orient_clause_equalities;
use crate::inference::{
    LiteralSelector, SelectAll, SelectMaximal, SelectNegMaxWeightOrMaximal,
    SelectUniqueMaximalOrNegOrMaximal,
};
use crate::selectors::ClauseSelector;
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
    pub fn to_json(&self, time_seconds: f64) -> crate::core::json::SaturationResultJson {
        use crate::core::json::SaturationResultJson;

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
    /// Set A: Processed/active clauses (used for generating inferences)
    processed: HashSet<usize>,
    /// Set P: Unprocessed/passive clauses (awaiting selection)
    unprocessed: VecDeque<usize>,
    /// Set N: New clauses (awaiting forward simplification)
    new: VecDeque<usize>,
    /// Configuration
    config: SaturationConfig,
    /// Subsumption checker for redundancy elimination
    subsumption_checker: SubsumptionChecker,
    /// Clause selector
    clause_selector: Box<dyn ClauseSelector>,
    /// Literal selector
    literal_selector: Box<dyn LiteralSelector>,
    /// Tracked clause memory usage in bytes
    clause_memory_bytes: usize,
    /// Current iteration (used for clause age)
    current_iteration: usize,
    /// Index of unit equality clause indices in A ∪ P (for fast forward demodulation)
    unit_equalities: HashSet<usize>,
    /// Number of initial input clauses
    initial_clause_count: usize,
    /// Completed saturation iterations (for trace)
    iterations: Vec<SaturationStep>,
    /// Per-iteration accumulators (reset each iteration)
    current_simplifications: Vec<ClauseSimplification>,
    current_generation: Vec<GeneratingInference>,
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
        let mut clauses = Vec::new();
        let mut derivations = Vec::new();
        let mut new = VecDeque::new();
        let mut subsumption_checker = SubsumptionChecker::new();
        let mut clause_memory_bytes = 0usize;

        // Add initial clauses with IDs (tautologies enter N and are caught in step 1a)
        // Note: clauses in N are added as pending (not active for subsumption)
        let mut clause_idx = 0;
        for mut clause in initial_clauses.into_iter() {
            // Orient equalities before adding
            let mut oriented = clause.clone();
            orient_clause_equalities(&mut oriented);

            clause.id = Some(clause_idx);

            // Track clause memory
            clause_memory_bytes += clause.memory_bytes();

            // Add to subsumption checker as pending (not active yet)
            let idx = subsumption_checker.add_clause_pending(oriented.clone());
            assert_eq!(idx, clause_idx);

            // Record derivation for input clause
            derivations.push(Some(Derivation::Input));

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

        SaturationState {
            clauses,
            derivations,
            processed: HashSet::new(),
            unprocessed: VecDeque::new(),
            new,
            subsumption_checker,
            config,
            clause_selector,
            literal_selector,
            clause_memory_bytes,
            current_iteration: 0,
            unit_equalities: HashSet::new(),
            initial_clause_count,
            iterations: Vec::new(),
            current_simplifications: Vec::new(),
            current_generation: Vec::new(),
        }
    }

    /// Check if a clause is a unit positive equality
    fn is_unit_equality(clause: &Clause) -> bool {
        clause.literals.len() == 1
            && clause.literals[0].polarity
            && clause.literals[0].atom.is_equality()
    }

    /// Set the literal selector
    pub fn set_literal_selector(&mut self, selector: Box<dyn LiteralSelector>) {
        self.literal_selector = selector;
    }

    /// Set the clause selector
    pub fn set_clause_selector(&mut self, selector: Box<dyn ClauseSelector>) {
        self.clause_selector = selector;
    }

    /// Run the saturation algorithm
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
            while let Some(clause_idx) = self.new.pop_front() {
                // 1a: Check empty clause and tautology immediately
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
                if self.clauses[clause_idx].is_tautology() {
                    self.current_simplifications.push(ClauseSimplification {
                        clause_idx,
                        outcome: Some(SimplificationOutcome::Forward(ForwardSimplification::Tautology)),
                    });
                    if let Some(p) = profile.as_mut() {
                        p.tautologies_deleted += 1;
                    }
                    continue;
                }

                let clause = &self.clauses[clause_idx];

                // 1b: Forward simplification (demod + subsumption)
                // SimplifyFwd: try to simplify using indexed unit equalities from A ∪ P
                let mut current_clause = clause.clone();
                let mut simplified = false;
                let mut demod_premise = None;

                let unit_eq_indices: Vec<usize> = self.unit_equalities.iter().copied().collect();

                let t_demod = profile.as_ref().map(|_| Instant::now());
                for unit_idx in unit_eq_indices {
                    let unit_clause = &self.clauses[unit_idx];
                    let results =
                        demodulation::demodulate(unit_clause, &current_clause, unit_idx, clause_idx);
                    if !results.is_empty() {
                        current_clause = results[0].conclusion.clone();
                        orient_clause_equalities(&mut current_clause);
                        demod_premise = Some(unit_idx);
                        simplified = true;
                        if let Some(p) = profile.as_mut() {
                            p.clauses_demodulated_forward += 1;
                            p.demodulation_count += 1;
                        }
                        break;
                    }
                }
                if let (Some(p), Some(t)) = (profile.as_mut(), t_demod) {
                    p.forward_demod_time += t.elapsed();
                }

                if simplified {
                    // Forward demodulation: the original clause is deleted,
                    // the result re-enters N and gets its own simplification pass.
                    let new_idx = self.clauses.len();
                    let mut clause_with_id = current_clause.clone();
                    clause_with_id.id = Some(new_idx);

                    let idx_from_subsumption = self
                        .subsumption_checker
                        .add_clause_pending(current_clause.clone());
                    assert_eq!(idx_from_subsumption, new_idx);

                    self.clause_memory_bytes += clause_with_id.memory_bytes();
                    self.clauses.push(clause_with_id);
                    self.derivations.push(Some(Derivation::Demodulation {
                        demodulator: demod_premise.unwrap(),
                        target: clause_idx,
                    }));
                    self.new.push_back(new_idx);

                    // Record: original clause was forward-demodulated (deleted)
                    self.current_simplifications.push(ClauseSimplification {
                        clause_idx,
                        outcome: Some(SimplificationOutcome::Forward(
                            ForwardSimplification::Demodulation {
                                demodulator: demod_premise.unwrap(),
                                result: new_idx,
                            },
                        )),
                    });
                } else {
                    // No demodulation — check forward subsumption
                    let t_sub = profile.as_ref().map(|_| Instant::now());
                    let subsumer = self.subsumption_checker.find_subsumer(&current_clause);
                    if let (Some(p), Some(t)) = (profile.as_mut(), t_sub) {
                        p.forward_subsumption_time += t.elapsed();
                    }
                    if let Some(subsumer_idx) = subsumer {
                        self.current_simplifications.push(ClauseSimplification {
                            clause_idx,
                            outcome: Some(SimplificationOutcome::Forward(
                                ForwardSimplification::Subsumption { subsumer: subsumer_idx },
                            )),
                        });
                        if let Some(p) = profile.as_mut() {
                            p.clauses_subsumed_forward += 1;
                        }
                        continue;
                    }

                    // 1c: Backward simplification + transfer to P
                    // Clause survives - activate it
                    self.subsumption_checker.activate_clause(clause_idx);

                    // Track unit equalities for demodulation index
                    if Self::is_unit_equality(&self.clauses[clause_idx]) {
                        self.unit_equalities.insert(clause_idx);
                    }

                    // DeleteBwd: remove clauses in P and A subsumed by this clause
                    let t_bwd = profile.as_ref().map(|_| Instant::now());
                    let all_other: Vec<usize> = self
                        .processed
                        .iter()
                        .copied()
                        .chain(self.unprocessed.iter().copied())
                        .collect();
                    let subsumed = self
                        .subsumption_checker
                        .find_subsumed_by(clause_idx, &all_other);
                    if let Some(p) = profile.as_mut() {
                        p.clauses_subsumed_backward += subsumed.len();
                    }

                    let mut backward_effects = Vec::new();
                    for &idx in &subsumed {
                        backward_effects.push(BackwardSimplification::Subsumption {
                            deleted_clause: idx,
                        });
                        self.processed.remove(&idx);
                        self.unprocessed.retain(|&x| x != idx);
                        self.unit_equalities.remove(&idx);
                    }
                    if let (Some(p), Some(t)) = (profile.as_mut(), t_bwd) {
                        p.backward_subsumption_time += t.elapsed();
                    }

                    // SimplifyBwd: if this clause is a unit equality, simplify P and A
                    if Self::is_unit_equality(&self.clauses[clause_idx]) {
                        let t_bdemod = profile.as_ref().map(|_| Instant::now());
                        let new_before = self.new.len();
                        let bwd_demod_effects = self.backward_demodulate_with_unit(clause_idx);
                        backward_effects.extend(bwd_demod_effects);
                        if let (Some(p), Some(t)) = (profile.as_mut(), t_bdemod) {
                            p.backward_demod_time += t.elapsed();
                            p.clauses_demodulated_backward += self.new.len().saturating_sub(new_before);
                        }
                    }

                    // Transfer: move to unprocessed (P)
                    self.unprocessed.push_back(clause_idx);

                    // Record: clause survived and was transferred
                    self.current_simplifications.push(ClauseSimplification {
                        clause_idx,
                        outcome: Some(SimplificationOutcome::Backward {
                            effects: backward_effects,
                        }),
                    });
                }
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

            // === Step 4: Generate inferences and activate ===
            let t0 = profile.as_ref().map(|_| Instant::now());
            let new_inferences = self.generate_inferences(given_idx, &mut profile);
            if let (Some(p), Some(t)) = (profile.as_mut(), t0) {
                p.generate_inferences_time += t.elapsed();
                p.clauses_generated += new_inferences.len();
            }

            // Move given clause to processed
            self.processed.insert(given_idx);

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
                        if let Some(p) = profile.as_mut() {
                            p.clauses_added += 1;
                        }
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

    // forward_simplify_new is now inlined in saturate() as Step 1

    /// Select the next given clause using the configured selector
    fn select_given_clause(&mut self) -> Option<usize> {
        self.clause_selector
            .select(&mut self.unprocessed, &self.clauses)
    }

    /// Generate all inferences between given clause and processed clauses
    fn generate_inferences(
        &self,
        given_idx: usize,
        profile: &mut Option<SaturationProfile>,
    ) -> Vec<InferenceResult> {
        let mut results = Vec::new();
        let given_clause = &self.clauses[given_idx];
        let selector = self.literal_selector.as_ref();

        // Factoring on given clause
        let t0 = profile.as_ref().map(|_| Instant::now());
        let before = results.len();
        results.extend(factoring(given_clause, given_idx, selector));
        if let (Some(p), Some(t)) = (profile.as_mut(), t0) {
            p.factoring_count += results.len() - before;
            p.factoring_time += t.elapsed();
        }

        // Equality resolution on given clause
        let t0 = profile.as_ref().map(|_| Instant::now());
        let before = results.len();
        results.extend(equality_resolution(given_clause, given_idx, selector));
        if let (Some(p), Some(t)) = (profile.as_mut(), t0) {
            p.equality_resolution_count += results.len() - before;
            p.equality_resolution_time += t.elapsed();
        }

        // Equality factoring on given clause
        let t0 = profile.as_ref().map(|_| Instant::now());
        let before = results.len();
        results.extend(equality_factoring(given_clause, given_idx, selector));
        if let (Some(p), Some(t)) = (profile.as_mut(), t0) {
            p.equality_factoring_count += results.len() - before;
            p.equality_factoring_time += t.elapsed();
        }

        // Inferences with each processed clause
        for &processed_idx in &self.processed {
            let processed_clause = &self.clauses[processed_idx];

            // Resolution
            let t0 = profile.as_ref().map(|_| Instant::now());
            let before = results.len();
            results.extend(resolution(
                given_clause,
                processed_clause,
                given_idx,
                processed_idx,
                selector,
            ));
            results.extend(resolution(
                processed_clause,
                given_clause,
                processed_idx,
                given_idx,
                selector,
            ));
            if let (Some(p), Some(t)) = (profile.as_mut(), t0) {
                p.resolution_count += results.len() - before;
                p.resolution_time += t.elapsed();
            }

            // Superposition
            let t0 = profile.as_ref().map(|_| Instant::now());
            let before = results.len();
            results.extend(superposition(
                given_clause,
                processed_clause,
                given_idx,
                processed_idx,
                selector,
            ));
            results.extend(superposition(
                processed_clause,
                given_clause,
                processed_idx,
                given_idx,
                selector,
            ));
            if let (Some(p), Some(t)) = (profile.as_mut(), t0) {
                p.superposition_count += results.len() - before;
                p.superposition_time += t.elapsed();
            }
        }

        // IMPORTANT: Also do self-inferences (given clause with itself)
        // This is needed for cases like associativity self-superposition
        let t0 = profile.as_ref().map(|_| Instant::now());
        let before = results.len();
        results.extend(resolution(
            given_clause,
            given_clause,
            given_idx,
            given_idx,
            selector,
        ));
        if let (Some(p), Some(t)) = (profile.as_mut(), t0) {
            p.resolution_count += results.len() - before;
            p.resolution_time += t.elapsed();
        }

        let t0 = profile.as_ref().map(|_| Instant::now());
        let before = results.len();
        results.extend(superposition(
            given_clause,
            given_clause,
            given_idx,
            given_idx,
            selector,
        ));
        if let (Some(p), Some(t)) = (profile.as_mut(), t0) {
            p.superposition_count += results.len() - before;
            p.superposition_time += t.elapsed();
        }

        results
    }

    /// Add a new clause to the new set (no forward simplification yet).
    /// Tautology and subsumption checks are deferred to Step 1 of the main loop.
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
        clause_with_id.role = crate::core::ClauseRole::Derived;

        // Add to subsumption checker as pending
        let idx_from_subsumption = self
            .subsumption_checker
            .add_clause_pending(current_clause.clone());
        assert_eq!(idx_from_subsumption, new_idx);

        // Track clause memory
        self.clause_memory_bytes += clause_with_id.memory_bytes();

        self.clauses.push(clause_with_id);
        self.derivations.push(Some(inference.derivation));
        self.new.push_back(new_idx);

        Some(new_idx)
    }

    /// Perform backward demodulation using a newly processed unit equality.
    /// Returns the backward simplification effects for the trace.
    fn backward_demodulate_with_unit(&mut self, unit_idx: usize) -> Vec<BackwardSimplification> {
        let unit_clause = &self.clauses[unit_idx];
        let mut effects = Vec::new();

        // Collect clauses to demodulate (avoid modifying while iterating)
        let mut clauses_to_demodulate = Vec::new();

        // Check all processed clauses (except the unit itself)
        for &idx in &self.processed {
            if idx != unit_idx {
                clauses_to_demodulate.push(idx);
            }
        }

        // Check all unprocessed clauses
        for &idx in &self.unprocessed {
            if idx != unit_idx {
                clauses_to_demodulate.push(idx);
            }
        }

        // Track which clauses were replaced
        let mut replaced_clauses = Vec::new();

        // Try to demodulate each clause
        for clause_idx in clauses_to_demodulate {
            let original_clause = self.clauses[clause_idx].clone();

            // Try demodulation
            let results =
                demodulation::demodulate(unit_clause, &original_clause, unit_idx, clause_idx);

            if !results.is_empty() {
                // Clause was simplified - mark it for replacement
                let simplified_clause = results[0].conclusion.clone();

                // Only replace if actually different
                if simplified_clause != original_clause {
                    replaced_clauses.push((clause_idx, simplified_clause, results[0].clone()));
                }
            }
        }

        // Now process replacements
        for (old_idx, _new_clause, inference_result) in replaced_clauses {
            // Mark old clause as inactive by removing from processed/unprocessed
            self.processed.remove(&old_idx);
            self.unprocessed.retain(|&idx| idx != old_idx);
            self.unit_equalities.remove(&old_idx);

            // Add the simplified clause to the new set
            if let Some(new_idx) = self.add_clause_to_new(inference_result) {
                effects.push(BackwardSimplification::Demodulation {
                    old_clause: old_idx,
                    result: new_idx,
                });
            }
        }

        effects
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Atom, CNFFormula, Constant, Literal, PredicateSymbol, Term, Variable};
    use crate::selectors::AgeWeightSelector;

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
        assert!(profile.resolution_count > 0, "should have resolution inferences");
        assert_eq!(profile.selector_name, "AgeWeight");

        // Verify JSON serialization works
        let json = serde_json::to_string(&profile).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(value["total_time"].as_f64().unwrap() > 0.0);
        assert!(value["iterations"].as_u64().unwrap() > 0);
        assert_eq!(value["selector_name"].as_str().unwrap(), "AgeWeight");
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
