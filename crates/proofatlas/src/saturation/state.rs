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
use crate::core::{Clause, Proof, ProofStep};
use crate::inference::{
    demodulation, equality_factoring, equality_resolution, factoring, resolution, superposition,
    InferenceResult, InferenceRule,
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
            literal_selection: LiteralSelectionStrategy::Sel0,
            max_clause_memory_mb: None,
            enable_profiling: false,
        }
    }
}

/// Result of saturation
#[derive(Debug, Clone)]
pub enum SaturationResult {
    /// Empty clause derived - proof found (includes all proof steps)
    Proof(Proof),
    /// Saturated without finding empty clause (includes all proof steps and final clauses)
    Saturated(Vec<ProofStep>, Vec<Clause>),
    /// Resource limit reached (includes proof steps so far and final clauses)
    ResourceLimit(Vec<ProofStep>, Vec<Clause>),
    /// Timeout reached (includes proof steps so far and final clauses)
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
    /// Set A: Processed/active clauses (used for generating inferences)
    processed: HashSet<usize>,
    /// Set P: Unprocessed/passive clauses (awaiting selection)
    unprocessed: VecDeque<usize>,
    /// Set N: New clauses (awaiting forward simplification)
    new: VecDeque<usize>,
    /// Proof steps
    proof_steps: Vec<ProofStep>,
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
        let mut clauses = Vec::new();
        let mut new = VecDeque::new();
        let mut subsumption_checker = SubsumptionChecker::new();
        let mut clause_memory_bytes = 0usize;

        let mut proof_steps = Vec::new();

        // Add initial clauses with IDs, filtering tautologies
        // Note: clauses in N are added as pending (not active for subsumption)
        let mut clause_idx = 0;
        for mut clause in initial_clauses.into_iter() {
            // Orient equalities before adding
            let mut oriented = clause.clone();
            orient_clause_equalities(&mut oriented);

            // Skip tautologies (e.g., P(x) ∨ ~P(x) ∨ ...)
            if oriented.is_tautology() {
                continue;
            }

            clause.id = Some(clause_idx);

            // Track clause memory
            clause_memory_bytes += clause.memory_bytes();

            // Add to subsumption checker as pending (not active yet)
            let idx = subsumption_checker.add_clause_pending(oriented.clone());
            assert_eq!(idx, clause_idx);

            // Create proof step for initial clause
            proof_steps.push(ProofStep {
                inference: InferenceResult {
                    rule: InferenceRule::Input,
                    premises: vec![],
                    conclusion: oriented,
                },
                clause_idx,
            });

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
            processed: HashSet::new(),
            unprocessed: VecDeque::new(),
            new,
            subsumption_checker,
            proof_steps,
            config,
            clause_selector,
            literal_selector,
            clause_memory_bytes,
            current_iteration: 0,
            unit_equalities: HashSet::new(),
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

    pub fn saturate(mut self) -> (SaturationResult, Option<SaturationProfile>) {
        let mut profile = if self.config.enable_profiling {
            Some(SaturationProfile::default())
        } else {
            None
        };

        let start_time = Instant::now();

        let result = loop {
            // Forward simplification: process new clauses until fixed point, move to unprocessed
            let t0 = profile.as_ref().map(|_| Instant::now());
            let fwd_result = self.forward_simplify_new(&mut profile);
            if let (Some(p), Some(t)) = (profile.as_mut(), t0) {
                p.forward_simplify_time += t.elapsed();
            }
            if let Some(result) = fwd_result {
                break result;
            }

            // Select given clause from unprocessed
            let t0 = profile.as_ref().map(|_| Instant::now());
            let given_idx = match self.select_given_clause() {
                Some(idx) => idx,
                None => {
                    // No more clauses to process
                    break SaturationResult::Saturated(
                        self.proof_steps.clone(),
                        self.clauses.clone(),
                    );
                }
            };
            if let (Some(p), Some(t)) = (profile.as_mut(), t0) {
                p.select_given_time += t.elapsed();
            }

            // Check clause memory limit (directly comparable across provers)
            if let Some(limit_mb) = self.config.max_clause_memory_mb {
                if self.clause_memory_bytes >= limit_mb * 1024 * 1024 {
                    break SaturationResult::ResourceLimit(
                        self.proof_steps.clone(),
                        self.clauses.clone(),
                    );
                }
            }

            // Check other limits
            if self.config.max_iterations > 0 && self.current_iteration >= self.config.max_iterations {
                break SaturationResult::ResourceLimit(
                    self.proof_steps.clone(),
                    self.clauses.clone(),
                );
            }
            if self.config.max_clauses > 0 && self.clauses.len() >= self.config.max_clauses {
                break SaturationResult::ResourceLimit(
                    self.proof_steps.clone(),
                    self.clauses.clone(),
                );
            }
            if start_time.elapsed() > self.config.timeout {
                break SaturationResult::Timeout(self.proof_steps.clone(), self.clauses.clone());
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

            // Check if given clause is empty
            let given_clause = &self.clauses[given_idx];
            if given_clause.is_empty() {
                break SaturationResult::Proof(Proof {
                    steps: self.proof_steps.clone(),
                    empty_clause_idx: given_idx,
                    all_clauses: self.clauses.clone(),
                });
            }

            // Record the selection of the given clause as a proof step
            self.proof_steps.push(ProofStep {
                inference: InferenceResult {
                    rule: InferenceRule::GivenClauseSelection,
                    premises: vec![],
                    conclusion: given_clause.clone(),
                },
                clause_idx: given_idx,
            });

            // Generate new clauses by inference with processed clauses
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
                    if self.add_clause_to_new(inference).is_some() {
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

        (result, profile)
    }

    /// Process new clauses: forward simplify by A ∪ P, then backward simplify A and P
    /// Returns Some(result) if empty clause found
    fn forward_simplify_new(
        &mut self,
        profile: &mut Option<SaturationProfile>,
    ) -> Option<SaturationResult> {
        while let Some(clause_idx) = self.new.pop_front() {
            let clause = &self.clauses[clause_idx];

            // SimplifyFwd: try to simplify using indexed unit equalities from A ∪ P
            let mut current_clause = clause.clone();
            let mut simplified = false;
            let mut demod_premise = None;

            // Iterate only over known unit equalities (not all clauses)
            let unit_eq_indices: Vec<usize> = self.unit_equalities.iter().copied().collect();

            let t0 = profile.as_ref().map(|_| Instant::now());
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
            if let (Some(p), Some(t)) = (profile.as_mut(), t0) {
                p.forward_demod_time += t.elapsed();
            }

            if simplified {
                // Check if simplified to tautology
                if current_clause.is_tautology() {
                    if let Some(p) = profile.as_mut() {
                        p.tautologies_deleted += 1;
                    }
                    continue;
                }

                // Check if subsumed after simplification
                let t0 = profile.as_ref().map(|_| Instant::now());
                let subsumed = self.subsumption_checker.is_subsumed(&current_clause);
                if let (Some(p), Some(t)) = (profile.as_mut(), t0) {
                    p.forward_subsumption_time += t.elapsed();
                }
                if subsumed {
                    if let Some(p) = profile.as_mut() {
                        p.clauses_subsumed_forward += 1;
                    }
                    continue;
                }

                // Add simplified clause as new clause, put back in new set (still pending)
                let new_idx = self.clauses.len();
                let mut clause_with_id = current_clause.clone();
                clause_with_id.id = Some(new_idx);

                let idx_from_subsumption = self
                    .subsumption_checker
                    .add_clause_pending(current_clause.clone());
                assert_eq!(idx_from_subsumption, new_idx);

                self.clause_memory_bytes += clause_with_id.memory_bytes();
                self.clauses.push(clause_with_id.clone());
                self.new.push_back(new_idx);

                // Record proof step
                self.proof_steps.push(ProofStep {
                    inference: InferenceResult {
                        rule: InferenceRule::Demodulation,
                        premises: vec![demod_premise.unwrap(), clause_idx],
                        conclusion: clause_with_id.clone(),
                    },
                    clause_idx: new_idx,
                });

                // Check if we derived empty clause
                if clause_with_id.is_empty() {
                    return Some(SaturationResult::Proof(Proof {
                        steps: self.proof_steps.clone(),
                        empty_clause_idx: new_idx,
                        all_clauses: self.clauses.clone(),
                    }));
                }
            } else {
                // DeleteFwd: check if subsumed by A ∪ P
                let t0 = profile.as_ref().map(|_| Instant::now());
                let subsumed = self.subsumption_checker.is_subsumed(&current_clause);
                if let (Some(p), Some(t)) = (profile.as_mut(), t0) {
                    p.forward_subsumption_time += t.elapsed();
                }
                if subsumed {
                    if let Some(p) = profile.as_mut() {
                        p.clauses_subsumed_forward += 1;
                    }
                    continue;
                }

                // Clause survives - activate it and do backward simplification before transfer to P
                self.subsumption_checker.activate_clause(clause_idx);

                // Track unit equalities for demodulation index
                if Self::is_unit_equality(&self.clauses[clause_idx]) {
                    self.unit_equalities.insert(clause_idx);
                }

                // DeleteBwd: remove clauses in P and A subsumed by this clause
                let t0 = profile.as_ref().map(|_| Instant::now());
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
                for idx in subsumed {
                    self.processed.remove(&idx);
                    self.unprocessed.retain(|&x| x != idx);
                    self.unit_equalities.remove(&idx);
                }
                if let (Some(p), Some(t)) = (profile.as_mut(), t0) {
                    p.backward_subsumption_time += t.elapsed();
                }

                // SimplifyBwd: if this clause is a unit equality, simplify P and A
                if Self::is_unit_equality(&self.clauses[clause_idx]) {
                    let t0 = profile.as_ref().map(|_| Instant::now());
                    let new_before = self.new.len();
                    self.backward_demodulate_with_unit(clause_idx);
                    if let (Some(p), Some(t)) = (profile.as_mut(), t0) {
                        p.backward_demod_time += t.elapsed();
                        p.clauses_demodulated_backward += self.new.len().saturating_sub(new_before);
                    }
                }

                // Transfer: move to unprocessed (P)
                self.unprocessed.push_back(clause_idx);
            }
        }

        None
    }

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

    /// Add a new clause to the new set (no forward simplification yet)
    fn add_clause_to_new(&mut self, inference: InferenceResult) -> Option<usize> {
        // Check clause size limit
        if inference.conclusion.literals.len() > self.config.max_clause_size {
            return None;
        }

        // Check if tautology
        if inference.conclusion.is_tautology() {
            return None;
        }

        // Orient equalities first so we check the canonical form
        let mut current_clause = inference.conclusion.clone();
        orient_clause_equalities(&mut current_clause);

        // Check subsumption for redundancy elimination
        if self.subsumption_checker.is_subsumed(&current_clause) {
            return None;
        }

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

        self.clauses.push(clause_with_id.clone());
        self.new.push_back(new_idx);

        // Record proof step
        let mut inf = inference;
        inf.conclusion = clause_with_id.clone();
        self.proof_steps.push(ProofStep {
            inference: inf,
            clause_idx: new_idx,
        });

        Some(new_idx)
    }

    /// Perform backward demodulation using a newly processed unit equality
    fn backward_demodulate_with_unit(&mut self, unit_idx: usize) {
        let unit_clause = &self.clauses[unit_idx];

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
            self.add_clause_to_new(inference_result);
        }
    }

    /// Consume the saturation state and return the clauses and proof steps
    pub fn into_data(self) -> (Vec<Clause>, Vec<ProofStep>) {
        (self.clauses, self.proof_steps)
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
        let (result, profile) = crate::saturation::saturate(formula, SaturationConfig::default(), create_selector());

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
        let (result, profile) = crate::saturation::saturate(formula, config, create_selector());

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
}
