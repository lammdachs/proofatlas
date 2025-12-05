//! Main saturation state and algorithm

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
    pub step_limit: Option<usize>,
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
            max_clauses: 10000,
            max_iterations: 10000,
            max_clause_size: 100,
            timeout: Duration::from_secs(60),
            literal_selection: LiteralSelectionStrategy::Sel0,
            step_limit: None,
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

/// Main saturation state
pub struct SaturationState {
    /// All clauses (processed and unprocessed)
    clauses: Vec<Clause>,
    /// Indices of processed clauses
    processed: HashSet<usize>,
    /// Queue of unprocessed clause indices
    unprocessed: VecDeque<usize>,
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
}

impl SaturationState {
    /// Create new saturation state from initial clauses
    ///
    /// # Arguments
    /// * `initial_clauses` - The initial clause set
    /// * `config` - Saturation configuration
    /// * `clause_selector` - ONNX-based clause selector
    pub fn new(
        initial_clauses: Vec<Clause>,
        config: SaturationConfig,
        clause_selector: Box<dyn ClauseSelector>,
    ) -> Self {
        let mut clauses = Vec::new();
        let mut unprocessed = VecDeque::new();
        let mut subsumption_checker = SubsumptionChecker::new();

        let mut proof_steps = Vec::new();

        // Add initial clauses with IDs
        for (i, mut clause) in initial_clauses.into_iter().enumerate() {
            clause.id = Some(i);
            // Orient equalities before adding
            let mut oriented = clause.clone();
            orient_clause_equalities(&mut oriented);

            // Add to subsumption checker
            let idx = subsumption_checker.add_clause(oriented.clone());
            assert_eq!(idx, i); // Initial clauses should match their index

            // Create proof step for initial clause
            proof_steps.push(ProofStep {
                inference: InferenceResult {
                    rule: InferenceRule::Input,
                    premises: vec![], // No parents for initial clauses
                    conclusion: oriented,
                },
                clause_idx: i,
            });

            clauses.push(clause);
            unprocessed.push_back(i);
        }

        // Create literal selector based on configuration
        let literal_selector: Box<dyn LiteralSelector> = match config.literal_selection {
            LiteralSelectionStrategy::Sel0 => Box::new(SelectAll),
            LiteralSelectionStrategy::Sel20 => Box::new(SelectMaximal::new()),
            LiteralSelectionStrategy::Sel21 => Box::new(SelectUniqueMaximalOrNegOrMaximal::new()),
            LiteralSelectionStrategy::Sel22 => Box::new(SelectNegMaxWeightOrMaximal::new()),
        };

        SaturationState {
            clauses,
            processed: HashSet::new(),
            unprocessed,
            subsumption_checker,
            proof_steps,
            config,
            clause_selector,
            literal_selector,
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

    /// Run the saturation algorithm
    /// Get all clauses (for JSON export)
    pub fn get_clauses(&self) -> &[Clause] {
        &self.clauses
    }

    pub fn saturate(mut self) -> SaturationResult {
        let start_time = Instant::now();
        let mut iterations = 0;
        let mut steps = 0;

        while let Some(given_idx) = self.select_given_clause() {
            // Check step limit if specified
            if let Some(limit) = self.config.step_limit {
                if steps >= limit {
                    return SaturationResult::ResourceLimit(
                        self.proof_steps.clone(),
                        self.clauses.clone(),
                    );
                }
            }

            // Check other limits
            if iterations >= self.config.max_iterations {
                return SaturationResult::ResourceLimit(
                    self.proof_steps.clone(),
                    self.clauses.clone(),
                );
            }
            if self.clauses.len() >= self.config.max_clauses {
                return SaturationResult::ResourceLimit(
                    self.proof_steps.clone(),
                    self.clauses.clone(),
                );
            }
            if start_time.elapsed() > self.config.timeout {
                return SaturationResult::Timeout(self.proof_steps.clone(), self.clauses.clone());
            }

            iterations += 1;
            steps += 1;

            // Process the given clause
            let given_clause = &self.clauses[given_idx];

            // Check if it's the empty clause
            if given_clause.is_empty() {
                return SaturationResult::Proof(Proof {
                    steps: self.proof_steps.clone(),
                    empty_clause_idx: given_idx,
                    all_clauses: self.clauses.clone(),
                });
            }

            // Record the selection of the given clause as a proof step
            // (This helps track the saturation process even when no inferences are generated)
            self.proof_steps.push(ProofStep {
                inference: InferenceResult {
                    rule: InferenceRule::GivenClauseSelection,
                    premises: vec![],
                    conclusion: given_clause.clone(),
                },
                clause_idx: given_idx,
            });

            // Generate new clauses by inference with processed clauses
            let new_inferences = self.generate_inferences(given_idx);

            // Add given clause to processed
            self.processed.insert(given_idx);

            // If given clause is a unit equality, perform backward demodulation
            if given_clause.literals.len() == 1
                && given_clause.literals[0].polarity
                && given_clause.literals[0].atom.is_equality()
            {
                self.backward_demodulate_with_unit(given_idx);
            }

            // Process new inferences - deduplicate within the batch first
            let mut seen_in_batch = HashSet::new();
            let mut unique_inferences = Vec::new();

            for inference in new_inferences {
                // Orient the clause first to get canonical form
                let mut oriented = inference.conclusion.clone();
                orient_clause_equalities(&mut oriented);
                let clause_str = format!("{}", oriented);
                if seen_in_batch.insert(clause_str) {
                    unique_inferences.push(inference);
                }
            }

            for inference in unique_inferences {
                if let Some(new_idx) = self.add_clause(inference) {
                    // Check if we derived empty clause
                    if self.clauses[new_idx].is_empty() {
                        return SaturationResult::Proof(Proof {
                            steps: self.proof_steps.clone(),
                            empty_clause_idx: new_idx,
                            all_clauses: self.clauses.clone(),
                        });
                    }
                }
            }
        }

        // No more clauses to process
        SaturationResult::Saturated(self.proof_steps.clone(), self.clauses.clone())
    }

    /// Select the next given clause using the configured selector
    fn select_given_clause(&mut self) -> Option<usize> {
        self.clause_selector
            .select(&mut self.unprocessed, &self.clauses)
    }

    /// Generate all inferences between given clause and processed clauses
    fn generate_inferences(&self, given_idx: usize) -> Vec<InferenceResult> {
        let mut results = Vec::new();
        let given_clause = &self.clauses[given_idx];
        let selector = self.literal_selector.as_ref();

        // Factoring on given clause
        results.extend(factoring(given_clause, given_idx, selector));

        // Equality resolution on given clause
        results.extend(equality_resolution(given_clause, given_idx, selector));

        // Equality factoring on given clause
        results.extend(equality_factoring(given_clause, given_idx, selector));

        // Inferences with each processed clause
        for &processed_idx in &self.processed {
            let processed_clause = &self.clauses[processed_idx];

            // Resolution
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

            // Superposition
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
        }

        // IMPORTANT: Also do self-inferences (given clause with itself)
        // This is needed for cases like associativity self-superposition
        results.extend(resolution(
            given_clause,
            given_clause,
            given_idx,
            given_idx,
            selector,
        ));
        results.extend(superposition(
            given_clause,
            given_clause,
            given_idx,
            given_idx,
            selector,
        ));

        results
    }

    /// Add a new clause if it's not redundant
    fn add_clause(&mut self, inference: InferenceResult) -> Option<usize> {
        // Check clause size limit
        if inference.conclusion.literals.len() > self.config.max_clause_size {
            return None;
        }

        // Check if tautology
        if inference.conclusion.is_tautology() {
            return None;
        }

        // Orient equalities first so we check the canonical form
        let mut oriented_clause = inference.conclusion.clone();
        orient_clause_equalities(&mut oriented_clause);

        // Apply demodulation with all unit equalities
        let demodulated_clause = self.demodulate_clause(oriented_clause.clone(), &inference);

        // Check subsumption for redundancy elimination
        if self.subsumption_checker.is_subsumed(&demodulated_clause) {
            return None;
        }

        // Add the clause
        let new_idx = self.clauses.len();
        let mut clause_with_id = demodulated_clause.clone();
        clause_with_id.id = Some(new_idx);

        // Add to subsumption checker
        let idx_from_subsumption = self
            .subsumption_checker
            .add_clause(demodulated_clause.clone());
        assert_eq!(idx_from_subsumption, new_idx);

        self.clauses.push(clause_with_id.clone());
        self.unprocessed.push_back(new_idx);

        // Record proof step with demodulated clause
        let mut final_inference = inference;
        final_inference.conclusion = clause_with_id;
        self.proof_steps.push(ProofStep {
            inference: final_inference,
            clause_idx: new_idx,
        });

        Some(new_idx)
    }

    /// Apply demodulation to a clause using all available unit equalities
    fn demodulate_clause(&self, clause: Clause, _original_inference: &InferenceResult) -> Clause {
        let mut changed = true;
        let mut current_clause = clause;

        // Keep applying demodulation until no more changes
        while changed {
            changed = false;

            // Try demodulation with each processed unit equality
            for &unit_idx in &self.processed {
                let unit_clause = &self.clauses[unit_idx];

                // Check if it's a unit equality
                if unit_clause.literals.len() == 1
                    && unit_clause.literals[0].polarity
                    && unit_clause.literals[0].atom.is_equality()
                {
                    // Try to demodulate
                    let results =
                        demodulation::demodulate(unit_clause, &current_clause, unit_idx, 0);
                    if !results.is_empty() {
                        // Apply the first demodulation (there should be at most one)
                        current_clause = results[0].conclusion.clone();
                        changed = true;
                        break; // Start over with the new clause
                    }
                }
            }
        }

        current_clause
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
            // Remove old clause from subsumption checker
            // Note: We'll add the new clause through the normal add_clause process

            // Mark old clause as inactive by removing from processed/unprocessed
            self.processed.remove(&old_idx);
            self.unprocessed.retain(|&idx| idx != old_idx);

            // Add the simplified clause as a new clause
            // This will apply orientation, subsumption checking, etc.
            self.add_clause(inference_result);
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
        let result = crate::saturation::saturate(formula, SaturationConfig::default(), create_selector());

        match result {
            SaturationResult::Proof(_) => {} // Expected
            _ => panic!("Expected to find proof"),
        }
    }
}
