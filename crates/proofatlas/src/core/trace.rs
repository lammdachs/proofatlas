//! Types for saturation trace and clause derivations.

use serde::{Deserialize, Serialize};

/// How a clause was derived (for proofs and internal derivation tracking).
///
/// Encodes both the inference rule and its premises in a single value,
/// replacing the old `InferenceRule` + `premises: Vec<usize>` pair.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Derivation {
    Input,
    Resolution { parent1: usize, parent2: usize },
    Factoring { parent: usize },
    Superposition { parent1: usize, parent2: usize },
    EqualityResolution { parent: usize },
    EqualityFactoring { parent: usize },
    Demodulation { demodulator: usize, target: usize },
}

impl Derivation {
    /// Get the premise clause indices.
    pub fn premises(&self) -> Vec<usize> {
        match self {
            Derivation::Input => vec![],
            Derivation::Resolution { parent1, parent2 } => vec![*parent1, *parent2],
            Derivation::Factoring { parent } => vec![*parent],
            Derivation::Superposition { parent1, parent2 } => vec![*parent1, *parent2],
            Derivation::EqualityResolution { parent } => vec![*parent],
            Derivation::EqualityFactoring { parent } => vec![*parent],
            Derivation::Demodulation { demodulator, target } => vec![*demodulator, *target],
        }
    }

    /// Get a human-readable rule name.
    pub fn rule_name(&self) -> &'static str {
        match self {
            Derivation::Input => "Input",
            Derivation::Resolution { .. } => "Resolution",
            Derivation::Factoring { .. } => "Factoring",
            Derivation::Superposition { .. } => "Superposition",
            Derivation::EqualityResolution { .. } => "EqualityResolution",
            Derivation::EqualityFactoring { .. } => "EqualityFactoring",
            Derivation::Demodulation { .. } => "Demodulation",
        }
    }
}

/// One iteration of the given-clause algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaturationStep {
    pub simplifications: Vec<ClauseSimplification>,
    /// Index of the clause selected from P. `None` means saturated (P was empty).
    pub given_clause: Option<usize>,
    pub generating_inferences: Vec<GeneratingInference>,
}

/// What happened when a clause was popped from N for simplification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClauseSimplification {
    pub clause_idx: usize,
    /// `None` means the empty clause was found (proof terminates).
    /// `Some(Forward(..))` means the clause was deleted.
    /// `Some(Backward(..))` means the clause survived and was transferred to P.
    pub outcome: Option<SimplificationOutcome>,
}

/// Result of forward simplification on a clause from N.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum SimplificationOutcome {
    /// Clause eliminated by forward simplification (deleted, not transferred).
    Forward(ForwardSimplification),
    /// Clause survived forward simplification; backward effects applied, then transferred to P.
    Backward { effects: Vec<BackwardSimplification> },
}

/// A forward simplification that deletes a clause.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ForwardSimplification {
    Tautology,
    Demodulation { demodulator: usize, result: usize },
    Subsumption { subsumer: usize },
}

/// A backward simplification effect on existing clauses in P or A.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum BackwardSimplification {
    Subsumption { deleted_clause: usize },
    Demodulation { old_clause: usize, result: usize },
}

/// A generating inference that produced a new clause.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum GeneratingInference {
    Resolution { clause_idx: usize, parents: (usize, usize) },
    Factoring { clause_idx: usize, parent: usize },
    Superposition { clause_idx: usize, parents: (usize, usize) },
    EqualityResolution { clause_idx: usize, parent: usize },
    EqualityFactoring { clause_idx: usize, parent: usize },
}

impl GeneratingInference {
    /// Convert a `Derivation` (from an inference result) into a `GeneratingInference`.
    ///
    /// Panics if the derivation is `Input` or `Demodulation` (not generating inferences).
    pub fn from_derivation(clause_idx: usize, d: &Derivation) -> Self {
        match d {
            Derivation::Resolution { parent1, parent2 } => {
                Self::Resolution { clause_idx, parents: (*parent1, *parent2) }
            }
            Derivation::Factoring { parent } => {
                Self::Factoring { clause_idx, parent: *parent }
            }
            Derivation::Superposition { parent1, parent2 } => {
                Self::Superposition { clause_idx, parents: (*parent1, *parent2) }
            }
            Derivation::EqualityResolution { parent } => {
                Self::EqualityResolution { clause_idx, parent: *parent }
            }
            Derivation::EqualityFactoring { parent } => {
                Self::EqualityFactoring { clause_idx, parent: *parent }
            }
            _ => unreachable!("Input/Demodulation are not generating inferences"),
        }
    }
}

/// Complete trace of a saturation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaturationTrace {
    /// All clauses generated during saturation (indexed by clause_idx).
    pub clauses: Vec<String>,
    /// Number of clauses from the initial input.
    pub initial_clause_count: usize,
    /// One entry per iteration of the given-clause loop.
    pub iterations: Vec<SaturationStep>,
}
