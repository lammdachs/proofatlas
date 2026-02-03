//! Proof tracking structures

use crate::core::Clause;
use crate::core::Derivation;
use crate::inference::InferenceResult;

/// A single step in a proof derivation. Every step produces a clause.
#[derive(Debug, Clone)]
pub struct ProofStep {
    pub clause_idx: usize,
    pub derivation: Derivation,
    pub conclusion: Clause,
}

impl ProofStep {
    /// Create from an inference result
    pub fn from_inference(clause_idx: usize, result: InferenceResult) -> Self {
        ProofStep {
            clause_idx,
            derivation: result.derivation,
            conclusion: result.conclusion,
        }
    }
}

/// A proof is a sequence of inference steps
#[derive(Debug, Clone)]
pub struct Proof {
    pub steps: Vec<ProofStep>,
    pub empty_clause_idx: usize,
    /// All clauses generated during saturation (for ML training data extraction)
    pub all_clauses: Vec<Clause>,
}
