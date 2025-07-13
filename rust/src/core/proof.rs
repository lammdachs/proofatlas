//! Proof tracking structures

use crate::inference::InferenceResult;

/// A single step in a proof
#[derive(Debug, Clone)]
pub struct ProofStep {
    pub inference: InferenceResult,
    pub clause_idx: usize,
}

/// A proof is a sequence of inference steps
#[derive(Debug, Clone)]
pub struct Proof {
    pub steps: Vec<ProofStep>,
    pub empty_clause_idx: usize,
}