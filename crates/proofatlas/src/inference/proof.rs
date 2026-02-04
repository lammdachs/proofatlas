//! Proof tracking structures

use crate::fol::{Clause, Interner};
use crate::json::ProofJson;
use super::derivation::Derivation;

/// A single step in a proof derivation. Every step produces a clause.
#[derive(Debug, Clone)]
pub struct ProofStep {
    pub clause_idx: usize,
    pub derivation: Derivation,
    pub conclusion: Clause,
}

/// A proof is a sequence of inference steps
#[derive(Debug, Clone)]
pub struct Proof {
    pub steps: Vec<ProofStep>,
    pub empty_clause_idx: usize,
    /// All clauses generated during saturation (for ML training data extraction)
    pub all_clauses: Vec<Clause>,
}

impl Proof {
    /// Convert to JSON representation
    pub fn to_json(&self, _interner: &Interner) -> ProofJson {
        // ProofJson doesn't need interner - it only contains step indices and rule names
        self.into()
    }
}
