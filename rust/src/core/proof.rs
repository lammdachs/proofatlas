//! Proof representation and tracking

use super::problem::Problem;

/// Inference rules used in proofs
#[derive(Debug, Clone, PartialEq)]
pub enum InferenceRule {
    Resolution { 
        lit1_idx: usize,  // Literal index in parent 1
        lit2_idx: usize,  // Literal index in parent 2
    },
    Factoring {
        lit_indices: Vec<usize>,  // Literals that were factored
    },
    Superposition {
        from_lit: usize,    // Equality literal s=t
        into_lit: usize,    // Target literal
        position: Vec<usize>,  // Position in term tree
        positive: bool,     // Positive or negative superposition
    },
    EqualityResolution,  // s ≠ s → ⊥
    EqualityFactoring,   // s=t ∨ s=t' → s=t ∨ t=t'
    ForwardSubsumption,  // New clause subsumes existing clauses
    BackwardSubsumption, // Existing clause subsumes new clause
}

/// A single proof step representing a state change
#[derive(Debug, Clone)]
pub struct ProofStep {
    /// Which rule was applied
    pub rule: InferenceRule,
    
    /// Parent clause indices
    pub parents: Vec<usize>,
    
    /// Which literals were selected in parents
    pub selected_literals: Vec<usize>,
    
    /// Clause moved from unprocessed to processed
    pub given_clause: Option<usize>,
    
    /// New clauses added to unprocessed
    pub added_clauses: Vec<usize>,
    
    /// Clauses removed by subsumption
    pub deleted_clauses: Vec<usize>,
}

/// Complete proof with all clauses and steps
#[derive(Debug)]
pub struct Proof {
    /// Contains ALL clauses ever derived
    pub problem: Problem,
    
    /// Sequence of state changes
    pub steps: Vec<ProofStep>,
    
    /// Step where clause first appears
    pub clause_first_step: Vec<usize>,
    
    /// Step where clause last appears (or deleted)
    pub clause_last_step: Vec<usize>,
}


/// Result of saturation
pub enum SaturationResult {
    Proof(Proof),
    Saturated,
    ResourceLimit,
}

#[cfg(test)]
#[path = "proof_tests.rs"]
mod tests;