//! Clause derivation tracking.
//!
//! Records how each clause was derived (inference rule + premises).

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
