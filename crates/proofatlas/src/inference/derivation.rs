//! Clause derivation tracking.
//!
//! Records how each clause was derived (inference rule + premises).

use serde::{Deserialize, Serialize};

/// How a clause was derived (for proofs and internal derivation tracking).
///
/// This is a dynamic struct that stores the rule name and premise indices,
/// allowing new rules to be added without modifying this type.
/// Rules construct Derivation directly:
/// ```ignore
/// Derivation { rule_name: "Resolution".into(), premises: vec![p1, p2] }
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Derivation {
    /// Name of the inference rule that produced this clause
    pub rule_name: String,
    /// Indices of the premise clauses used in the inference
    pub premises: Vec<usize>,
}

impl Derivation {
    /// Create an Input derivation (no premises)
    pub fn input() -> Self {
        Derivation {
            rule_name: "Input".into(),
            premises: vec![],
        }
    }

    /// Create a Resolution derivation
    pub fn resolution(parent1: usize, parent2: usize) -> Self {
        Derivation {
            rule_name: "Resolution".into(),
            premises: vec![parent1, parent2],
        }
    }

    /// Create a Factoring derivation
    pub fn factoring(parent: usize) -> Self {
        Derivation {
            rule_name: "Factoring".into(),
            premises: vec![parent],
        }
    }

    /// Create a Superposition derivation
    pub fn superposition(parent1: usize, parent2: usize) -> Self {
        Derivation {
            rule_name: "Superposition".into(),
            premises: vec![parent1, parent2],
        }
    }

    /// Create an EqualityResolution derivation
    pub fn equality_resolution(parent: usize) -> Self {
        Derivation {
            rule_name: "EqualityResolution".into(),
            premises: vec![parent],
        }
    }

    /// Create an EqualityFactoring derivation
    pub fn equality_factoring(parent: usize) -> Self {
        Derivation {
            rule_name: "EqualityFactoring".into(),
            premises: vec![parent],
        }
    }

    /// Create a Demodulation derivation
    pub fn demodulation(demodulator: usize, target: usize) -> Self {
        Derivation {
            rule_name: "Demodulation".into(),
            premises: vec![demodulator, target],
        }
    }

    /// Check if this is an Input derivation
    pub fn is_input(&self) -> bool {
        self.rule_name == "Input"
    }

    /// Check if this is a Demodulation derivation
    pub fn is_demodulation(&self) -> bool {
        self.rule_name == "Demodulation"
    }

    /// Get the demodulator index (for Demodulation derivations)
    /// Returns None if not a Demodulation
    pub fn demodulator(&self) -> Option<usize> {
        if self.rule_name == "Demodulation" && self.premises.len() >= 1 {
            Some(self.premises[0])
        } else {
            None
        }
    }

    /// Get the target index (for Demodulation derivations)
    /// Returns None if not a Demodulation
    pub fn target(&self) -> Option<usize> {
        if self.rule_name == "Demodulation" && self.premises.len() >= 2 {
            Some(self.premises[1])
        } else {
            None
        }
    }

    /// Get the premise clause indices (for backwards compatibility)
    pub fn premises(&self) -> Vec<usize> {
        self.premises.clone()
    }

    /// Get a human-readable rule name (for backwards compatibility)
    pub fn rule_name(&self) -> &str {
        &self.rule_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derivation_constructors() {
        let input = Derivation::input();
        assert_eq!(input.rule_name, "Input");
        assert!(input.premises.is_empty());
        assert!(input.is_input());

        let res = Derivation::resolution(1, 2);
        assert_eq!(res.rule_name, "Resolution");
        assert_eq!(res.premises, vec![1, 2]);

        let demod = Derivation::demodulation(5, 10);
        assert!(demod.is_demodulation());
        assert_eq!(demod.demodulator(), Some(5));
        assert_eq!(demod.target(), Some(10));
    }

    #[test]
    fn test_serialization() {
        let deriv = Derivation::resolution(1, 2);
        let json = serde_json::to_string(&deriv).unwrap();
        let parsed: Derivation = serde_json::from_str(&json).unwrap();
        assert_eq!(deriv, parsed);
    }
}
