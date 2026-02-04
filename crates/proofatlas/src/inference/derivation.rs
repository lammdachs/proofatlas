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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derivation_input() {
        let input = Derivation::input();
        assert_eq!(input.rule_name, "Input");
        assert!(input.premises.is_empty());
    }

    #[test]
    fn test_derivation_direct_construction() {
        let res = Derivation {
            rule_name: "Resolution".into(),
            premises: vec![1, 2],
        };
        assert_eq!(res.rule_name, "Resolution");
        assert_eq!(res.premises, vec![1, 2]);

        let demod = Derivation {
            rule_name: "Demodulation".into(),
            premises: vec![5, 10],
        };
        assert_eq!(demod.rule_name, "Demodulation");
        assert_eq!(demod.premises, vec![5, 10]);
    }

    #[test]
    fn test_serialization() {
        let deriv = Derivation {
            rule_name: "Resolution".into(),
            premises: vec![1, 2],
        };
        let json = serde_json::to_string(&deriv).unwrap();
        let parsed: Derivation = serde_json::from_str(&json).unwrap();
        assert_eq!(deriv, parsed);
    }
}
