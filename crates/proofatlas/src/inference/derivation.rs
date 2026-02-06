//! Re-export from crate::state for backward compatibility
pub use crate::state::Derivation;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::Position;

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
            premises: vec![Position::clause(1), Position::clause(2)],
        };
        assert_eq!(res.rule_name, "Resolution");
        assert_eq!(res.clause_indices(), vec![1, 2]);

        let demod = Derivation {
            rule_name: "Demodulation".into(),
            premises: vec![Position::clause(5), Position::clause(10)],
        };
        assert_eq!(demod.rule_name, "Demodulation");
        assert_eq!(demod.clause_indices(), vec![5, 10]);
    }

    #[test]
    fn test_serialization() {
        let deriv = Derivation {
            rule_name: "Resolution".into(),
            premises: vec![Position::clause(1), Position::clause(2)],
        };
        let json = serde_json::to_string(&deriv).unwrap();
        let parsed: Derivation = serde_json::from_str(&json).unwrap();
        assert_eq!(deriv, parsed);
    }
}
