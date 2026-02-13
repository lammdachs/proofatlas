//! Clause feature extraction for ML models.
//!
//! Provides a shared `extract_clause_features` function used by:
//! - GCN/GcnEncoder batch tensor construction
//! - FeaturesScoreProcessor / FeaturesEmbeddingProcessor in the pipeline
//! - Python bindings for training data extraction

use crate::logic::Clause;

/// Number of clause-level features.
pub const NUM_CLAUSE_FEATURES: usize = 9;

/// Extract 9 clause-level features for ML models.
///
/// Feature layout (must match Python `selectors/features.py` and training data):
/// - `[0]` age
/// - `[1]` role (0=axiom, 1=hypothesis, 2=definition, 3=negated_conjecture, 4=derived)
/// - `[2]` derivation rule (0=input, 1=resolution, ..., 6=demodulation)
/// - `[3]` number of literals
/// - `[4]` max term depth
/// - `[5]` total symbol count
/// - `[6]` distinct symbol count
/// - `[7]` total variable count
/// - `[8]` distinct variable count
pub fn extract_clause_features(clause: &Clause) -> [f32; NUM_CLAUSE_FEATURES] {
    [
        clause.age as f32,
        clause.role.to_feature_value(),
        clause.derivation_rule as f32,
        clause.literals.len() as f32,
        clause.max_depth() as f32,
        clause.symbol_count() as f32,
        clause.distinct_symbol_count() as f32,
        clause.variable_count() as f32,
        clause.distinct_variable_count() as f32,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::{Clause, ClauseRole, Literal, PredicateSymbol};
    use crate::logic::interner::PredicateId;

    #[test]
    fn test_extract_clause_features_empty() {
        let clause = Clause::new(vec![]);
        let features = extract_clause_features(&clause);
        assert_eq!(features[0], 0.0); // age
        assert_eq!(features[1], 0.0); // role (Axiom)
        assert_eq!(features[2], 0.0); // derivation_rule (input)
        assert_eq!(features[3], 0.0); // 0 literals
    }

    #[test]
    fn test_extract_clause_features_with_literals() {
        let p = PredicateSymbol {
            id: PredicateId(0),
            arity: 0,
        };
        let mut clause = Clause::new(vec![
            Literal::positive(p.clone(), vec![]),
            Literal::negative(p.clone(), vec![]),
        ]);
        clause.age = 42;
        clause.role = ClauseRole::NegatedConjecture;
        clause.derivation_rule = 1; // Resolution

        let features = extract_clause_features(&clause);
        assert_eq!(features[0], 42.0); // age
        assert_eq!(features[1], 3.0);  // NegatedConjecture
        assert_eq!(features[2], 1.0);  // Resolution
        assert_eq!(features[3], 2.0);  // 2 literals
    }
}
