//! ONNX-based inference for clause selection
//!
//! This module provides ML-based clause scoring using a trained model
//! exported to ONNX format. Uses tract-onnx for pure Rust inference,
//! which is compatible with WASM.
//!
//! The scorer takes a set of clauses and scores them all together,
//! allowing the model to consider clause interactions (e.g., which
//! clauses can resolve together, which are redundant).

#[cfg(feature = "onnx")]
use tract_onnx::prelude::*;

use crate::core::Clause;
#[cfg(feature = "onnx")]
use crate::ml::{GraphBuilder, FEATURE_DIM};
use std::path::Path;

/// Error types for ML inference
#[derive(Debug)]
pub enum InferenceError {
    ModelNotLoaded,
    SessionError(String),
    InferenceError(String),
    EmptyClauseSet,
}

impl std::fmt::Display for InferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferenceError::ModelNotLoaded => write!(f, "Model not loaded"),
            InferenceError::SessionError(e) => write!(f, "ONNX session error: {}", e),
            InferenceError::InferenceError(e) => write!(f, "Inference error: {}", e),
            InferenceError::EmptyClauseSet => write!(f, "Cannot score empty clause set"),
        }
    }
}

impl std::error::Error for InferenceError {}

/// Type alias for the tract model
#[cfg(feature = "onnx")]
type TractModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// ML-based clause scorer using ONNX models via tract.
///
/// Scores all clauses in a set together, allowing the model to consider
/// context and clause interactions. The score for each clause may depend
/// on all other clauses in the set.
pub struct ClauseScorer {
    #[cfg(feature = "onnx")]
    model: Option<TractModel>,
    #[cfg(not(feature = "onnx"))]
    _phantom: std::marker::PhantomData<()>,
}

impl ClauseScorer {
    /// Create a new clause scorer (without loading a model)
    pub fn new() -> Self {
        ClauseScorer {
            #[cfg(feature = "onnx")]
            model: None,
            #[cfg(not(feature = "onnx"))]
            _phantom: std::marker::PhantomData,
        }
    }

    /// Load an ONNX model from a file
    #[cfg(feature = "onnx")]
    pub fn load_model<P: AsRef<Path>>(&mut self, model_path: P) -> Result<(), InferenceError> {
        // Load the model without specifying input facts - let tract infer them from the model
        // The model is exported with fixed shapes, tract will handle dynamic shapes at runtime
        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .map_err(|e| InferenceError::SessionError(e.to_string()))?
            .into_optimized()
            .map_err(|e| InferenceError::SessionError(e.to_string()))?
            .into_runnable()
            .map_err(|e| InferenceError::SessionError(e.to_string()))?;

        self.model = Some(model);
        Ok(())
    }

    #[cfg(not(feature = "onnx"))]
    pub fn load_model<P: AsRef<Path>>(&mut self, _model_path: P) -> Result<(), InferenceError> {
        Err(InferenceError::SessionError(
            "onnx feature not enabled".to_string(),
        ))
    }

    /// Load an ONNX model from bytes (for WASM where file system is not available)
    #[cfg(feature = "onnx")]
    pub fn load_model_from_bytes(&mut self, model_bytes: &[u8]) -> Result<(), InferenceError> {
        let model = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(model_bytes))
            .map_err(|e| InferenceError::SessionError(e.to_string()))?
            .into_optimized()
            .map_err(|e| InferenceError::SessionError(e.to_string()))?
            .into_runnable()
            .map_err(|e| InferenceError::SessionError(e.to_string()))?;

        self.model = Some(model);
        Ok(())
    }

    #[cfg(not(feature = "onnx"))]
    pub fn load_model_from_bytes(&mut self, _model_bytes: &[u8]) -> Result<(), InferenceError> {
        Err(InferenceError::SessionError(
            "onnx feature not enabled".to_string(),
        ))
    }

    /// Check if a model is loaded
    pub fn is_model_loaded(&self) -> bool {
        #[cfg(feature = "onnx")]
        {
            self.model.is_some()
        }
        #[cfg(not(feature = "onnx"))]
        {
            false
        }
    }

    /// Score a set of clauses together.
    ///
    /// The score for each clause depends on all other clauses in the set,
    /// allowing the model to learn clause interactions.
    ///
    /// Returns a score for each clause where higher values indicate the
    /// clause is more likely to be useful for finding a proof.
    ///
    /// # Arguments
    /// * `clauses` - Slice of clauses to score together
    ///
    /// # Returns
    /// * `Ok(Vec<f32>)` - One score per clause, in the same order as input
    /// * `Err(InferenceError)` - If model not loaded or inference fails
    #[cfg(feature = "onnx")]
    pub fn score_clauses(&self, clauses: &[&Clause]) -> Result<Vec<f32>, InferenceError> {
        // Use default max_age of 1000
        self.score_clauses_with_context(clauses, 1000)
    }

    /// Score clauses with context information for age normalization.
    ///
    /// # Arguments
    /// * `clauses` - Slice of clauses to score together
    /// * `max_age` - Maximum age for normalization (clause ages divided by this)
    #[cfg(feature = "onnx")]
    pub fn score_clauses_with_context(
        &self,
        clauses: &[&Clause],
        max_age: usize,
    ) -> Result<Vec<f32>, InferenceError> {
        if clauses.is_empty() {
            return Err(InferenceError::EmptyClauseSet);
        }

        let model = self.model.as_ref().ok_or(InferenceError::ModelNotLoaded)?;

        // Build graphs for all clauses and concatenate node features
        let mut all_node_features: Vec<f32> = Vec::new();
        let mut clause_node_counts: Vec<usize> = Vec::new();

        for clause in clauses {
            let graph = GraphBuilder::build_from_clause_with_context(clause, max_age);
            clause_node_counts.push(graph.num_nodes);

            // Flatten node features
            for features in &graph.node_features {
                all_node_features.extend_from_slice(features);
            }
        }

        let total_nodes = clause_node_counts.iter().sum::<usize>();
        let num_clauses = clauses.len();

        // Build pool matrix [num_clauses, total_nodes]
        // Each row has 1/num_nodes for nodes in that clause, 0 elsewhere
        let mut pool_matrix_data = vec![0.0f32; num_clauses * total_nodes];
        let mut current_start = 0usize;
        for (clause_idx, &num_nodes) in clause_node_counts.iter().enumerate() {
            if num_nodes > 0 {
                let weight = 1.0 / num_nodes as f32;
                for node_idx in current_start..current_start + num_nodes {
                    pool_matrix_data[clause_idx * total_nodes + node_idx] = weight;
                }
            }
            current_start += num_nodes;
        }

        // Create tract tensors
        let node_features_tensor: Tensor =
            tract_ndarray::Array2::from_shape_vec((total_nodes, FEATURE_DIM), all_node_features)
                .map_err(|e| InferenceError::InferenceError(e.to_string()))?
                .into();

        let pool_matrix_tensor: Tensor =
            tract_ndarray::Array2::from_shape_vec((num_clauses, total_nodes), pool_matrix_data)
                .map_err(|e| InferenceError::InferenceError(e.to_string()))?
                .into();

        // Run inference
        let outputs = model
            .run(tvec!(
                node_features_tensor.into(),
                pool_matrix_tensor.into()
            ))
            .map_err(|e| InferenceError::InferenceError(e.to_string()))?;

        // Extract scores
        let scores_tensor = outputs[0]
            .to_array_view::<f32>()
            .map_err(|e| InferenceError::InferenceError(e.to_string()))?;

        // Apply sigmoid to get probabilities
        let scores: Vec<f32> = scores_tensor
            .iter()
            .map(|&logit| 1.0 / (1.0 + (-logit).exp()))
            .collect();

        Ok(scores)
    }

    #[cfg(not(feature = "onnx"))]
    pub fn score_clauses(&self, _clauses: &[&Clause]) -> Result<Vec<f32>, InferenceError> {
        Err(InferenceError::ModelNotLoaded)
    }

    #[cfg(not(feature = "onnx"))]
    pub fn score_clauses_with_context(
        &self,
        _clauses: &[&Clause],
        _max_age: usize,
    ) -> Result<Vec<f32>, InferenceError> {
        Err(InferenceError::ModelNotLoaded)
    }
}

impl Default for ClauseScorer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Atom, Clause, Literal, PredicateSymbol, Term, Variable};

    fn make_test_clause(name: &str) -> Clause {
        let p = PredicateSymbol {
            name: name.to_string(),
            arity: 1,
        };
        let x = Term::Variable(Variable {
            name: "X".to_string(),
        });
        Clause::new(vec![Literal::positive(Atom {
            predicate: p,
            args: vec![x],
        })])
    }

    #[test]
    fn test_scorer_without_model() {
        let scorer = ClauseScorer::new();
        assert!(!scorer.is_model_loaded());
    }

    #[test]
    #[cfg(feature = "onnx")]
    fn test_scorer_with_nonexistent_model() {
        let mut scorer = ClauseScorer::new();
        let result = scorer.load_model("/nonexistent/model.onnx");
        assert!(result.is_err());
    }

    #[test]
    fn test_score_without_model() {
        let scorer = ClauseScorer::new();
        let clause = make_test_clause("P");
        let result = scorer.score_clauses(&[&clause]);
        assert!(result.is_err());
    }

    #[test]
    fn test_score_empty_set() {
        let scorer = ClauseScorer::new();
        // Even without a model, empty set should give EmptyClauseSet error
        // (though actually it will give ModelNotLoaded first in current impl)
        let result = scorer.score_clauses(&[]);
        assert!(result.is_err());
    }
}
