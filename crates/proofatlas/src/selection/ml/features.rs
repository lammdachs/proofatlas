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

// =============================================================================
// FeaturesEmbedder — combined model, returns scores as 1-D "embeddings"
// =============================================================================

/// Features embedder using PyTorch for inference.
///
/// Loads a TorchScript model that takes `[N, 9]` clause features and
/// outputs scores directly. Scores are treated as 1-dimensional
/// "embeddings" for caching purposes (same pattern as GcnEmbedder).
#[cfg(feature = "ml")]
pub struct FeaturesEmbedder {
    model: tch::CModule,
    device: tch::Device,
}

#[cfg(feature = "ml")]
impl FeaturesEmbedder {
    /// Create a new features embedder from a TorchScript model.
    pub fn new<P: AsRef<std::path::Path>>(model_path: P, use_cuda: bool) -> Result<Self, String> {
        let device = if use_cuda && tch::Cuda::is_available() {
            tch::Device::Cuda(0)
        } else {
            tch::Device::Cpu
        };

        let model = tch::CModule::load_on_device(model_path.as_ref(), device)
            .map_err(|e| format!("Failed to load TorchScript features model: {}", e))?;

        Ok(Self { model, device })
    }

    /// Run the model on a features tensor and return per-clause results.
    fn forward_tensor(&self, features_tensor: tch::Tensor, n: usize) -> Vec<Vec<f32>> {
        let output = tch::no_grad(|| {
            self.model
                .forward_ts(&[features_tensor])
                .expect("Features forward failed")
        });

        let output_cpu = output.to_device(tch::Device::Cpu).view([-1]);
        let scores: Vec<f32> =
            Vec::<f32>::try_from(&output_cpu).expect("Failed to convert scores to Vec<f32>");

        scores.iter().take(n).map(|&s| vec![s]).collect()
    }
}

#[cfg(feature = "ml")]
impl crate::selection::cached::ClauseEmbedder for FeaturesEmbedder {
    fn embed_batch(&self, clauses: &[&Clause]) -> Vec<Vec<f32>> {
        if clauses.is_empty() {
            return vec![];
        }

        let n = clauses.len();
        let mut feat_flat = Vec::with_capacity(n * NUM_CLAUSE_FEATURES);
        for clause in clauses {
            feat_flat.extend_from_slice(&extract_clause_features(clause));
        }
        let features_tensor = tch::Tensor::from_slice(&feat_flat)
            .view([n as i64, NUM_CLAUSE_FEATURES as i64])
            .to_device(self.device);

        self.forward_tensor(features_tensor, n)
    }

    fn embed_features(&self, features: &[&[f32]]) -> Vec<Vec<f32>> {
        if features.is_empty() {
            return vec![];
        }

        let n = features.len();
        let feat_flat: Vec<f32> = features.iter().flat_map(|f| f.iter().copied()).collect();
        let features_tensor = tch::Tensor::from_slice(&feat_flat)
            .view([n as i64, NUM_CLAUSE_FEATURES as i64])
            .to_device(self.device);

        self.forward_tensor(features_tensor, n)
    }

    fn embedding_dim(&self) -> usize {
        1 // Scores are treated as 1-dim embeddings
    }

    fn name(&self) -> &str {
        "features"
    }
}

/// Load a standalone features embedder.
#[cfg(feature = "ml")]
pub fn load_features_embedder<P: AsRef<std::path::Path>>(
    model_path: P,
    use_cuda: bool,
) -> Result<FeaturesEmbedder, String> {
    FeaturesEmbedder::new(model_path, use_cuda)
}

// =============================================================================
// FeaturesEncoder — encoder-only, returns hidden_dim embeddings
// =============================================================================

/// Features encoder that returns real embeddings (not scores).
///
/// Loads an encoder-only TorchScript model (exported from model.encode()).
/// Returns `[N, hidden_dim]` embeddings for caching. Used with
/// `TorchScriptScorer` for separated encoder/scorer architecture.
#[cfg(feature = "ml")]
pub struct FeaturesEncoder {
    model: tch::CModule,
    device: tch::Device,
    hidden_dim: usize,
}

#[cfg(feature = "ml")]
impl FeaturesEncoder {
    /// Create a new features encoder from a TorchScript encoder model.
    pub fn new<P: AsRef<std::path::Path>>(
        model_path: P,
        hidden_dim: usize,
        use_cuda: bool,
    ) -> Result<Self, String> {
        let device = if use_cuda && tch::Cuda::is_available() {
            tch::Device::Cuda(0)
        } else {
            tch::Device::Cpu
        };

        let model = tch::CModule::load_on_device(model_path.as_ref(), device)
            .map_err(|e| format!("Failed to load TorchScript features encoder: {}", e))?;

        Ok(Self {
            model,
            device,
            hidden_dim,
        })
    }

    /// Run the encoder on a features tensor and return per-clause embeddings.
    fn forward_tensor(&self, features_tensor: tch::Tensor, n: usize) -> Vec<Vec<f32>> {
        let output = tch::no_grad(|| {
            self.model
                .forward_ts(&[features_tensor])
                .expect("Features encoder forward failed")
        });

        let output_cpu = output.to_device(tch::Device::Cpu);
        let flat: Vec<f32> =
            Vec::<f32>::try_from(&output_cpu.view([-1])).expect("Failed to convert embeddings");

        flat.chunks(self.hidden_dim)
            .take(n)
            .map(|chunk| chunk.to_vec())
            .collect()
    }
}

#[cfg(feature = "ml")]
impl crate::selection::cached::ClauseEmbedder for FeaturesEncoder {
    fn embed_batch(&self, clauses: &[&Clause]) -> Vec<Vec<f32>> {
        if clauses.is_empty() {
            return vec![];
        }

        let n = clauses.len();
        let mut feat_flat = Vec::with_capacity(n * NUM_CLAUSE_FEATURES);
        for clause in clauses {
            feat_flat.extend_from_slice(&extract_clause_features(clause));
        }
        let features_tensor = tch::Tensor::from_slice(&feat_flat)
            .view([n as i64, NUM_CLAUSE_FEATURES as i64])
            .to_device(self.device);

        self.forward_tensor(features_tensor, n)
    }

    fn embed_features(&self, features: &[&[f32]]) -> Vec<Vec<f32>> {
        if features.is_empty() {
            return vec![];
        }

        let n = features.len();
        let feat_flat: Vec<f32> = features.iter().flat_map(|f| f.iter().copied()).collect();
        let features_tensor = tch::Tensor::from_slice(&feat_flat)
            .view([n as i64, NUM_CLAUSE_FEATURES as i64])
            .to_device(self.device);

        self.forward_tensor(features_tensor, n)
    }

    fn embedding_dim(&self) -> usize {
        self.hidden_dim
    }

    fn name(&self) -> &str {
        "features_encoder"
    }
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
