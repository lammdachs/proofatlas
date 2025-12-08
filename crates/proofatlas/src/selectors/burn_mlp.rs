//! Burn-based MLP clause selector
//!
//! This implements a simple MLP baseline for clause selection that doesn't
//! use graph structure - it just pools node features and scores clauses.
//! Uses the Burn ML framework for inference.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, Recorder};
use burn::tensor::activation::relu;
use burn_import::safetensors::{AdapterType, LoadArgs, SafetensorsFileRecorder};
use std::collections::VecDeque;
use std::path::Path;

use crate::core::Clause;
use crate::ml::{GraphBuilder, FEATURE_DIM};

use super::ClauseSelector;

/// MLP encoder for node features
#[derive(Module, Debug)]
pub struct MlpEncoder<B: Backend> {
    layers: Vec<Linear<B>>,
}

impl<B: Backend> MlpEncoder<B> {
    pub fn new(device: &B::Device, input_dim: usize, hidden_dim: usize, num_layers: usize) -> Self {
        let mut layers = Vec::new();

        // First layer: input_dim -> hidden_dim
        layers.push(LinearConfig::new(input_dim, hidden_dim).init(device));

        // Remaining layers: hidden_dim -> hidden_dim
        for _ in 1..num_layers {
            layers.push(LinearConfig::new(hidden_dim, hidden_dim).init(device));
        }

        Self { layers }
    }

    /// Forward pass with ReLU activations
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut h = x;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(h);
            // ReLU activation after each layer
            h = relu(h);
            // Note: dropout would be applied during training only
            let _ = i; // placeholder
        }
        h
    }
}

/// MLP-based clause selection model
///
/// Architecture:
///   node_features → MLP encoder → pool to clauses → score
#[derive(Module, Debug)]
pub struct MlpModel<B: Backend> {
    encoder: MlpEncoder<B>,
    scorer: Linear<B>,
    hidden_dim: usize,
}

impl<B: Backend> MlpModel<B> {
    /// Create a new MLP model with random initialization
    pub fn new(
        device: &B::Device,
        input_dim: usize,
        hidden_dim: usize,
        num_layers: usize,
    ) -> Self {
        let encoder = MlpEncoder::new(device, input_dim, hidden_dim, num_layers);
        let scorer = LinearConfig::new(hidden_dim, 1).init(device);

        Self {
            encoder,
            scorer,
            hidden_dim,
        }
    }

    /// Load model weights from a safetensors file exported from PyTorch
    ///
    /// The safetensors file should contain weights with PyTorch naming:
    /// - `encoder.{2*i}.weight` / `encoder.{2*i}.bias` for encoder layers
    /// - `scorer.weight` / `scorer.bias` for scorer
    ///
    /// # Arguments
    /// * `path` - Path to the safetensors file
    /// * `device` - Device to load the model on
    /// * `input_dim` - Input feature dimension (default 13)
    /// * `hidden_dim` - Hidden layer dimension
    /// * `num_layers` - Number of encoder layers
    ///
    /// # Returns
    /// * `Ok(Self)` if weights load successfully
    /// * `Err(String)` with error message if loading fails
    pub fn load_from_safetensors<P: AsRef<Path>>(
        path: P,
        device: &B::Device,
        input_dim: usize,
        hidden_dim: usize,
        num_layers: usize,
    ) -> Result<Self, String> {
        // First create a model with the correct architecture
        let model = Self::new(device, input_dim, hidden_dim, num_layers);

        // Load weights from safetensors with PyTorch adapter
        let load_args = LoadArgs::new(path.as_ref().into()).with_adapter_type(AdapterType::PyTorch);

        let record = SafetensorsFileRecorder::<FullPrecisionSettings>::default()
            .load(load_args, device)
            .map_err(|e| format!("Failed to load safetensors: {}", e))?;

        // Load the record into the model
        Ok(model.load_record(record))
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `node_features` - [total_nodes, input_dim]
    /// * `pool_matrix` - [num_clauses, total_nodes] for mean pooling
    ///
    /// # Returns
    /// * Logits [num_clauses, 1]
    pub fn forward(&self, node_features: Tensor<B, 2>, pool_matrix: Tensor<B, 2>) -> Tensor<B, 2> {
        // Encode node features
        let h = self.encoder.forward(node_features);

        // Pool to clause embeddings: [num_clauses, hidden_dim]
        let clause_emb = pool_matrix.matmul(h);

        // Score clauses: [num_clauses, 1]
        self.scorer.forward(clause_emb)
    }
}

/// Burn-based MLP clause selector
pub struct BurnMlpSelector<B: Backend> {
    model: MlpModel<B>,
    device: B::Device,
    /// Maximum age for normalization
    max_age: usize,
    /// Random number generator state (simple LCG)
    rng_state: u64,
}

impl<B: Backend> BurnMlpSelector<B> {
    /// Create a new MLP selector with the given model
    pub fn new(model: MlpModel<B>, device: B::Device) -> Self {
        Self {
            model,
            device,
            max_age: 1000,
            rng_state: 12345,
        }
    }

    /// Set the maximum age for normalization
    pub fn with_max_age(mut self, max_age: usize) -> Self {
        self.max_age = max_age;
        self
    }

    /// Generate a random float in [0, 1)
    fn next_random(&mut self) -> f64 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        (self.rng_state >> 33) as f64 / (1u64 << 31) as f64
    }

    /// Sample an index from logits using softmax sampling
    fn sample_from_logits(&mut self, logits: &[f32]) -> usize {
        if logits.is_empty() {
            return 0;
        }
        if logits.len() == 1 {
            return 0;
        }

        // Find max for numerical stability
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Compute softmax
        let exp_scores: Vec<f64> = logits
            .iter()
            .map(|&x| ((x - max_logit) as f64).exp())
            .collect();
        let sum: f64 = exp_scores.iter().sum();
        let probs: Vec<f64> = exp_scores.iter().map(|&e| e / sum).collect();

        // Sample from distribution
        let r = self.next_random();
        let mut cumsum = 0.0;

        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return i;
            }
        }

        logits.len() - 1
    }

    /// Score clauses using the MLP model
    fn score_clauses(&self, clauses: &[&Clause]) -> Vec<f32> {
        if clauses.is_empty() {
            return Vec::new();
        }

        // Build graphs and collect node features
        let mut all_node_features: Vec<f32> = Vec::new();
        let mut clause_node_counts: Vec<usize> = Vec::new();

        for clause in clauses {
            let graph = GraphBuilder::build_from_clause_with_context(clause, self.max_age);
            clause_node_counts.push(graph.num_nodes);

            // Flatten node features
            for features in &graph.node_features {
                all_node_features.extend_from_slice(features);
            }
        }

        let total_nodes = clause_node_counts.iter().sum::<usize>();
        let num_clauses = clauses.len();

        // Build pool matrix [num_clauses, total_nodes]
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

        // Create tensors
        let node_features_tensor =
            Tensor::<B, 1>::from_floats(all_node_features.as_slice(), &self.device)
                .reshape([total_nodes, FEATURE_DIM]);

        let pool_tensor = Tensor::<B, 1>::from_floats(pool_matrix_data.as_slice(), &self.device)
            .reshape([num_clauses, total_nodes]);

        // Run model
        let logits = self.model.forward(node_features_tensor, pool_tensor);

        // Extract scores
        let logits_data: Vec<f32> = logits.into_data().to_vec().unwrap();
        logits_data
    }
}

impl<B: Backend> ClauseSelector for BurnMlpSelector<B>
where
    B: Backend,
    B::Device: Clone,
{
    fn select(&mut self, unprocessed: &mut VecDeque<usize>, clauses: &[Clause]) -> Option<usize> {
        if unprocessed.is_empty() {
            return None;
        }

        // Collect clause references for scoring
        let clause_refs: Vec<&Clause> = unprocessed.iter().map(|&idx| &clauses[idx]).collect();

        // Score clauses
        let scores = self.score_clauses(&clause_refs);

        if scores.is_empty() {
            // Fallback to FIFO
            return unprocessed.pop_front();
        }

        // Sample from scores
        let selected_idx = self.sample_from_logits(&scores);

        // Remove and return the selected clause
        unprocessed.remove(selected_idx)
    }

    fn name(&self) -> &str {
        "BurnMLP"
    }
}

// Convenience type aliases for common backends
pub type NdarrayMlpSelector = BurnMlpSelector<burn_ndarray::NdArray<f32>>;

/// Create an MLP selector with the ndarray backend (random initialization)
pub fn create_ndarray_mlp_selector(
    input_dim: usize,
    hidden_dim: usize,
    num_layers: usize,
) -> NdarrayMlpSelector {
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let model = MlpModel::new(&device, input_dim, hidden_dim, num_layers);
    BurnMlpSelector::new(model, device)
}

/// Load an MLP selector from a safetensors file (PyTorch format)
///
/// # Arguments
/// * `path` - Path to the safetensors file
/// * `input_dim` - Input feature dimension (default 13 for clause graphs)
/// * `hidden_dim` - Hidden layer dimension
/// * `num_layers` - Number of encoder layers
///
/// # Returns
/// * `Ok(NdarrayMlpSelector)` if weights load successfully
/// * `Err(String)` with error message if loading fails
pub fn load_ndarray_mlp_selector<P: AsRef<Path>>(
    path: P,
    input_dim: usize,
    hidden_dim: usize,
    num_layers: usize,
) -> Result<NdarrayMlpSelector, String> {
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let model = MlpModel::load_from_safetensors(path, &device, input_dim, hidden_dim, num_layers)?;
    Ok(BurnMlpSelector::new(model, device))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Atom, Literal, PredicateSymbol, Term, Variable};

    fn make_test_clause(name: &str, num_args: usize) -> Clause {
        let args: Vec<Term> = (0..num_args)
            .map(|j| {
                Term::Variable(Variable {
                    name: format!("X{}", j),
                })
            })
            .collect();

        let atom = Atom {
            predicate: PredicateSymbol {
                name: name.to_string(),
                arity: num_args,
            },
            args,
        };

        Clause::new(vec![Literal::positive(atom)])
    }

    #[test]
    fn test_mlp_model_forward() {
        let device = burn_ndarray::NdArrayDevice::Cpu;
        let model: MlpModel<burn_ndarray::NdArray<f32>> = MlpModel::new(&device, 13, 64, 2);

        // Create simple input
        let node_features = Tensor::zeros([5, 13], &device);
        let pool = Tensor::ones([2, 5], &device) / 5.0f32;

        let output = model.forward(node_features, pool);
        assert_eq!(output.dims(), [2, 1]);
    }

    #[test]
    fn test_mlp_selector_basic() {
        let mut selector = create_ndarray_mlp_selector(13, 64, 2);
        let clauses = vec![
            make_test_clause("P", 2),
            make_test_clause("Q", 1),
            make_test_clause("R", 3),
        ];
        let mut unprocessed: VecDeque<usize> = (0..3).collect();

        // Should select something
        let selected = selector.select(&mut unprocessed, &clauses);
        assert!(selected.is_some());
        assert_eq!(unprocessed.len(), 2);
    }

    #[test]
    fn test_mlp_selector_empty() {
        let mut selector = create_ndarray_mlp_selector(13, 64, 2);
        let clauses: Vec<Clause> = vec![];
        let mut unprocessed: VecDeque<usize> = VecDeque::new();

        assert_eq!(selector.select(&mut unprocessed, &clauses), None);
    }
}
