//! Burn-based GCN clause selector
//!
//! This implements a Graph Convolutional Network (GCN) for clause selection,
//! using the Burn ML framework for inference. Weights are loaded from
//! safetensors files exported from PyTorch training.

use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, Recorder};
use burn::tensor::activation::relu;
use burn_import::safetensors::{AdapterType, LoadArgs, SafetensorsFileRecorder};
use std::collections::VecDeque;
use std::path::Path;

use crate::core::Clause;
use crate::ml::{GraphBuilder, FEATURE_DIM};

use super::ClauseSelector;

/// GCN layer: h' = σ(A · h · W)
/// Where A is the normalized adjacency matrix
#[derive(Module, Debug)]
pub struct GcnLayer<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> GcnLayer<B> {
    pub fn new(device: &B::Device, in_dim: usize, out_dim: usize) -> Self {
        let linear = LinearConfig::new(in_dim, out_dim).init(device);
        Self { linear }
    }

    /// Forward pass: A @ x @ W
    pub fn forward(&self, x: Tensor<B, 2>, adj: Tensor<B, 2>) -> Tensor<B, 2> {
        // Message passing: aggregate neighbor features
        let h = adj.matmul(x);
        // Linear transformation
        self.linear.forward(h)
    }
}

/// MLP scorer head for clause scoring
#[derive(Module, Debug)]
pub struct ScorerHead<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
}

impl<B: Backend> ScorerHead<B> {
    pub fn new(device: &B::Device, hidden_dim: usize) -> Self {
        let linear1 = LinearConfig::new(hidden_dim, hidden_dim).init(device);
        let linear2 = LinearConfig::new(hidden_dim, 1).init(device);
        Self { linear1, linear2 }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.linear1.forward(x);
        let h = relu(h);
        // Note: dropout is only used during training, not inference
        self.linear2.forward(h)
    }
}

/// GCN-based clause selection model
///
/// Architecture:
///   node_features → GCN layers → LayerNorm → ReLU → pool → scorer → logits
#[derive(Module, Debug)]
pub struct GcnModel<B: Backend> {
    /// GCN convolutional layers
    convs: Vec<GcnLayer<B>>,
    /// Layer normalization after each conv
    norms: Vec<LayerNorm<B>>,
    /// Scoring head MLP
    scorer: ScorerHead<B>,
    /// Hidden dimension
    hidden_dim: usize,
}

impl<B: Backend> GcnModel<B> {
    /// Create a new GCN model with random initialization
    pub fn new(
        device: &B::Device,
        input_dim: usize,
        hidden_dim: usize,
        num_layers: usize,
    ) -> Self {
        let mut convs = Vec::new();
        let mut norms = Vec::new();

        // First layer: input_dim -> hidden_dim
        convs.push(GcnLayer::new(device, input_dim, hidden_dim));
        norms.push(LayerNormConfig::new(hidden_dim).init(device));

        // Remaining layers: hidden_dim -> hidden_dim
        for _ in 1..num_layers {
            convs.push(GcnLayer::new(device, hidden_dim, hidden_dim));
            norms.push(LayerNormConfig::new(hidden_dim).init(device));
        }

        let scorer = ScorerHead::new(device, hidden_dim);

        Self {
            convs,
            norms,
            scorer,
            hidden_dim,
        }
    }

    /// Load model weights from a safetensors file exported from PyTorch
    ///
    /// The safetensors file should contain weights with PyTorch naming:
    /// - `convs.{i}.linear.weight` / `convs.{i}.linear.bias` for GCN layers
    /// - `norms.{i}.weight` / `norms.{i}.bias` for LayerNorm
    /// - `scorer.0.weight` / `scorer.0.bias` for first scorer linear
    /// - `scorer.3.weight` / `scorer.3.bias` for second scorer linear
    ///
    /// # Arguments
    /// * `path` - Path to the safetensors file
    /// * `device` - Device to load the model on
    /// * `input_dim` - Input feature dimension (default 13)
    /// * `hidden_dim` - Hidden layer dimension
    /// * `num_layers` - Number of GCN layers
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
    /// * `adj` - Normalized adjacency matrix [total_nodes, total_nodes]
    /// * `pool_matrix` - [num_clauses, total_nodes] for mean pooling
    ///
    /// # Returns
    /// * Logits [num_clauses, 1]
    pub fn forward(
        &self,
        node_features: Tensor<B, 2>,
        adj: Tensor<B, 2>,
        pool_matrix: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let num_layers = self.convs.len();
        let mut x = node_features;

        // Apply GCN layers with LayerNorm and ReLU
        for (i, (conv, norm)) in self.convs.iter().zip(self.norms.iter()).enumerate() {
            x = conv.forward(x, adj.clone());
            x = norm.forward(x);
            x = relu(x);
            // No dropout during inference (would be applied during training only)
            let _ = i < num_layers - 1; // placeholder for dropout logic
        }

        // Pool to clause embeddings: [num_clauses, hidden_dim]
        let clause_emb = pool_matrix.matmul(x);

        // Score clauses: [num_clauses, 1]
        self.scorer.forward(clause_emb)
    }
}

/// Burn-based GCN clause selector
pub struct BurnGcnSelector<B: Backend> {
    model: GcnModel<B>,
    device: B::Device,
    /// Maximum age for normalization
    max_age: usize,
    /// Random number generator state (simple LCG)
    rng_state: u64,
}

impl<B: Backend> BurnGcnSelector<B> {
    /// Create a new GCN selector with the given model
    pub fn new(model: GcnModel<B>, device: B::Device) -> Self {
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

    /// Score clauses using the GCN model
    fn score_clauses(&self, clauses: &[&Clause]) -> Vec<f32> {
        if clauses.is_empty() {
            return Vec::new();
        }

        // Build graphs and collect node features
        let mut all_node_features: Vec<f32> = Vec::new();
        let mut clause_node_counts: Vec<usize> = Vec::new();
        let mut all_edges: Vec<(usize, usize)> = Vec::new();
        let mut node_offset: usize = 0;

        for clause in clauses {
            let graph = GraphBuilder::build_from_clause_with_context(clause, self.max_age);
            clause_node_counts.push(graph.num_nodes);

            // Flatten node features
            for features in &graph.node_features {
                all_node_features.extend_from_slice(features);
            }

            // Offset edges and add
            for &(src, dst) in &graph.edge_indices {
                all_edges.push((src + node_offset, dst + node_offset));
            }

            node_offset += graph.num_nodes;
        }

        let total_nodes = clause_node_counts.iter().sum::<usize>();
        let num_clauses = clauses.len();

        // Build adjacency matrix with self-loops and normalize
        let mut adj_data = vec![0.0f32; total_nodes * total_nodes];

        // Add self-loops
        for i in 0..total_nodes {
            adj_data[i * total_nodes + i] = 1.0;
        }

        // Add edges (undirected)
        for &(src, dst) in &all_edges {
            adj_data[src * total_nodes + dst] = 1.0;
            adj_data[dst * total_nodes + src] = 1.0;
        }

        // Compute degree and normalize (symmetric normalization)
        let mut degrees: Vec<f32> = vec![0.0; total_nodes];
        for i in 0..total_nodes {
            for j in 0..total_nodes {
                degrees[i] += adj_data[i * total_nodes + j];
            }
        }

        // D^{-1/2} A D^{-1/2}
        for i in 0..total_nodes {
            for j in 0..total_nodes {
                if adj_data[i * total_nodes + j] > 0.0 {
                    let norm = (degrees[i] * degrees[j]).sqrt();
                    if norm > 0.0 {
                        adj_data[i * total_nodes + j] /= norm;
                    }
                }
            }
        }

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

        let adj_tensor = Tensor::<B, 1>::from_floats(adj_data.as_slice(), &self.device)
            .reshape([total_nodes, total_nodes]);

        let pool_tensor = Tensor::<B, 1>::from_floats(pool_matrix_data.as_slice(), &self.device)
            .reshape([num_clauses, total_nodes]);

        // Run model
        let logits = self.model.forward(node_features_tensor, adj_tensor, pool_tensor);

        // Extract scores
        let logits_data: Vec<f32> = logits.into_data().to_vec().unwrap();
        logits_data
    }
}

impl<B: Backend> ClauseSelector for BurnGcnSelector<B>
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
        "BurnGCN"
    }
}

// Convenience type aliases for common backends
pub type NdarrayGcnSelector = BurnGcnSelector<burn_ndarray::NdArray<f32>>;

/// Create a GCN selector with the ndarray backend (random initialization)
pub fn create_ndarray_gcn_selector(
    input_dim: usize,
    hidden_dim: usize,
    num_layers: usize,
) -> NdarrayGcnSelector {
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let model = GcnModel::new(&device, input_dim, hidden_dim, num_layers);
    BurnGcnSelector::new(model, device)
}

/// Load a GCN selector from a safetensors file (PyTorch format)
///
/// # Arguments
/// * `path` - Path to the safetensors file
/// * `input_dim` - Input feature dimension (default 13 for clause graphs)
/// * `hidden_dim` - Hidden layer dimension
/// * `num_layers` - Number of GCN layers
///
/// # Returns
/// * `Ok(NdarrayGcnSelector)` if weights load successfully
/// * `Err(String)` with error message if loading fails
pub fn load_ndarray_gcn_selector<P: AsRef<Path>>(
    path: P,
    input_dim: usize,
    hidden_dim: usize,
    num_layers: usize,
) -> Result<NdarrayGcnSelector, String> {
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let model = GcnModel::load_from_safetensors(path, &device, input_dim, hidden_dim, num_layers)?;
    Ok(BurnGcnSelector::new(model, device))
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
    fn test_gcn_model_forward() {
        let device = burn_ndarray::NdArrayDevice::Cpu;
        let model: GcnModel<burn_ndarray::NdArray<f32>> = GcnModel::new(&device, 13, 64, 3);

        // Create simple input
        let node_features = Tensor::zeros([5, 13], &device);
        let adj = Tensor::ones([5, 5], &device) / 5.0f32;
        let pool = Tensor::ones([2, 5], &device) / 5.0f32;

        let output = model.forward(node_features, adj, pool);
        assert_eq!(output.dims(), [2, 1]);
    }

    #[test]
    fn test_gcn_selector_basic() {
        let mut selector = create_ndarray_gcn_selector(13, 64, 3);
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
    fn test_gcn_selector_empty() {
        let mut selector = create_ndarray_gcn_selector(13, 64, 3);
        let clauses: Vec<Clause> = vec![];
        let mut unprocessed: VecDeque<usize> = VecDeque::new();

        assert_eq!(selector.select(&mut unprocessed, &clauses), None);
    }
}
