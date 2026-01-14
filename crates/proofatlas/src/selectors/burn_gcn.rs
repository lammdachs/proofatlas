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
use std::collections::{HashMap, VecDeque};
use std::path::Path;

use crate::core::Clause;

use super::ClauseSelector;

/// Node feature embedding: 3-dim raw → 15-dim embedded
/// Matches PyTorch NodeFeatureEmbedding structure (div_term directly on struct)
#[derive(Module, Debug)]
pub struct NodeFeatureEmbedding<B: Backend> {
    /// Precomputed div_term for sinusoidal encoding
    div_term: Tensor<B, 1>,
    /// sin_dim for encoding
        sin_dim: usize,
    /// Output dimension: 6 (type one-hot) + 1 (log arity) + sin_dim (arg_pos)
        output_dim: usize,
}

impl<B: Backend> NodeFeatureEmbedding<B> {
    pub fn new(device: &B::Device, sin_dim: usize) -> Self {
        let half_dim = sin_dim / 2;
        let mut div_term_data = Vec::with_capacity(half_dim);
        let log_10000 = 10000.0_f32.ln();
        for i in 0..half_dim {
            let val = (2.0 * i as f32 * (-log_10000) / sin_dim as f32).exp();
            div_term_data.push(val);
        }
        let div_term = Tensor::from_floats(div_term_data.as_slice(), device);
        let output_dim = 6 + 1 + sin_dim;
        Self { div_term, sin_dim, output_dim }
    }

    /// Encode values using sinusoidal positional encoding
    fn sinusoidal_encode(&self, values: Tensor<B, 1>) -> Tensor<B, 2> {
        let n = values.dims()[0];
        let half_dim = self.sin_dim / 2;
        let values = values.reshape([n, 1]);
        let div_term = self.div_term.clone().reshape([1, half_dim]);
        let scaled = values.matmul(div_term);
        let sin_enc = scaled.clone().sin();
        let cos_enc = scaled.cos();
        let stacked = Tensor::stack::<3>(vec![sin_enc, cos_enc], 2);
        stacked.reshape([n, self.sin_dim])
    }

    /// Embed raw node features
    /// Input: [N, 3] (type, arity, arg_pos)
    /// Output: [N, output_dim]
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let device = x.device();
        let n = x.dims()[0];

        // Extract features
        let node_type = x.clone().slice([0..n, 0..1]).flatten(0, 1);
        let arity = x.clone().slice([0..n, 1..2]).flatten(0, 1);
        let arg_pos = x.slice([0..n, 2..3]).flatten(0, 1);

        // Node type to one-hot (6 classes)
        let node_type_int = node_type.int();
        let node_type_clamped = node_type_int.clamp(0, 5);
        let type_onehot = one_hot::<B>(node_type_clamped, 6, &device);

        // Arity: log1p scaled
        let ones = Tensor::<B, 1>::ones([n], &device);
        let arity_enc = (arity + ones).log().reshape([n, 1]);

        // Arg position: sinusoidal encoded
        let arg_pos_enc = self.sinusoidal_encode(arg_pos);

        Tensor::cat(vec![type_onehot, arity_enc, arg_pos_enc], 1)
    }
}

/// Clause feature embedding: 3-dim raw → 21-dim embedded
/// Matches PyTorch ClauseFeatureEmbedding structure (div_term directly on struct)
#[derive(Module, Debug)]
pub struct ClauseFeatureEmbedding<B: Backend> {
    /// Precomputed div_term for sinusoidal encoding
    div_term: Tensor<B, 1>,
    /// sin_dim for encoding
        sin_dim: usize,
    /// Output dimension: sin_dim (age) + 5 (role one-hot) + sin_dim (size)
        output_dim: usize,
}

impl<B: Backend> ClauseFeatureEmbedding<B> {
    pub fn new(device: &B::Device, sin_dim: usize) -> Self {
        let half_dim = sin_dim / 2;
        let mut div_term_data = Vec::with_capacity(half_dim);
        let log_10000 = 10000.0_f32.ln();
        for i in 0..half_dim {
            let val = (2.0 * i as f32 * (-log_10000) / sin_dim as f32).exp();
            div_term_data.push(val);
        }
        let div_term = Tensor::from_floats(div_term_data.as_slice(), device);
        let output_dim = sin_dim + 5 + sin_dim;
        Self { div_term, sin_dim, output_dim }
    }

    /// Encode values using sinusoidal positional encoding
    fn sinusoidal_encode(&self, values: Tensor<B, 1>) -> Tensor<B, 2> {
        let n = values.dims()[0];
        let half_dim = self.sin_dim / 2;
        let values = values.reshape([n, 1]);
        let div_term = self.div_term.clone().reshape([1, half_dim]);
        let scaled = values.matmul(div_term);
        let sin_enc = scaled.clone().sin();
        let cos_enc = scaled.cos();
        let stacked = Tensor::stack::<3>(vec![sin_enc, cos_enc], 2);
        stacked.reshape([n, self.sin_dim])
    }

    /// Embed raw clause features
    /// Input: [num_clauses, 3] (age, role, size)
    /// Output: [num_clauses, output_dim]
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let device = x.device();
        let n = x.dims()[0];

        // Extract features
        let age = x.clone().slice([0..n, 0..1]).flatten(0, 1);
        let role = x.clone().slice([0..n, 1..2]).flatten(0, 1);
        let size = x.slice([0..n, 2..3]).flatten(0, 1);

        // Age: sinusoidal encoded (scaled by 100)
        let age_scaled = age * 100.0;
        let age_enc = self.sinusoidal_encode(age_scaled);

        // Role: one-hot (5 classes)
        let role_int = role.int();
        let role_clamped = role_int.clamp(0, 4);
        let role_onehot = one_hot::<B>(role_clamped, 5, &device);

        // Size: sinusoidal encoded
        let size_enc = self.sinusoidal_encode(size);

        Tensor::cat(vec![age_enc, role_onehot, size_enc], 1)
    }
}

/// One-hot encoding helper
fn one_hot<B: Backend>(indices: Tensor<B, 1, Int>, num_classes: usize, device: &B::Device) -> Tensor<B, 2> {
    let n = indices.dims()[0];
    let indices_data: Vec<i64> = indices.to_data().to_vec().unwrap();

    let mut result = vec![0.0f32; n * num_classes];
    for (i, &idx) in indices_data.iter().enumerate() {
        let idx = idx.max(0).min(num_classes as i64 - 1) as usize;
        result[i * num_classes + idx] = 1.0;
    }

    Tensor::<B, 1>::from_floats(result.as_slice(), device).reshape([n, num_classes])
}

/// GCN layer: h' = σ(A · h · W)
#[derive(Module, Debug)]
pub struct GcnLayer<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> GcnLayer<B> {
    pub fn new(device: &B::Device, in_dim: usize, out_dim: usize) -> Self {
        let linear = LinearConfig::new(in_dim, out_dim).init(device);
        Self { linear }
    }

    pub fn forward(&self, x: Tensor<B, 2>, adj: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = adj.matmul(x);
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
        self.linear2.forward(h)
    }
}

/// GCN-based clause selection model
///
/// New architecture matching PyTorch ClauseGCN:
///   node_features (3d) → node_embedding (15d) → GCN layers → pool
///   clause_features (3d) → clause_embedding (21d)
///   concat(pooled, clause_emb) → clause_proj → scorer → logits
#[derive(Module, Debug)]
pub struct GcnModel<B: Backend> {
    /// Node feature embedding (3d → 15d)
    node_embedding: NodeFeatureEmbedding<B>,
    /// Clause feature embedding (3d → 21d)
    clause_embedding: ClauseFeatureEmbedding<B>,
    /// GCN convolutional layers
    convs: Vec<GcnLayer<B>>,
    /// Layer normalization after each conv
    norms: Vec<LayerNorm<B>>,
    /// Projection from concat(pooled, clause_emb) to hidden_dim
    clause_proj: Linear<B>,
    /// Scoring head MLP
    scorer: ScorerHead<B>,
    /// Hidden dimension
    hidden_dim: usize,
    /// Feature embedding alias for weight loading
    feature_embedding: NodeFeatureEmbedding<B>,
}

impl<B: Backend> GcnModel<B> {
    /// Create a new GCN model with random initialization
    pub fn new(
        device: &B::Device,
        hidden_dim: usize,
        num_layers: usize,
        sin_dim: usize,
    ) -> Self {
        // Node embedding: 3d → 6 + 1 + sin_dim = 15d (for sin_dim=8)
        let node_embedding = NodeFeatureEmbedding::new(device, sin_dim);
        let node_embed_dim = node_embedding.output_dim;

        // Clause embedding: 3d → sin_dim + 5 + sin_dim = 21d (for sin_dim=8)
        let clause_embedding = ClauseFeatureEmbedding::new(device, sin_dim);
        let clause_embed_dim = clause_embedding.output_dim;

        // GCN layers
        let mut convs = Vec::new();
        let mut norms = Vec::new();

        // First layer: node_embed_dim → hidden_dim
        convs.push(GcnLayer::new(device, node_embed_dim, hidden_dim));
        norms.push(LayerNormConfig::new(hidden_dim).init(device));

        // Remaining layers: hidden_dim → hidden_dim
        for _ in 1..num_layers {
            convs.push(GcnLayer::new(device, hidden_dim, hidden_dim));
            norms.push(LayerNormConfig::new(hidden_dim).init(device));
        }

        // Clause projection: hidden_dim + clause_embed_dim → hidden_dim
        let clause_proj = LinearConfig::new(hidden_dim + clause_embed_dim, hidden_dim).init(device);

        let scorer = ScorerHead::new(device, hidden_dim);

        // Feature embedding alias (same as node_embedding for weight loading)
        let feature_embedding = NodeFeatureEmbedding::new(device, sin_dim);

        Self {
            node_embedding,
            clause_embedding,
            convs,
            norms,
            clause_proj,
            scorer,
            hidden_dim,
            feature_embedding,
        }
    }

    /// Load model weights from a safetensors file exported from PyTorch
    pub fn load_from_safetensors<P: AsRef<Path>>(
        path: P,
        device: &B::Device,
        hidden_dim: usize,
        num_layers: usize,
        sin_dim: usize,
    ) -> Result<Self, String> {
        // Create model with correct architecture
        let model = Self::new(device, hidden_dim, num_layers, sin_dim);

        // Load weights from safetensors with PyTorch adapter
        let load_args = LoadArgs::new(path.as_ref().into()).with_adapter_type(AdapterType::PyTorch);

        let record = SafetensorsFileRecorder::<FullPrecisionSettings>::default()
            .load(load_args, device)
            .map_err(|e| format!("Failed to load safetensors: {}", e))?;

        Ok(model.load_record(record))
    }

    /// Encode clauses to embeddings (GCN layers + pooling)
    pub fn encode(
        &self,
        node_features: Tensor<B, 2>,
        adj: Tensor<B, 2>,
        pool_matrix: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        // Embed node features: [N, 3] → [N, 15]
        let mut x = self.node_embedding.forward(node_features);

        // Apply GCN layers with LayerNorm and ReLU
        for (conv, norm) in self.convs.iter().zip(self.norms.iter()) {
            x = conv.forward(x, adj.clone());
            x = norm.forward(x);
            x = relu(x);
        }

        // Pool to clause embeddings: [num_clauses, hidden_dim]
        pool_matrix.matmul(x)
    }

    /// Forward pass: node_features + clause_features → scores
    pub fn forward(
        &self,
        node_features: Tensor<B, 2>,
        adj: Tensor<B, 2>,
        pool_matrix: Tensor<B, 2>,
        clause_features: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        // Encode node features through GCN
        let clause_emb = self.encode(node_features, adj, pool_matrix);

        // Embed clause features
        let clause_feat_emb = self.clause_embedding.forward(clause_features);

        // Concatenate and project
        let concat = Tensor::cat(vec![clause_emb, clause_feat_emb], 1);
        let projected = self.clause_proj.forward(concat);

        // Score
        self.scorer.forward(projected)
    }

    /// Score clauses (returns logits, not probabilities)
    pub fn score(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.scorer.forward(x)
    }
}

/// NdArray-backed GCN selector type
pub type NdarrayGcnSelector = BurnGcnSelector<burn_ndarray::NdArray<f32>>;

/// GCN-based clause selector using Burn
pub struct BurnGcnSelector<B: Backend> {
    model: GcnModel<B>,
    device: B::Device,
    /// Clause age tracking
    clause_ages: HashMap<usize, usize>,
    /// Current step counter for age calculation
    current_step: usize,
    /// Cached GCN embeddings per clause (hidden_dim floats)
    clause_gcn_cache: HashMap<usize, Vec<f32>>,
}

impl<B: Backend> BurnGcnSelector<B> {
    pub fn new(model: GcnModel<B>, device: B::Device) -> Self {
        Self {
            model,
            device,
            clause_ages: HashMap::new(),
            current_step: 0,
            clause_gcn_cache: HashMap::new(),
        }
    }

    /// Register a clause and its creation time
    pub fn register_clause(&mut self, clause_id: usize) {
        self.clause_ages.insert(clause_id, self.current_step);
    }

    /// Compute GCN embedding for a single clause and cache it
    fn compute_and_cache_embedding(&mut self, clause_id: usize, clause: &Clause) {
        use crate::ml::GraphBuilder;

        if self.clause_gcn_cache.contains_key(&clause_id) {
            return;
        }

        let graph = GraphBuilder::build_from_clause_with_context(clause, 1);
        let num_nodes = graph.node_features.len();

        if num_nodes == 0 {
            // Empty clause - cache zeros
            self.clause_gcn_cache.insert(clause_id, vec![0.0; self.model.hidden_dim]);
            return;
        }

        // Build node features tensor [num_nodes, 3]
        let node_feat_flat: Vec<f32> = graph.node_features.iter()
            .flat_map(|f| [f[0], f[1], f[2]])
            .collect();
        let node_features = Tensor::<B, 1>::from_floats(node_feat_flat.as_slice(), &self.device)
            .reshape([num_nodes, 3]);

        // Build adjacency matrix for single clause
        let adj = self.build_single_adjacency(&graph.edge_indices, num_nodes);

        // Build pool matrix [1, num_nodes] - mean pooling
        let weight = 1.0 / num_nodes as f32;
        let pool_data: Vec<f32> = vec![weight; num_nodes];
        let pool_matrix = Tensor::<B, 1>::from_floats(pool_data.as_slice(), &self.device)
            .reshape([1, num_nodes]);

        // Run GCN encode (node_embedding + GCN layers + pooling)
        let embedding = self.model.encode(node_features, adj, pool_matrix);

        // Cache the embedding [1, hidden_dim] -> Vec<f32>
        let embedding_data: Vec<f32> = embedding.to_data().to_vec().unwrap();
        self.clause_gcn_cache.insert(clause_id, embedding_data);
    }

    /// Build adjacency matrix for a single clause
    fn build_single_adjacency(&self, edges: &[(usize, usize)], num_nodes: usize) -> Tensor<B, 2> {
        let mut adj_data = vec![0.0f32; num_nodes * num_nodes];

        // Add edges (bidirectional)
        for &(src, dst) in edges {
            if src < num_nodes && dst < num_nodes {
                adj_data[src * num_nodes + dst] = 1.0;
                adj_data[dst * num_nodes + src] = 1.0;
            }
        }

        // Add self-loops
        for i in 0..num_nodes {
            adj_data[i * num_nodes + i] = 1.0;
        }

        // Normalize: D^(-1/2) A D^(-1/2)
        let mut degrees = vec![0.0f32; num_nodes];
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                degrees[i] += adj_data[i * num_nodes + j];
            }
        }

        for i in 0..num_nodes {
            for j in 0..num_nodes {
                if degrees[i] > 0.0 && degrees[j] > 0.0 {
                    adj_data[i * num_nodes + j] /= (degrees[i] * degrees[j]).sqrt();
                }
            }
        }

        Tensor::<B, 1>::from_floats(adj_data.as_slice(), &self.device).reshape([num_nodes, num_nodes])
    }

    /// Score clauses using cached GCN embeddings
    fn score_clauses(&self, clause_ids: &[usize], clauses: &[Clause]) -> Option<Vec<f32>> {
        let num_clauses = clause_ids.len();
        if num_clauses == 0 {
            return None;
        }

        let hidden_dim = self.model.hidden_dim;
        let max_age = self.current_step.max(1);

        // Build tensors from cached embeddings
        let mut gcn_emb_flat = Vec::with_capacity(num_clauses * hidden_dim);
        let mut clause_feat_flat = Vec::with_capacity(num_clauses * 3);

        for &clause_id in clause_ids {
            // Get cached GCN embedding
            let gcn_emb = self.clause_gcn_cache.get(&clause_id)?;
            gcn_emb_flat.extend_from_slice(gcn_emb);

            // Compute clause features (age changes over time)
            let clause = &clauses[clause_id];
            let age = self.clause_ages.get(&clause_id).copied().unwrap_or(0);
            let age_normalized = age as f32 / max_age as f32;
            let role = clause.role.to_feature_value();
            let size = clause.literals.len() as f32;
            clause_feat_flat.extend_from_slice(&[age_normalized, role, size]);
        }

        // Build tensors
        let gcn_embeddings = Tensor::<B, 1>::from_floats(gcn_emb_flat.as_slice(), &self.device)
            .reshape([num_clauses, hidden_dim]);
        let clause_features = Tensor::<B, 1>::from_floats(clause_feat_flat.as_slice(), &self.device)
            .reshape([num_clauses, 3]);

        // Embed clause features
        let clause_feat_emb = self.model.clause_embedding.forward(clause_features);

        // Concatenate and project
        let concat = Tensor::cat(vec![gcn_embeddings, clause_feat_emb], 1);
        let projected = self.model.clause_proj.forward(concat);

        // Score
        let scores = self.model.scorer.forward(projected);
        Some(scores.to_data().to_vec().unwrap())
    }
}

impl<B: Backend> ClauseSelector for BurnGcnSelector<B> {
    fn name(&self) -> &str {
        "burn_gcn"
    }

    fn select(&mut self, unprocessed: &mut VecDeque<usize>, clauses: &[Clause]) -> Option<usize> {
        self.current_step += 1;

        if unprocessed.is_empty() {
            return None;
        }

        // Collect clause IDs
        let clause_ids: Vec<usize> = unprocessed.iter().copied().collect();

        // Register new clauses and compute embeddings for uncached ones
        for &id in &clause_ids {
            if !self.clause_ages.contains_key(&id) {
                self.register_clause(id);
            }
            // Compute and cache GCN embedding if not already cached
            self.compute_and_cache_embedding(id, &clauses[id]);
        }

        // Score all clauses using cached embeddings
        let scores = self.score_clauses(&clause_ids, clauses)?;

        // Get argmax
        let (best_idx, _) = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())?;

        // Remove selected clause from unprocessed
        let selected_id = clause_ids[best_idx];
        unprocessed.retain(|&id| id != selected_id);

        Some(selected_id)
    }

    fn reset(&mut self) {
        self.clause_ages.clear();
        self.clause_gcn_cache.clear();
        self.current_step = 0;
    }
}

// Factory functions

/// Create a new GCN selector with random weights (for testing)
pub fn create_ndarray_gcn_selector(
    hidden_dim: usize,
    num_layers: usize,
) -> NdarrayGcnSelector {
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let model = GcnModel::new(&device, hidden_dim, num_layers, 8);
    BurnGcnSelector::new(model, device)
}

/// Load a GCN selector from safetensors weights
pub fn load_ndarray_gcn_selector<P: AsRef<Path>>(
    path: P,
    hidden_dim: usize,
    num_layers: usize,
) -> Result<NdarrayGcnSelector, String> {
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let model = GcnModel::load_from_safetensors(path, &device, hidden_dim, num_layers, 8)?;
    Ok(BurnGcnSelector::new(model, device))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Atom, ClauseRole, Literal, PredicateSymbol, Term, Variable};

    fn make_test_clause(id: usize, name: &str, num_args: usize) -> Clause {
        let args: Vec<Term> = (0..num_args)
            .map(|j| Term::Variable(Variable { name: format!("X{}", j) }))
            .collect();

        Clause {
            literals: vec![Literal::positive(Atom {
                predicate: PredicateSymbol { name: name.to_string(), arity: num_args },
                args,
            })],
            id: Some(id),
            role: ClauseRole::Derived,
            age: 0,
        }
    }

    #[test]
    fn test_gcn_model_creation() {
        let device = burn_ndarray::NdArrayDevice::Cpu;
        let model: GcnModel<burn_ndarray::NdArray<f32>> = GcnModel::new(&device, 256, 6, 8);
        assert_eq!(model.hidden_dim, 256);
        assert_eq!(model.convs.len(), 6);
    }

    #[test]
    fn test_gcn_selector_basic() {
        let mut selector = create_ndarray_gcn_selector(64, 3);

        let clause1 = make_test_clause(0, "p", 2);
        let clause2 = make_test_clause(1, "q", 1);

        let clauses = vec![clause1, clause2];
        let mut unprocessed: VecDeque<usize> = vec![0, 1].into_iter().collect();

        let result = selector.select(&mut unprocessed, &clauses);

        assert!(result.is_some());
        assert_eq!(unprocessed.len(), 1); // One was selected and removed
    }

    #[test]
    fn test_gcn_selector_empty() {
        let mut selector = create_ndarray_gcn_selector(64, 3);
        let clauses: Vec<Clause> = vec![];
        let mut unprocessed: VecDeque<usize> = VecDeque::new();
        assert!(selector.select(&mut unprocessed, &clauses).is_none());
    }
}
