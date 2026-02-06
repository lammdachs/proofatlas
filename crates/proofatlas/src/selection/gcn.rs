//! GCN clause selector with embedding caching
//!
//! This implements a Graph Convolutional Network (GCN) for clause selection,
//! using PyTorch via tch-rs for inference. Embeddings are cached per clause
//! to avoid redundant computation.
//!
//! The architecture separates embedding computation (GCN forward pass) from
//! scoring, allowing embeddings to be cached and reused across selections.

#[cfg(feature = "ml")]
use std::path::Path;

#[cfg(feature = "ml")]
use crate::fol::Clause;
#[cfg(feature = "ml")]
use super::graph::GraphBuilder;

#[cfg(feature = "ml")]
use super::cached::{CachingSelector, ClauseEmbedder, EmbeddingScorer};

/// GCN embedder using PyTorch for inference
///
/// Computes clause scores using a GCN model. The model operates on a graph
/// representation of clauses and outputs scores directly. These scores are
/// treated as 1-dimensional "embeddings" for caching purposes.
///
/// Since clauses in the batch graph are independent (no inter-clause edges),
/// each clause's score depends only on its own structure, making caching valid.
#[cfg(feature = "ml")]
pub struct GcnEmbedder {
    model: tch::CModule,
    device: tch::Device,
    /// Fixed max_age for normalization (default: 10000)
    max_age: usize,
}

#[cfg(feature = "ml")]
impl GcnEmbedder {
    /// Create a new GCN embedder from a TorchScript model
    pub fn new<P: AsRef<Path>>(model_path: P, use_cuda: bool) -> Result<Self, String> {
        let device = if use_cuda && tch::Cuda::is_available() {
            tch::Device::Cuda(0)
        } else {
            tch::Device::Cpu
        };

        let model = tch::CModule::load_on_device(model_path.as_ref(), device)
            .map_err(|e| format!("Failed to load TorchScript GCN model: {}", e))?;

        Ok(Self {
            model,
            device,
            max_age: 10000,
        })
    }

    /// Build tensors for batch inference
    fn build_batch_tensors(
        &self,
        clauses: &[&Clause],
    ) -> (tch::Tensor, tch::Tensor, tch::Tensor, tch::Tensor) {
        let num_clauses = clauses.len();

        // Build combined graph for all clauses
        let graph = GraphBuilder::build_from_clauses(clauses);
        let num_nodes = graph.num_nodes;

        if num_nodes == 0 {
            // Empty graph - return minimal sparse tensors
            let node_features = tch::Tensor::zeros([1, 3], (tch::Kind::Float, self.device));
            let empty_idx = tch::Tensor::zeros([2, 0], (tch::Kind::Int64, self.device));
            let empty_vals = tch::Tensor::zeros([0], (tch::Kind::Float, self.device));
            let adj = tch::Tensor::sparse_coo_tensor_indices_size(
                &empty_idx, &empty_vals, [1, 1], (tch::Kind::Float, self.device), true,
            );
            let pool_matrix = tch::Tensor::sparse_coo_tensor_indices_size(
                &empty_idx, &empty_vals,
                [num_clauses as i64, 1], (tch::Kind::Float, self.device), true,
            );
            let clause_features =
                tch::Tensor::zeros([num_clauses as i64, 3], (tch::Kind::Float, self.device));
            return (node_features, adj, pool_matrix, clause_features);
        }

        // Build node features tensor [num_nodes, 3]
        let node_feat_flat: Vec<f32> = graph.node_features.iter().flat_map(|f| *f).collect();
        let node_features = tch::Tensor::from_slice(&node_feat_flat)
            .view([num_nodes as i64, 3])
            .to_device(self.device);

        // Build normalized adjacency matrix
        let adj = self.build_adjacency(&graph.edge_indices, num_nodes);

        // Build pool matrix from clause_boundaries
        let pool_matrix =
            self.build_pool_matrix(&graph.clause_boundaries, num_clauses, num_nodes);

        // Build clause features [num_clauses, 3]: age_normalized, role, size
        let mut clause_feat_flat = Vec::with_capacity(num_clauses * 3);
        for clause in clauses {
            let age_normalized = clause.age as f32 / self.max_age as f32;
            let role = clause.role.to_feature_value();
            let size = clause.literals.len() as f32;
            clause_feat_flat.extend_from_slice(&[age_normalized.min(1.0), role, size]);
        }
        let clause_features = tch::Tensor::from_slice(&clause_feat_flat)
            .view([num_clauses as i64, 3])
            .to_device(self.device);

        (node_features, adj, pool_matrix, clause_features)
    }

    /// Build normalized sparse adjacency matrix (COO format)
    ///
    /// Constructs D^(-1/2) A D^(-1/2) as a sparse COO tensor, where A includes
    /// bidirectional edges and self-loops. Memory is O(edges + nodes) instead of
    /// O(nodes²).
    fn build_adjacency(&self, edges: &[(usize, usize)], num_nodes: usize) -> tch::Tensor {
        use std::collections::HashSet;

        // Collect unique edges (bidirectional + self-loops)
        let mut edge_set = HashSet::new();
        for &(src, dst) in edges {
            if src < num_nodes && dst < num_nodes {
                edge_set.insert((src, dst));
                edge_set.insert((dst, src));
            }
        }
        for i in 0..num_nodes {
            edge_set.insert((i, i));
        }

        // Compute degrees from the edge set
        let mut degrees = vec![0.0f32; num_nodes];
        for &(src, _) in &edge_set {
            degrees[src] += 1.0;
        }

        // Build COO indices and normalized values
        let nnz = edge_set.len();
        let mut rows = Vec::with_capacity(nnz);
        let mut cols = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        for &(src, dst) in &edge_set {
            rows.push(src as i64);
            cols.push(dst as i64);
            values.push(1.0 / (degrees[src] * degrees[dst]).sqrt());
        }

        let indices = tch::Tensor::from_slice2(&[&rows, &cols]).to_device(self.device);
        let vals = tch::Tensor::from_slice(&values).to_device(self.device);
        let n = num_nodes as i64;

        tch::Tensor::sparse_coo_tensor_indices_size(
            &indices,
            &vals,
            [n, n],
            (tch::Kind::Float, self.device),
            false,
        ).coalesce()
    }

    /// Build sparse pool matrix from clause boundaries (COO format)
    ///
    /// Each clause maps to a mean-pool over its nodes. Memory is O(nodes)
    /// instead of O(clauses × nodes).
    fn build_pool_matrix(
        &self,
        clause_boundaries: &[(usize, usize)],
        num_clauses: usize,
        num_nodes: usize,
    ) -> tch::Tensor {
        let mut rows = Vec::with_capacity(num_nodes);
        let mut cols = Vec::with_capacity(num_nodes);
        let mut values = Vec::with_capacity(num_nodes);

        for (clause_idx, &(start, end)) in clause_boundaries.iter().enumerate() {
            let clause_size = end - start;
            if clause_size > 0 {
                let weight = 1.0 / clause_size as f32;
                for node_idx in start..end {
                    if node_idx < num_nodes {
                        rows.push(clause_idx as i64);
                        cols.push(node_idx as i64);
                        values.push(weight);
                    }
                }
            }
        }

        let indices = tch::Tensor::from_slice2(&[&rows, &cols]).to_device(self.device);
        let vals = tch::Tensor::from_slice(&values).to_device(self.device);

        tch::Tensor::sparse_coo_tensor_indices_size(
            &indices,
            &vals,
            [num_clauses as i64, num_nodes as i64],
            (tch::Kind::Float, self.device),
            false,
        ).coalesce()
    }
}

#[cfg(feature = "ml")]
impl ClauseEmbedder for GcnEmbedder {
    fn embed_batch(&self, clauses: &[&Clause]) -> Vec<Vec<f32>> {
        if clauses.is_empty() {
            return vec![];
        }

        // Build batch tensors
        let (node_features, adj, pool_matrix, clause_features) = self.build_batch_tensors(clauses);

        // Run inference
        let scores = tch::no_grad(|| {
            self.model
                .forward_ts(&[node_features, adj, pool_matrix, clause_features])
                .expect("GCN forward failed")
        });

        // Convert scores to 1-element embeddings for caching
        let scores_cpu = scores.to_device(tch::Device::Cpu).view([-1]);
        let scores_vec: Vec<f32> =
            Vec::<f32>::try_from(&scores_cpu).expect("Failed to convert scores to Vec<f32>");

        scores_vec.iter().map(|&s| vec![s]).collect()
    }

    fn embedding_dim(&self) -> usize {
        1 // Scores are treated as 1-dim embeddings
    }

    fn name(&self) -> &str {
        "gcn"
    }
}

/// Pass-through scorer for GCN embedder
///
/// Since the TorchScript model outputs scores directly (treated as embeddings),
/// this scorer simply returns them unchanged.
#[cfg(feature = "ml")]
pub struct GcnScorer;

#[cfg(feature = "ml")]
impl EmbeddingScorer for GcnScorer {
    fn score_batch(&self, embeddings: &[&[f32]]) -> Vec<f32> {
        // Embeddings are already scores (1-element each)
        embeddings.iter().map(|e| e[0]).collect()
    }

    fn name(&self) -> &str {
        "gcn_scorer"
    }
}

/// GCN selector with embedding caching
///
/// This type alias combines the GCN embedder with the caching infrastructure.
/// Embeddings (scores) are computed once per clause and cached by clause string.
#[cfg(feature = "ml")]
pub type GcnSelector = CachingSelector<GcnEmbedder, GcnScorer>;

/// Load a GCN selector from a TorchScript model
///
/// # Arguments
/// * `model_path` - Path to the TorchScript model (.pt file)
/// * `use_cuda` - Whether to use CUDA for inference
///
/// # Returns
/// A GCN selector with embedding caching enabled
#[cfg(feature = "ml")]
pub fn load_gcn_selector<P: AsRef<Path>>(
    model_path: P,
    use_cuda: bool,
) -> Result<GcnSelector, String> {
    let embedder = GcnEmbedder::new(model_path, use_cuda)?;
    let scorer = GcnScorer;
    Ok(CachingSelector::new(embedder, scorer))
}

#[cfg(test)]
#[cfg(feature = "ml")]
mod tests {
    use super::*;

    #[test]
    fn test_gcn_selector_creation() {
        // Skip if model doesn't exist
        let model_path = std::path::Path::new(".weights/gcn_model.pt");
        if !model_path.exists() {
            println!("Skipping test: gcn_model.pt not found");
            return;
        }

        let selector = load_gcn_selector(model_path, false);
        assert!(
            selector.is_ok(),
            "Failed to create selector: {:?}",
            selector
        );
    }

    #[test]
    fn test_gcn_selector_caching() {
        use crate::fol::{Interner, Literal, PredicateSymbol, Term, Constant};
        use crate::selection::ClauseSelector;
        use indexmap::IndexSet;

        // Skip if model doesn't exist
        let model_path = std::path::Path::new(".weights/gcn_model.pt");
        if !model_path.exists() {
            println!("Skipping test: gcn_model.pt not found");
            return;
        }

        let mut interner = Interner::new();
        let mut selector = load_gcn_selector(model_path, false).unwrap();

        // Create test clauses
        let p = PredicateSymbol {
            id: interner.intern_predicate("P"),
            arity: 1,
        };
        let a = Term::Constant(Constant {
            id: interner.intern_constant("a"),
        });
        let b = Term::Constant(Constant {
            id: interner.intern_constant("b"),
        });

        let clause1 = Clause::new(vec![Literal::positive(
            p.clone(),
            vec![a.clone()],
        )]);
        let clause2 = Clause::new(vec![Literal::positive(
            p.clone(),
            vec![b.clone()],
        )]);
        let clause3 = Clause::new(vec![Literal::positive(
            p.clone(),
            vec![a.clone()],
        ), Literal::positive(
            p.clone(),
            vec![b.clone()],
        )]);

        let clauses = vec![clause1, clause2, clause3];
        let mut unprocessed: IndexSet<usize> = (0..3).collect();

        // First selection should populate cache
        let _ = selector.select(&mut unprocessed, &clauses);
        assert_eq!(selector.cache_size(), 3);

        // Reset should clear cache
        selector.reset();
        assert_eq!(selector.cache_size(), 0);
    }
}
