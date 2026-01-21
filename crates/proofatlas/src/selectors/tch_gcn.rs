//! tch-rs based GCN clause selector
//!
//! This implements a Graph Convolutional Network (GCN) for clause selection,
//! using tch-rs (PyTorch bindings) for inference. Weights are loaded from
//! TorchScript models exported from PyTorch training.

#[cfg(feature = "torch")]
use std::collections::HashMap;
#[cfg(feature = "torch")]
use std::collections::VecDeque;
#[cfg(feature = "torch")]
use std::path::Path;

#[cfg(feature = "torch")]
use crate::core::Clause;
#[cfg(feature = "torch")]
use crate::ml::graph::GraphBuilder;

#[cfg(feature = "torch")]
use super::ClauseSelector;

/// TorchScript-based GCN clause selector using tch-rs
///
/// Uses a TorchScript model that takes:
/// - node_features: [N, 3] - node type, arity, arg_pos
/// - adj: [N, N] - normalized adjacency matrix
/// - pool_matrix: [C, N] - clause to node pooling
/// - clause_features: [C, 3] - age, role, size
///
/// And outputs clause scores [C].
#[cfg(feature = "torch")]
pub struct TchGcnSelector {
    model: tch::CModule,
    device: tch::Device,
    /// Clause age tracking
    clause_ages: HashMap<usize, usize>,
    /// Current step counter for age calculation
    current_step: usize,
    /// Cached GCN embeddings per clause (hidden_dim floats)
    #[allow(dead_code)]
    clause_gcn_cache: HashMap<usize, Vec<f32>>,
}

#[cfg(feature = "torch")]
impl TchGcnSelector {
    /// Create a new GCN selector from a TorchScript model
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
            clause_ages: HashMap::new(),
            current_step: 0,
            clause_gcn_cache: HashMap::new(),
        })
    }

    /// Register a clause and its creation time
    fn register_clause(&mut self, clause_id: usize) {
        self.clause_ages.insert(clause_id, self.current_step);
    }

    /// Build tensors for the full batch of clauses
    fn build_batch_tensors(
        &self,
        clause_ids: &[usize],
        clauses: &[Clause],
    ) -> (tch::Tensor, tch::Tensor, tch::Tensor, tch::Tensor) {
        let num_clauses = clause_ids.len();
        let max_age = self.current_step.max(1);

        // Build combined graph for all clauses
        let selected_clauses: Vec<&Clause> = clause_ids.iter().map(|&id| &clauses[id]).collect();

        // Build graph from clauses
        let graph = GraphBuilder::build_from_clauses(&selected_clauses);
        let num_nodes = graph.num_nodes;

        if num_nodes == 0 {
            // Empty graph - return zeros
            let node_features = tch::Tensor::zeros([1, 3], (tch::Kind::Float, self.device));
            let adj = tch::Tensor::eye(1, (tch::Kind::Float, self.device));
            let pool_matrix = tch::Tensor::ones([num_clauses as i64, 1], (tch::Kind::Float, self.device));
            let clause_features = tch::Tensor::zeros([num_clauses as i64, 3], (tch::Kind::Float, self.device));
            return (node_features, adj, pool_matrix, clause_features);
        }

        // Build node features tensor [num_nodes, 3]
        let node_feat_flat: Vec<f32> = graph
            .node_features
            .iter()
            .flat_map(|f| *f)
            .collect();
        let node_features = tch::Tensor::from_slice(&node_feat_flat)
            .view([num_nodes as i64, 3])
            .to_device(self.device);

        // Build normalized adjacency matrix
        let adj = self.build_adjacency(&graph.edge_indices, num_nodes);

        // Build pool matrix from clause_boundaries
        let pool_matrix = self.build_pool_matrix(&graph.clause_boundaries, num_clauses, num_nodes);

        // Build clause features [num_clauses, 3]
        let mut clause_feat_flat = Vec::with_capacity(num_clauses * 3);
        for &clause_id in clause_ids {
            let clause = &clauses[clause_id];
            let age = self.clause_ages.get(&clause_id).copied().unwrap_or(0);
            let age_normalized = age as f32 / max_age as f32;
            let role = clause.role.to_feature_value();
            let size = clause.literals.len() as f32;
            clause_feat_flat.extend_from_slice(&[age_normalized, role, size]);
        }
        let clause_features = tch::Tensor::from_slice(&clause_feat_flat)
            .view([num_clauses as i64, 3])
            .to_device(self.device);

        (node_features, adj, pool_matrix, clause_features)
    }

    /// Build normalized adjacency matrix
    fn build_adjacency(&self, edges: &[(usize, usize)], num_nodes: usize) -> tch::Tensor {
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

        tch::Tensor::from_slice(&adj_data)
            .view([num_nodes as i64, num_nodes as i64])
            .to_device(self.device)
    }

    /// Build pool matrix from clause boundaries
    fn build_pool_matrix(
        &self,
        clause_boundaries: &[(usize, usize)],
        num_clauses: usize,
        num_nodes: usize,
    ) -> tch::Tensor {
        let mut pool_data = vec![0.0f32; num_clauses * num_nodes];

        for (clause_idx, &(start, end)) in clause_boundaries.iter().enumerate() {
            let clause_size = end - start;
            if clause_size > 0 {
                let weight = 1.0 / clause_size as f32;
                for node_idx in start..end {
                    if node_idx < num_nodes {
                        pool_data[clause_idx * num_nodes + node_idx] = weight;
                    }
                }
            }
        }

        tch::Tensor::from_slice(&pool_data)
            .view([num_clauses as i64, num_nodes as i64])
            .to_device(self.device)
    }
}

#[cfg(feature = "torch")]
impl ClauseSelector for TchGcnSelector {
    fn name(&self) -> &str {
        "tch_gcn"
    }

    fn select(&mut self, unprocessed: &mut VecDeque<usize>, clauses: &[Clause]) -> Option<usize> {
        self.current_step += 1;

        if unprocessed.is_empty() {
            return None;
        }

        // Collect clause IDs
        let clause_ids: Vec<usize> = unprocessed.iter().copied().collect();

        // Register new clauses
        for &id in &clause_ids {
            if !self.clause_ages.contains_key(&id) {
                self.register_clause(id);
            }
        }

        // Build batch tensors
        let (node_features, adj, pool_matrix, clause_features) =
            self.build_batch_tensors(&clause_ids, clauses);

        // Run inference
        let scores = tch::no_grad(|| {
            self.model
                .forward_ts(&[node_features, adj, pool_matrix, clause_features])
                .expect("GCN forward failed")
        });

        // Get scores as Vec<f32>
        let scores_cpu = scores.to_device(tch::Device::Cpu).view([-1]);
        let scores_vec: Vec<f32> = Vec::<f32>::try_from(&scores_cpu)
            .expect("Failed to convert scores to Vec<f32>");

        // Get argmax
        let (best_idx, _) = scores_vec
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

// Factory function

/// Load a GCN selector from a TorchScript model
#[cfg(feature = "torch")]
pub fn load_tch_gcn_selector<P: AsRef<Path>>(
    model_path: P,
    use_cuda: bool,
) -> Result<TchGcnSelector, String> {
    TchGcnSelector::new(model_path, use_cuda)
}

#[cfg(test)]
#[cfg(feature = "torch")]
mod tests {
    use super::*;
    use crate::core::{Atom, ClauseRole, Literal, PredicateSymbol, Term, Variable};

    fn make_test_clause(id: usize, name: &str, num_args: usize) -> Clause {
        let args: Vec<Term> = (0..num_args)
            .map(|j| Term::Variable(Variable {
                name: format!("X{}", j),
            }))
            .collect();

        Clause {
            literals: vec![Literal::positive(Atom {
                predicate: PredicateSymbol {
                    name: name.to_string(),
                    arity: num_args,
                },
                args,
            })],
            id: Some(id),
            role: ClauseRole::Derived,
            age: 0,
        }
    }

    #[test]
    fn test_tch_gcn_selector_creation() {
        // Skip if model doesn't exist
        let model_path = std::path::Path::new(".weights/gcn_model.pt");
        if !model_path.exists() {
            println!("Skipping test: gcn_model.pt not found");
            return;
        }

        let selector = load_tch_gcn_selector(model_path, false);
        assert!(selector.is_ok(), "Failed to create selector: {:?}", selector);
    }
}
