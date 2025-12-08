# Machine Learning Training Pipeline

Complete pipeline for training a GNN to learn clause selection.

## Overview

```
TPTP Problems → Run Prover → Extract Training Data → Train GNN → Integrate Model
```

## Phase 1: Data Collection

### Run Prover on Problem Set

```python
from proofatlas import parse_tptp_file, saturate, SaturationConfig
import json

problems_dir = ".tptp/TPTP-v9.0.0/Problems/"
output_dir = "data/proofs/"

config = SaturationConfig()
config.timeout = 60  # 60 seconds per problem

for problem_file in glob(f"{problems_dir}/**/*.p"):
    formula = parse_tptp_file(problem_file)
    result = saturate(formula, config)

    if result.is_proof():
        # Save proof for training data extraction
        proof_data = {
            "problem": problem_file,
            "proof": result.to_json(),
            "num_clauses": len(result.all_clauses()),
            "num_steps": len(result.proof_steps()),
        }

        with open(f"{output_dir}/{problem_name}.json", "w") as f:
            json.dump(proof_data, f)
```

### Extract Training Data from Proofs

```python
from proofatlas import extract_training_data
from proofatlas.ml import to_torch_tensors
import torch

training_examples = []

for proof_file in glob("data/proofs/*.json"):
    with open(proof_file) as f:
        proof_data = json.load(f)

    # Extract training data (clause labels)
    examples = extract_training_data(proof_data["proof"])

    # For each example, get clause graph
    for example in examples:
        clause_id = example["clause_idx"]
        label = example["label"]  # 1 = in proof, 0 = not in proof

        # Get clause graph
        graph = get_clause_graph(proof_data, clause_id)
        tensors = to_torch_tensors(graph)

        training_examples.append({
            "edge_index": tensors["edge_index"],
            "node_features": tensors["x"],
            "node_types": tensors["node_types"],
            "label": label,
            "problem": proof_file,
        })

# Save training dataset
torch.save(training_examples, "data/training_data.pt")
```

## Phase 2: GNN Model

### Model Architecture

```python
import torch
import torch.nn as nn
from proofatlas.ml import extract_graph_embeddings

class GCNLayer(nn.Module):
    """Simple GCN layer using pure PyTorch"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        row, col = edge_index
        out = torch.zeros_like(x)
        out.index_add_(0, row, x[col])
        return self.linear(out).relu()

class ClauseGNN(nn.Module):
    """GNN for predicting clause quality (proof relevance)"""

    def __init__(self, node_feature_dim=20, hidden_dim=64, num_layers=2):
        super().__init__()

        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNLayer(node_feature_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNLayer(hidden_dim, hidden_dim))

        # Output layer
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch=None):
        # GNN layers
        for conv in self.convs:
            x = conv(x, edge_index)

        # Global pooling (graph-level embedding)
        if batch is None:
            # Single graph
            x = torch.mean(x, dim=0, keepdim=True)
        else:
            # Batched graphs
            x = extract_graph_embeddings(x, batch, method='mean')

        # Classification
        return self.classifier(x).squeeze()
```

### Training Loop

```python
from proofatlas.ml import batch_graphs
import torch.optim as optim

# Load training data
training_examples = torch.load("data/training_data.pt")

# Split train/val/test
train_size = int(0.8 * len(training_examples))
val_size = int(0.1 * len(training_examples))
train_examples = training_examples[:train_size]
val_examples = training_examples[train_size:train_size + val_size]

# Initialize model
model = ClauseGNN(node_feature_dim=20, hidden_dim=64, num_layers=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# Training loop
batch_size = 32
for epoch in range(100):
    model.train()
    total_loss = 0

    # Mini-batch training
    for i in range(0, len(train_examples), batch_size):
        batch_examples = train_examples[i:i+batch_size]

        # Extract graphs and labels
        graphs = [ex["graph"] for ex in batch_examples]
        labels = [ex["label"] for ex in batch_examples]

        # Batch graphs together
        batched = batch_graphs(graphs, labels=labels)

        optimizer.zero_grad()

        # Forward pass
        predictions = model(batched['x'], batched['edge_index'], batched['batch'])
        loss = criterion(predictions, batched['y'])

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(val_examples), batch_size):
            batch_examples = val_examples[i:i+batch_size]
            graphs = [ex["graph"] for ex in batch_examples]
            labels = [ex["label"] for ex in batch_examples]
            batched = batch_graphs(graphs, labels=labels)

            predictions = model(batched['x'], batched['edge_index'], batched['batch'])
            predicted = (torch.sigmoid(predictions) > 0.5).float()
            correct += (predicted == batched['y']).sum().item()
            total += len(batched['y'])

    accuracy = correct / total
    print(f"Epoch {epoch}: Loss={total_loss:.4f}, Val Acc={accuracy:.4f}")

# Save trained model
torch.save(model.state_dict(), "models/clause_gnn.pt")
```

## Phase 3: Model Integration

### Export Model for Inference

```python
# Export model to ONNX for faster inference
import torch.onnx

model.eval()
dummy_x = torch.randn(10, 20)  # 10 nodes, 20 features
dummy_edge_index = torch.randint(0, 10, (2, 15))  # 15 edges

torch.onnx.export(
    model,
    (dummy_x, dummy_edge_index),
    "models/clause_gnn.onnx",
    input_names=["node_features", "edge_index"],
    output_names=["score"],
    dynamic_axes={
        "node_features": {0: "num_nodes"},
        "edge_index": {1: "num_edges"},
    },
)
```

### Use Model in Prover

```rust
// In rust/src/selection/learned.rs

use crate::core::Clause;
use crate::ml::GraphBuilder;
use crate::selection::ClauseSelector;
use onnxruntime::{environment::Environment, GraphOptimizationLevel, LoggingLevel};
use std::path::Path;

pub struct LearnedClauseSelector {
    session: onnxruntime::session::Session<'static>,
    graph_builder: GraphBuilder,
}

impl LearnedClauseSelector {
    pub fn new(model_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let environment = Environment::builder()
            .with_name("clause_selection")
            .with_log_level(LoggingLevel::Warning)
            .build()?;

        let session = environment
            .new_session_builder()?
            .with_optimization_level(GraphOptimizationLevel::All)?
            .with_model_from_file(model_path)?;

        Ok(LearnedClauseSelector {
            session,
            graph_builder: GraphBuilder::new(),
        })
    }

    fn score_clause(&self, clause: &Clause) -> f32 {
        // Convert clause to graph
        let graph = self.graph_builder.build_clause_graph(clause);

        // Prepare ONNX inputs
        let node_features = graph.node_features_as_array();
        let edge_index = graph.edge_indices_as_array();

        // Run inference
        let outputs = self.session
            .run(vec![node_features, edge_index])
            .unwrap();

        // Extract score
        outputs[0].try_extract::<f32>().unwrap()[0]
    }
}

impl ClauseSelector for LearnedClauseSelector {
    fn select(
        &mut self,
        unprocessed: &mut VecDeque<usize>,
        clauses: &[Clause],
    ) -> Option<usize> {
        if unprocessed.is_empty() {
            return None;
        }

        // Score all clauses
        let mut best_idx = None;
        let mut best_score = f32::NEG_INFINITY;

        for &idx in unprocessed.iter() {
            let score = self.score_clause(&clauses[idx]);
            if score > best_score {
                best_score = score;
                best_idx = Some(idx);
            }
        }

        // Remove and return best clause
        if let Some(idx) = best_idx {
            unprocessed.retain(|&i| i != idx);
            Some(idx)
        } else {
            None
        }
    }
}
```

### Use in Prover Binary

```rust
// In rust/src/bin/prove.rs

use proofatlas::selection::LearnedClauseSelector;

fn main() {
    // ... parse args ...

    let mut state = SaturationState::new(initial_clauses, config);

    // Use learned selector if model path provided
    if let Some(model_path) = args.model {
        let selector = LearnedClauseSelector::new(&model_path)
            .expect("Failed to load model");
        state.set_clause_selector(Box::new(selector));
    }

    let result = state.saturate();
    // ... handle result ...
}
```

## Expected Results

### Data Collection
- ~1000 TPTP problems
- ~100-500 clauses per proof
- ~100K-500K training examples
- Balance: 5-20% positive (in proof), 80-95% negative

### Training
- Training time: 1-2 hours on GPU
- Model size: ~1 MB
- Accuracy: 70-85% on test set

### Inference
- Overhead: ~1-5ms per clause scoring
- Speedup: 2-10x faster proof search (fewer wrong choices)
- Proof success rate: 10-30% improvement on hard problems

## Implementation Timeline

1. **Week 1**: Python bindings for training data export
2. **Week 2**: GNN model implementation and training
3. **Week 3**: Rust integration (ONNX runtime)
4. **Week 4**: Evaluation and tuning

## Next Steps

1. Add Python bindings to `python_bindings.rs`:
   - `extract_training_data(proof) -> List[TrainingExample]`
   - `get_clause_graph(proof, clause_idx) -> ClauseGraphData`

2. Create `python/proofatlas/ml/training.py`:
   - Data collection script
   - GNN model definition
   - Training loop

3. Add Rust dependency: `onnxruntime = "0.0.14"`

4. Implement `rust/src/selection/learned.rs`
