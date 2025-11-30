# ONNX Interface for Clause Selection

This document describes the ONNX-based clause selection interface in ProofAtlas. The interface allows using neural network models to guide clause selection during theorem proving.

## Overview

ProofAtlas uses [tract-onnx](https://github.com/sonos/tract) for pure Rust ONNX inference, which enables ML-guided clause selection both in native builds and WebAssembly.

```
Clauses → Graph Representation → ONNX Model → Logits → Softmax Sampling → Selected Clause
```

## Feature Layout

Each clause is converted to a graph where nodes represent syntactic elements. Each node has a **12-dimensional feature vector**:

| Index | Feature | Description |
|-------|---------|-------------|
| 0-5 | Node type | One-hot encoding: clause, literal, predicate, function, variable, constant |
| 6 | Arity | Number of arguments (predicates and functions) |
| 7 | Depth | Distance from clause root in the syntax tree |
| 8 | Age | Clause age normalized to [0, 1] (divided by max_age) |
| 9 | Role | Clause role: 0=axiom, 1=hypothesis, 2=definition, 3=negated_conjecture, 4=derived |
| 10 | Polarity | Literal polarity: 1.0=positive, 0.0=negative |
| 11 | Is equality | 1.0 if predicate is equality (=), 0.0 otherwise |

### Node Types

| ID | Type | Description |
|----|------|-------------|
| 0 | CLAUSE | Root node for the clause |
| 1 | LITERAL | A literal (atom with polarity) |
| 2 | PREDICATE | Predicate symbol |
| 3 | FUNCTION | Function symbol |
| 4 | VARIABLE | Variable term |
| 5 | CONSTANT | Constant term |

### Example

For clause `P(f(x), a)`:

```
[0] CLAUSE (depth=0)
 └─[1] LITERAL (depth=1, polarity=1.0)
     └─[2] PREDICATE "P" (depth=2, arity=2)
         ├─[3] FUNCTION "f" (depth=3, arity=1)
         │   └─[4] VARIABLE "x" (depth=4)
         └─[5] CONSTANT "a" (depth=3)
```

## Model Input/Output

### Inputs

The ONNX model takes two inputs:

1. **node_features** `[total_nodes, 12]`: Concatenated node features from all clauses
2. **pool_matrix** `[num_clauses, total_nodes]`: Matrix mapping nodes to their parent clauses

The pool matrix has rows summing to 1, with each entry being `1/num_nodes` for nodes belonging to that clause.

### Output

- **scores** `[num_clauses]`: Raw logits (not probabilities) for each clause

Higher logits indicate clauses the model considers more valuable. The prover converts these to probabilities via softmax and samples from the distribution.

## Creating a Custom Model

### PyTorch Example

```python
import torch
import torch.nn as nn

class MyClauseSelector(nn.Module):
    def __init__(self, feature_dim=12, hidden_dim=64):
        super().__init__()
        # Transform node features
        self.node_encoder = nn.Linear(feature_dim, hidden_dim)
        # Score clause embeddings
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, node_features, pool_matrix):
        # Encode nodes: [total_nodes, hidden_dim]
        node_embeddings = torch.relu(self.node_encoder(node_features))

        # Pool to clauses: [num_clauses, hidden_dim]
        clause_embeddings = torch.mm(pool_matrix, node_embeddings)

        # Score clauses: [num_clauses]
        scores = self.scorer(clause_embeddings).squeeze(-1)

        return scores

# Export to ONNX
model = MyClauseSelector()
model.eval()

dummy_nodes = torch.randn(50, 12)  # 50 nodes, 12 features
dummy_pool = torch.randn(10, 50)   # 10 clauses, 50 nodes

torch.onnx.export(
    model,
    (dummy_nodes, dummy_pool),
    "my_selector.onnx",
    input_names=["node_features", "pool_matrix"],
    output_names=["scores"],
    dynamic_axes={
        "node_features": {0: "total_nodes"},
        "pool_matrix": {0: "num_clauses", 1: "total_nodes"},
        "scores": {0: "num_clauses"},
    },
    opset_version=14,
)
```

### Age-Weight Baseline Model

The default model mimics the age-weight ratio heuristic:

```python
class AgeWeightSelector(nn.Module):
    """
    With probability p, select the OLDEST clause (highest age).
    With probability 1-p, select the LIGHTEST clause (lowest depth).
    """
    def __init__(self, age_probability=0.5):
        super().__init__()
        self.register_buffer('p', torch.tensor(age_probability))

    def forward(self, node_features, pool_matrix):
        # Pool to clause features: [num_clauses, 12]
        clause_features = torch.mm(pool_matrix, node_features)

        # Extract age (index 8) and depth/weight (index 7)
        ages = clause_features[:, 8]
        weights = clause_features[:, 7]

        num_clauses = clause_features.size(0)

        # Find oldest and lightest clauses
        oldest_idx = torch.argmax(ages)
        lightest_idx = torch.argmin(weights)

        # Build logits: -inf except for oldest and lightest
        logits = torch.full((num_clauses,), -1e9)

        if oldest_idx == lightest_idx:
            logits[oldest_idx] = 0.0
        else:
            logits[oldest_idx] = torch.log(self.p + 1e-10)
            logits[lightest_idx] = torch.log(1 - self.p + 1e-10)

        return logits
```

## Using Models in Rust

### From File Path

```rust
use proofatlas::OnnxClauseSelector;

let selector = OnnxClauseSelector::new("path/to/model.onnx")?;
state.set_clause_selector(Box::new(selector));
```

### From Bytes (WASM)

```rust
let model_bytes: &[u8] = include_bytes!("model.onnx");
let selector = OnnxClauseSelector::from_bytes(model_bytes)?;
```

### Configuring Max Age

```rust
let selector = OnnxClauseSelector::new("model.onnx")?
    .with_max_age(2000);  // Normalize ages by dividing by 2000
```

## Selection Process

1. **Graph Building**: Each clause in the unprocessed set is converted to a graph
2. **Feature Extraction**: Node features are extracted (12 dimensions per node)
3. **Batching**: All clause graphs are concatenated with a pool matrix
4. **Inference**: ONNX model produces one logit per clause
5. **Sampling**: Softmax converts logits to probabilities; clause is sampled from distribution
6. **Fallback**: If scoring fails, falls back to age-weight ratio selection

### Softmax Sampling

The prover uses probabilistic selection rather than always picking the highest score:

```rust
// Compute softmax probabilities
let max_logit = logits.iter().max();
let exp_scores: Vec<f64> = logits.iter()
    .map(|x| ((x - max_logit) as f64).exp())
    .collect();
let sum: f64 = exp_scores.iter().sum();
let probs: Vec<f64> = exp_scores.iter().map(|e| e / sum).collect();

// Sample from distribution
let r = random();
let mut cumsum = 0.0;
for (i, p) in probs.iter().enumerate() {
    cumsum += p;
    if r < cumsum {
        return i;  // Selected clause index
    }
}
```

This allows exploration even when the model is confident, which can help avoid local minima in proof search.

## Testing Your Model

### Unit Test

```rust
#[test]
fn test_my_model() {
    let selector = OnnxClauseSelector::new("my_model.onnx")
        .expect("Failed to load model");

    // Create test clauses
    let clauses = vec![
        Clause::new(vec![...]),
        Clause::new(vec![...]),
    ];

    let mut unprocessed: VecDeque<usize> = (0..clauses.len()).collect();
    let selected = selector.select(&mut unprocessed, &clauses);

    assert!(selected.is_some());
}
```

### Integration Test

```rust
use proofatlas::{parse_tptp_file, SaturationConfig, SaturationState, OnnxClauseSelector};

let formula = parse_tptp_file("problem.p", &[])?;
let selector = OnnxClauseSelector::new("model.onnx")?;

let config = SaturationConfig::default();
let mut state = SaturationState::new(formula.clauses, config);
state.set_clause_selector(Box::new(selector));

let result = state.saturate();
assert!(matches!(result, SaturationResult::Proof(_)));
```

## Model Requirements

1. **Input shapes**: Must accept dynamic batch sizes (use `dynamic_axes` in export)
2. **Output**: Single tensor of shape `[num_clauses]`
3. **Opset version**: 14 or compatible (tract-onnx supports most common ops)
4. **Feature dimension**: Must match the 12-dimensional feature layout

## Supported ONNX Operations

tract-onnx supports most common operations. For clause selection, you typically need:

- Linear layers (MatMul, Add)
- Activations (ReLU, Sigmoid, Tanh)
- Reductions (ArgMax, ArgMin)
- Element-wise ops (Mul, Div, Log, Exp)
- Tensor manipulation (Squeeze, Unsqueeze)

For complex architectures (transformers, GNNs), verify compatibility with tract or use simpler architectures.

## Debugging

### Check Model Loading

```rust
let result = OnnxClauseSelector::new("model.onnx");
match result {
    Ok(_) => println!("Model loaded successfully"),
    Err(e) => println!("Failed to load: {}", e),
}
```

### Check Feature Dimensions

If you get dimension mismatch errors like `20 != 12`, your model expects a different feature count. Ensure your model was trained with the current 12-dimensional features.

### Inspect Graph Features

```rust
use proofatlas::ml::GraphBuilder;

let clause = Clause::new(vec![...]);
let graph = GraphBuilder::build_from_clause(&clause);

println!("Nodes: {}", graph.num_nodes);
println!("Features shape: {} x {}", graph.node_features.len(), 12);
for (i, features) in graph.node_features.iter().enumerate() {
    println!("Node {}: {:?}", i, features);
}
```

## File Locations

- **Rust graph builder**: `rust/src/ml/graph.rs`
- **ONNX inference**: `rust/src/ml/inference.rs`
- **Clause selector**: `rust/src/selection/clause.rs`
- **CI model generation**: `.github/workflows/deploy.yml`
