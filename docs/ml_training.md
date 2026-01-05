# Machine Learning for Clause Selection

ProofAtlas uses learned clause selection to guide the saturation-based theorem prover. This document describes the ML architecture, training pipeline, and configuration.

## Overview

The ML system learns to predict which clauses are likely to be useful for finding a proof. During inference, clauses are scored and selected probabilistically based on these predictions.

```
Training Pipeline:
  TPTP Problems
       ↓
  Rust Saturation (with trace collection)
       ↓
  Proof DAG Extraction (positive/negative labels)
       ↓
  PyTorch Training (InfoNCE contrastive loss)
       ↓
  Export to safetensors
       ↓
  Burn Inference (Rust)
```

## Graph Representation

Clauses are represented as tree-structured graphs with 13-dimensional node features.

### Node Types (one-hot encoded, dims 0-5)

| Type | Description |
|------|-------------|
| clause | Root node for each clause |
| literal | Positive or negative atom |
| predicate | Predicate symbol |
| function | Function application |
| variable | Logic variable |
| constant | Constant symbol |

### Additional Features (dims 6-12)

| Feature | Description |
|---------|-------------|
| arity | Number of arguments (normalized) |
| arg_position | Position in parent's argument list |
| depth | Depth in the tree structure |
| age | Clause age (normalized by max_age) |
| role | Clause role (axiom, conjecture, etc.) |
| polarity | Literal polarity (positive/negative) |
| is_equality | Whether predicate is equality |

### Graph Structure

```
clause
  ├── literal (polarity=+)
  │     └── predicate "P"
  │           ├── variable "X"
  │           └── constant "a"
  └── literal (polarity=-)
        └── predicate "Q"
              └── function "f"
                    └── variable "Y"
```

Edges are bidirectional (parent ↔ child).

## Model Architecture

### GCN (Graph Convolutional Network)

The primary model uses GCN layers for message passing:

```
Input: node_features [N, 13]
       adj          [N, N]  (normalized adjacency)
       pool_matrix  [C, N]  (clause pooling)

GCN Layers (×num_layers):
  h' = LayerNorm(ReLU(A · h · W))

Pooling:
  clause_emb = pool_matrix @ h  → [C, hidden_dim]

Scorer (MLP):
  logits = Linear(ReLU(Linear(clause_emb)))  → [C, 1]
```

### Available Models

| Model | Description | Burn Support |
|-------|-------------|--------------|
| gcn | Graph Convolutional Network | ✓ |
| gat | Graph Attention Network | ✗ |
| graphsage | GraphSAGE (sampling-based) | ✗ |
| mlp | Simple MLP baseline | ✓ |

## Training

### Loss Functions

**InfoNCE (default)**: Contrastive loss that learns relative preferences.

```python
loss = -log(exp(s_pos/τ) / (exp(s_pos/τ) + Σ exp(s_neg/τ)))
```

For each positive (proof clause), it should score higher than all negatives.

**Margin Ranking**: Pairwise margin loss.

```python
loss = max(0, margin - (s_pos - s_neg))
```

### Labels

Labels are extracted from completed proofs:
- `1` = clause is in the proof DAG (backward reachable from empty clause)
- `0` = clause was generated but not used in the proof

### Metrics

| Metric | Description |
|--------|-------------|
| train_loss | Training loss (InfoNCE/margin/BCE) |
| val_loss | Validation loss |
| val_acc | Binary accuracy (score > 0) |
| val_mrr | Mean Reciprocal Rank of positive clauses |

### Configuration

Training is configured via `TrainingConfig` (or `SelectorConfig`):

```python
from proofatlas.ml.config import SelectorConfig, ModelConfig, TrainingParams

config = SelectorConfig(
    name="my_gcn",
    model=ModelConfig(
        type="gcn",
        hidden_dim=64,
        num_layers=3,
        dropout=0.1,
    ),
    training=TrainingParams(
        batch_size=32,
        learning_rate=0.001,
        max_epochs=100,
        loss_type="info_nce",  # info_nce, margin, bce
        temperature=1.0,       # for InfoNCE
    ),
)
```

### Running Training

```python
from proofatlas.ml.training import train, ClauseDataset

# Create datasets with binary labels
train_dataset = ClauseDataset(
    node_features=[...],  # List of [num_nodes, 13] tensors
    edge_indices=[...],   # List of [2, num_edges] tensors
    labels=[...],         # List of 0/1 labels
)

model, metrics = train(train_dataset, val_dataset, config)
```

## Weight Export

Trained PyTorch models are exported to safetensors for Rust inference:

```python
import torch
from safetensors.torch import save_file

# Save model weights
save_file(model.state_dict(), ".weights/gcn_v1.safetensors")
```

The Burn-based selector loads these weights:

```rust
let selector = load_ndarray_gcn_selector(
    ".weights/gcn_v1.safetensors",
    13,   // input_dim
    64,   // hidden_dim
    3,    // num_layers
)?;
```

## Inference

During saturation, the selector:

1. Builds graphs for unprocessed clauses
2. Encodes clauses to embeddings (cached)
3. Scores embeddings with the MLP scorer
4. Samples from softmax distribution

```rust
// In saturation loop
let selected = selector.select(&mut unprocessed, &clauses);
```

### Embedding Cache

The `BurnGcnSelector` caches clause embeddings to avoid recomputation:

```rust
selector.clear_cache();  // Call when starting new problem
```

## File Locations

| Path | Description |
|------|-------------|
| `crates/proofatlas/src/ml/` | Rust graph building, weight loading |
| `crates/proofatlas/src/selectors/` | Rust/Burn selector implementations |
| `python/proofatlas/ml/` | Training infrastructure |
| `python/proofatlas/selectors/` | PyTorch model definitions |
| `configs/models.json` | Model architecture presets |
| `configs/training.json` | Training hyperparameter presets |
| `.weights/` | Trained model weights |
| `.data/traces/` | Collected proof traces |

## Hyperparameter Recommendations

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| hidden_dim | 64 | Start small, increase if underfitting |
| num_layers | 3 | More layers for deeper reasoning |
| dropout | 0.1 | Increase if overfitting |
| learning_rate | 0.001 | Reduce if loss is unstable |
| temperature | 0.5-1.0 | Lower = sharper preferences |
| batch_size | 32-64 | Larger for more stable gradients |

## Troubleshooting

### Loss not decreasing
- Check label balance (should have both 0s and 1s in each batch)
- Try lower learning rate
- Verify graph construction is correct

### Inference too slow
- Reduce hidden_dim
- Use embedding cache (don't clear between selections)
- Consider MLP baseline for speed

### Weight loading fails
- Verify safetensors format (PyTorch export)
- Check hidden_dim/num_layers match between training and inference
- Ensure input_dim is 13 (default feature dimension)
