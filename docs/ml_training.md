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
  Export to TorchScript (.pt)
       ↓
  tch-rs Inference (Rust, GPU-accelerated)
```

## Graph Representation

Clauses are represented as tree-structured graphs with 8-dimensional raw node features.
The model's `FeatureEmbedding` layer converts these to a richer representation with one-hot and sinusoidal encodings.

### Raw Feature Layout (8 dims)

| Index | Feature | Type | Description |
|-------|---------|------|-------------|
| 0 | node_type | int 0-5 | clause, literal, predicate, function, variable, constant |
| 1 | arity | int | Number of arguments |
| 2 | arg_position | int | Position in parent's argument list |
| 3 | depth | int | Depth in the tree structure |
| 4 | age | float 0-1 | Clause age (normalized by max_age) |
| 5 | role | int 0-4 | axiom, hypothesis, definition, negated_conjecture, derived |
| 6 | polarity | binary | 1=positive literal, 0=negative |
| 7 | is_equality | binary | 1 if equality predicate |

### Model-Side Encoding (FeatureEmbedding)

The model transforms raw features to:
- **Node type**: one-hot (6 dims)
- **Arity**: log1p scaled (1 dim)
- **Arg position**: sinusoidal (sin_dim dims)
- **Depth**: sinusoidal (sin_dim dims)
- **Age**: sinusoidal (sin_dim dims)
- **Role**: one-hot (5 dims)
- **Polarity**: kept (1 dim)
- **Is equality**: kept (1 dim)

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
Input: node_features [N, 8]
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

| Model | Description | Rust Inference |
|-------|-------------|----------------|
| gcn | Graph Convolutional Network | ✓ (tch-rs) |
| sentence | Sentence transformer (MiniLM) | ✓ (tch-rs) |
| gat | Graph Attention Network | ✗ |
| graphsage | GraphSAGE (sampling-based) | ✗ |

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

Training is configured via `TrainingConfig`:

```python
from proofatlas.ml.config import TrainingConfig, ModelConfig, TrainingParams

config = TrainingConfig(
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

Training is typically done via the CLI:

```bash
# Train a GCN model (collects traces if needed)
proofatlas-bench --config gcn_mlp_sel21 --retrain
```

Or programmatically:

```python
from proofatlas.ml.training import run_training

# Train from collected traces
weights_path = run_training(
    preset={"embedding": "gcn", "scorer": "mlp"},
    trace_dir=".data/traces/time_sel21",
    weights_dir=".weights",
    configs_dir="configs",
)
```

## Weight Export

Trained PyTorch models are exported to TorchScript for Rust inference:

```python
import torch

# For GCN models, use the export script
# python scripts/export_gcn.py

# For sentence models, use the export method
model.export_torchscript(".weights/sentence_encoder.pt")
```

The tch-rs selector loads TorchScript models:

```rust
use proofatlas::load_gcn_selector;

let selector = load_gcn_selector(
    ".weights/gcn_model.pt",
    true,  // use_cuda
)?;
```

## Inference

During saturation, the selector:

1. Builds graphs/tokenizes clauses
2. Runs forward pass through TorchScript model (GPU-accelerated)
3. Scores clauses with learned scorer
4. Samples from softmax distribution

```rust
// In saturation loop
let selected = selector.select(&mut unprocessed, &clauses);
```

### GPU Acceleration

ML selectors use tch-rs to run inference on GPU when available:

```rust
// Automatically uses CUDA if available
let selector = load_gcn_selector(".weights/gcn_model.pt", true)?;
```

## File Locations

| Path | Description |
|------|-------------|
| `crates/proofatlas/src/ml/` | Rust graph building |
| `crates/proofatlas/src/selectors/` | Rust/tch-rs selector implementations |
| `python/proofatlas/ml/` | Training infrastructure |
| `python/proofatlas/selectors/` | PyTorch model definitions |
| `scripts/export_gcn.py` | GCN TorchScript export |
| `configs/training.json` | Training hyperparameter presets |
| `.weights/` | TorchScript models (.pt files) |
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
- Enable CUDA (`use_cuda=true`)
- Consider sentence model for simpler problems

### Model loading fails
- Verify TorchScript export completed successfully
- Check that PyTorch version matches (tch-rs 0.22 requires PyTorch 2.9)
- Set `LIBTORCH_USE_PYTORCH=1` to use Python's PyTorch libraries
