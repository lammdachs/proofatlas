# ProofAtlas: Neural Clause Selection for Theorem Proving

ProofAtlas is a research framework for experimenting with neural guidance in automated theorem proving. The project focuses on comparing different **graph embeddings** and **transformer/attention mechanisms** for clause selection during saturation-based proof search.

## Research Focus

The core research question: **How can we best represent logical clauses as graphs and learn to select useful clauses during proof search?**

Key areas of investigation:
- **Graph representations**: Converting clauses to graphs with node features (type, arity, depth, age, etc.)
- **Graph Neural Networks**: Learning embeddings from clause structure
- **Transformers/Attention**: Scoring clauses using attention mechanisms over graph embeddings
- **Clause selection**: Replacing heuristics like age-weight ratio with learned selectors

## Architecture

```
Clauses → Graph Representation → GNN/Transformer → Logits → Softmax Sampling → Selected Clause
```

Each clause is converted to a graph where nodes represent syntactic elements (clause, literal, predicate, function, variable, constant). Nodes have a **13-dimensional feature vector** encoding type, arity, argument position, depth, age, role, polarity, and equality status.

## Live Demo

Try ProofAtlas in your browser: **[lexpk.github.io/proofatlas](https://lexpk.github.io/proofatlas)**

The web demo runs entirely client-side using WebAssembly with ONNX-based neural clause selection.

## Key Features

- **High-performance Rust core**: Saturation-based theorem prover with superposition calculus
- **ONNX inference**: Pure Rust inference via tract-onnx (works in native and WASM)
- **Graph-based clause representation**: 13-dimensional node features for ML models
- **Softmax sampling**: Probabilistic clause selection from model logits
- **TPTP support**: Parser for standard theorem proving problem format
- **WebAssembly build**: Run the prover in browsers with ML-guided selection

## Project Structure

```
proofatlas/
├── rust/                     # Core Rust implementation
│   ├── src/
│   │   ├── core/            # Terms, literals, clauses, substitutions
│   │   ├── inference/       # Resolution, factoring, superposition
│   │   ├── saturation/      # Saturation loop, subsumption
│   │   ├── ml/              # Graph building, ONNX inference
│   │   ├── parser/          # TPTP parser
│   │   └── selection/       # Clause/literal selection strategies
│   └── tests/
├── python/                   # Python bindings and ML training
│   ├── proofatlas/
│   │   └── ml/              # PyTorch models, training pipeline
│   └── tests/
├── wasm/                     # WebAssembly frontend
│   ├── index.html           # Web prover interface
│   └── models/              # ONNX models for web
├── docs/                     # Documentation
│   └── onnx_interface.md    # ONNX model specification
└── .selectors/              # Pre-trained ONNX selectors
```

## Installation

### Rust (Core Prover)

```bash
cd rust
cargo build --release
./target/release/prove problem.p --timeout 30
```

### Python (ML Training)

```bash
# Install with PyTorch for training
pip install -e ".[ml]"

# Or minimal install for bindings only
pip install -e .
```

### WebAssembly

The WASM build is automatically deployed to GitHub Pages. To build locally:

```bash
cd wasm
cargo build --target wasm32-unknown-unknown --release
wasm-bindgen target/wasm32-unknown-unknown/release/proofatlas_wasm.wasm \
    --out-dir pkg --target web
```

## Using ONNX Models

### From File Path (Rust)

```rust
use proofatlas::OnnxClauseSelector;

let selector = OnnxClauseSelector::new("path/to/model.onnx")?;
state.set_clause_selector(Box::new(selector));
```

### Training Custom Models (Python)

```python
import torch
import torch.nn as nn

class MyClauseSelector(nn.Module):
    def __init__(self, feature_dim=13, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Linear(feature_dim, hidden_dim)
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, node_features, pool_matrix):
        # node_features: [total_nodes, 13]
        # pool_matrix: [num_clauses, total_nodes]
        node_emb = torch.relu(self.encoder(node_features))
        clause_emb = torch.mm(pool_matrix, node_emb)
        return self.scorer(clause_emb).squeeze(-1)

# Export to ONNX
model = MyClauseSelector()
torch.onnx.export(
    model,
    (torch.randn(50, 13), torch.randn(10, 50)),
    "selector.onnx",
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

See [docs/onnx_interface.md](docs/onnx_interface.md) for the complete feature specification.

## Node Feature Layout

| Index | Feature | Description |
|-------|---------|-------------|
| 0-5 | Node type | One-hot: clause, literal, predicate, function, variable, constant |
| 6 | Arity | Number of arguments (predicates/functions) |
| 7 | Arg position | 0-indexed position as argument to parent |
| 8 | Depth | Distance from clause root |
| 9 | Age | Clause age normalized to [0, 1] |
| 10 | Role | 0=axiom, 1=hypothesis, 2=definition, 3=negated_conjecture, 4=derived |
| 11 | Polarity | 1.0=positive, 0.0=negative |
| 12 | Is equality | 1.0 if predicate is equality |

## Benchmarks

On 150 TPTP Unsatisfiable problems (10s timeout):

| Category | ProofAtlas | Vampire 5.0 |
|----------|------------|-------------|
| Unit Equalities | 18/50 (36%) | 23/50 (46%) |
| CNF Without Equality | 16/50 (32%) | 16/50 (32%) |
| CNF With Equality | 7/50 (14%) | 11/50 (22%) |
| **Total** | **41/150 (27%)** | **50/150 (33%)** |

## Running Tests

```bash
# Rust tests
cd rust && cargo test

# Python tests
cd python && python -m pytest tests/ -v
```

## Contributing

Contributions are welcome! Areas of interest:
- New graph embedding architectures (GCN, GAT, GraphSAGE)
- Transformer-based clause scoring
- Training data collection from proof traces
- Benchmark expansion

## License

BSD 0-Clause License - see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{proofatlas2024,
  title = {ProofAtlas: Neural Clause Selection for Theorem Proving},
  year = {2024},
  url = {https://github.com/apluska/proofatlas}
}
```
