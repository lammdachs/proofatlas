# Contributing to ProofAtlas

We welcome contributions to ProofAtlas! This document provides guidelines for contributing to the project.

## Research Focus

ProofAtlas is a research framework for neural clause selection in theorem proving. Key areas of interest:

- **Graph embeddings**: GCN, GAT, GraphSAGE, and other architectures for clause representation
- **Transformers/Attention**: Scoring clauses using attention mechanisms over graph embeddings
- **Training data**: Collecting and processing proof traces for supervised learning
- **Benchmark expansion**: Adding more TPTP problems and prover comparisons

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/proofatlas.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`

## Prerequisites

- **Conda/Micromamba**: Package manager (install via [miniforge](https://github.com/conda-forge/miniforge) or [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html))

## Installation & Build

### Development Environment (Recommended)

Use micromamba/conda to install all dependencies including Rust:

```bash
# Create environment with all dependencies (CPU)
micromamba create -n proofatlas \
    python=3.11 \
    rust=1.85 \
    rust-std-wasm32-unknown-unknown=1.85 \
    pytorch-cpu \
    onnx \
    onnxscript \
    pytest \
    -c conda-forge

# Activate environment
micromamba activate proofatlas
```

### GPU Support (Optional)

For training with CUDA GPUs (single-node multi-GPU).

**Requirements:**
- NVIDIA GPU with CUDA support
- NVIDIA driver installed (check with `nvidia-smi`)
- Driver version must support CUDA 12.4+ (driver 550+ recommended)

```bash
# Create environment with CUDA support
micromamba create -n proofatlas-gpu \
    python=3.11 \
    rust=1.85 \
    rust-std-wasm32-unknown-unknown=1.85 \
    pytorch-cuda=12.4 \
    onnx \
    onnxscript \
    pytest \
    -c pytorch -c nvidia -c conda-forge

# Activate environment
micromamba activate proofatlas-gpu

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

Multi-GPU training with PyTorch DDP:

```bash
# Train on all available GPUs
torchrun --standalone --nproc_per_node=gpu train.py

# Train on specific GPUs (e.g., GPU 0 and 1)
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py
```

For PyTorch Lightning, set the trainer config:

```python
from pytorch_lightning import Trainer

trainer = Trainer(
    accelerator="gpu",
    devices=-1,  # Use all available GPUs
    strategy="ddp",  # Distributed Data Parallel
)
```

### Rust (Core Prover)

```bash
cd rust

# Build release binary
cargo build --release

# Run the prover
./target/release/prove path/to/problem.p --timeout 30

# Run tests
cargo test
```

### Python (ML Training)

```bash
# Activate environment
micromamba activate proofatlas

# Run tests
cd python && python -m pytest tests/ -v
```

### WebAssembly

```bash
# Activate environment (includes wasm32 target)
micromamba activate proofatlas

# Install wasm-bindgen-cli
cargo install wasm-bindgen-cli --version 0.2.100 --locked

# Build WASM module
cd wasm
cargo build --target wasm32-unknown-unknown --release

# Generate JS bindings
wasm-bindgen target/wasm32-unknown-unknown/release/proofatlas_wasm.wasm \
    --out-dir pkg \
    --target web \
    --no-typescript

# Serve locally
python -m http.server 8000
# Open http://localhost:8000 in browser
```

### Generate ONNX Models

```bash
# Activate environment (includes pytorch and onnx)
micromamba activate proofatlas

# Generate age-weight selector model
python -c "
import torch
import torch.nn as nn

class AgeWeightSelector(nn.Module):
    def __init__(self, age_probability=0.5):
        super().__init__()
        self.register_buffer('p', torch.tensor(age_probability))

    def forward(self, node_features, pool_matrix):
        clause_features = torch.mm(pool_matrix, node_features)
        ages = clause_features[:, 9]
        weights = clause_features[:, 8]
        num_clauses = clause_features.size(0)
        oldest_idx = torch.argmax(ages)
        lightest_idx = torch.argmin(weights)
        indices = torch.arange(num_clauses, device=clause_features.device)
        oldest_mask = (indices == oldest_idx).float()
        lightest_mask = (indices == lightest_idx).float()
        log_p = torch.log(self.p + 1e-10)
        log_1mp = torch.log(1 - self.p + 1e-10)
        logits_diff = oldest_mask * log_p + lightest_mask * log_1mp + (1 - oldest_mask) * (1 - lightest_mask) * (-1e9)
        logits_same = oldest_mask * 0.0 + (1 - oldest_mask) * (-1e9)
        same_clause = (oldest_idx == lightest_idx)
        return torch.where(same_clause, logits_same, logits_diff)

model = AgeWeightSelector(age_probability=0.5)
model.eval()
torch.onnx.export(
    model, (torch.randn(50, 13), torch.randn(10, 50)),
    'selector.onnx',
    input_names=['node_features', 'pool_matrix'],
    output_names=['scores'],
    dynamic_axes={
        'node_features': {0: 'total_nodes'},
        'pool_matrix': {0: 'num_clauses', 1: 'total_nodes'},
        'scores': {0: 'num_clauses'},
    },
    opset_version=14,
)
print('Generated selector.onnx')
"
```

## Project Structure

```
proofatlas/
├── rust/                     # Core theorem prover
│   ├── src/
│   │   ├── core/            # Terms, literals, clauses
│   │   ├── inference/       # Resolution, superposition
│   │   ├── saturation/      # Saturation loop
│   │   ├── ml/              # Graph building, ONNX inference
│   │   └── selection/       # Clause/literal selection
│   └── tests/
├── python/                   # ML training pipeline
│   ├── proofatlas/
│   │   └── ml/              # PyTorch models
│   └── tests/
├── wasm/                     # Web interface
└── docs/                     # Documentation
```

## Development Process

### Code Style

**Rust:**
- Run `cargo fmt` before committing
- Run `cargo clippy` to check for common issues
- Write doc comments for public items

**Python:**
- Use Black for formatting: `black python/`
- Follow PEP 8 guidelines
- Use type hints where possible

### Testing

**Rust:**
```bash
cd rust
cargo test                    # Run all tests
cargo test --test '*'         # Integration tests only
cargo test -- --nocapture     # With output
```

**Python:**
```bash
cd python
python -m pytest tests/ -v
```

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb in present tense (e.g., "Add", "Fix", "Update")
- Reference issues when applicable: "Fix #123: Description"

Example:
```
Add GAT-based clause embedding

- Implement graph attention layer for clause graphs
- Add pooling to aggregate node embeddings
- Export model to ONNX format
```

## Pull Request Process

1. Ensure all tests pass (`cargo test` and `pytest`)
2. Update documentation if needed
3. Add tests for new functionality
4. Request review from maintainers

## Adding New Features

### New Graph Embedding Architectures

1. Add PyTorch model in `python/proofatlas/ml/model.py`
2. Ensure forward signature matches: `forward(node_features, pool_matrix) -> scores`
3. Export to ONNX with dynamic axes (see `docs/onnx_interface.md`)
4. Test with the Rust inference in `rust/src/ml/inference.rs`

### New Node Features

1. Update `rust/src/ml/graph.rs` with new feature constants
2. Update `FEATURE_DIM` constant
3. Update documentation in `docs/onnx_interface.md`
4. Regenerate ONNX models with new dimensions

### New Inference Rules

1. Add implementation in `rust/src/inference/`
2. Create tests in `rust/tests/`
3. Integrate with saturation loop in `rust/src/saturation/state.rs`

### Benchmark Problems

1. Add problem files to appropriate test directories
2. Update benchmark results in `wasm/index.html` if applicable
3. Document problem characteristics

## ONNX Model Requirements

Models must follow the interface in `docs/onnx_interface.md`:

- **Inputs**: `node_features [total_nodes, 13]`, `pool_matrix [num_clauses, total_nodes]`
- **Output**: `scores [num_clauses]`
- **Dynamic axes**: All batch dimensions must be dynamic
- **Opset version**: 14 or compatible

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Respect differing opinions

## Questions?

Feel free to open an issue for:
- Bug reports
- Feature requests
- Questions about the codebase
- Discussion of research ideas

Thank you for contributing to ProofAtlas!
