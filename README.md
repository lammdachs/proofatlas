# ProofAtlas: Neural Clause Selection for Theorem Proving

ProofAtlas is a research framework for experimenting with neural guidance in automated theorem proving. It combines a high-performance Rust theorem prover with ML-based clause selection.

## Live Demo

Try ProofAtlas in your browser: **[lexpk.github.io/proofatlas](https://lexpk.github.io/proofatlas)**

## Research Focus

**How can we best represent logical clauses as graphs and learn to select useful clauses during proof search?**

- **Graph representations**: Converting clauses to graphs with node features (type, arity, depth, age, etc.)
- **Graph Neural Networks**: GCN, MLP architectures for learning clause embeddings
- **Clause selection**: Replacing heuristics like age-weight ratio with learned selectors

## Project Structure

```
proofatlas/
│
├── crates/
│   ├── proofatlas/             # Core theorem prover (Rust)
│   │   └── src/
│   │       ├── core/           # Terms, literals, clauses, substitutions, KBO ordering
│   │       ├── inference/      # Inference rules: resolution, superposition, demodulation
│   │       ├── saturation/     # Saturation loop, forward/backward subsumption
│   │       ├── parser/         # TPTP parser with FOF→CNF conversion
│   │       ├── unification/    # Most General Unifier (MGU) computation
│   │       ├── selectors/      # Clause/literal selection strategies (tch-rs ML)
│   │       └── ml/             # Graph building from clauses
│   │
│   └── proofatlas-wasm/        # WebAssembly bindings for browser execution
│
├── python/proofatlas/          # Python package
│   ├── cli/                    # Command-line interface (bench entry point)
│   ├── ml/                     # Training configs, data loading, training loops
│   └── selectors/              # PyTorch model implementations (GCN, Sentence)
│
├── web/                        # Web frontend
│   ├── index.html              # Main prover interface
│   ├── app.js                  # Frontend logic
│   └── style.css               # Styling
│
├── configs/                    # JSON configuration for provers, training, benchmarks
│
├── scripts/                    # Utility scripts
│   ├── setup.py                # One-command project setup
│   ├── bench.py                # Multi-prover benchmarking with trace collection
│   ├── export.py               # Export results for web display
│   └── setup_*.py              # Setup TPTP, Vampire, SPASS
│
├── .data/                      # Runtime data (gitignored)
│   ├── traces/                 # Proof search traces for ML training
│   └── runs/                   # Benchmark results
├── .tptp/                      # TPTP problem library (gitignored)
├── .weights/                   # Trained model weights (gitignored)
├── .vampire/                   # Vampire prover binary (gitignored)
└── .spass/                     # SPASS prover binary (gitignored)
```

## Installation

See [INSTALL.md](INSTALL.md) for detailed instructions.

### Quick Start

```bash
pip install torch  # or with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install proofatlas
```

### From Source

```bash
git clone https://github.com/lexpk/proofatlas.git
cd proofatlas
python -m venv .venv && source .venv/bin/activate
python scripts/setup.py
```

## Usage

### Running the Prover

```bash
# Basic usage
proofatlas .tptp/TPTP-v9.0.0/Problems/PUZ/PUZ001-1.p

# With preset and timeout
proofatlas problem.p --config time_sel21 --timeout 30

# List available presets
proofatlas --list
```

### Benchmarking

```bash
# Run benchmarks with a preset
proofatlas-bench --config time_sel21

# Retrain ML model
proofatlas-bench --config gcn_mlp_sel21 --retrain
```

### Local Web Interface

```bash
# Serve locally (Python)
python -m http.server 8000 --directory web

# Or with Node.js
npx serve web
```

Then open http://localhost:8000 in your browser.

## Clause Selection

### Heuristic Selectors

```rust
use proofatlas::AgeWeightSelector;
let selector = AgeWeightSelector::new(0.5); // 50% age, 50% weight
```

### ML Selectors (tch-rs)

```rust
use proofatlas::load_gcn_selector;
let selector = load_gcn_selector(
    ".weights/gcn_model.pt",
    true,  // use_cuda
)?;
```

ML selectors are enabled by default and use GPU-accelerated inference via tch-rs. CUDA is used automatically when available.

## Node Features

Each clause is converted to a graph with 8-dimensional raw node features:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | Node type | 0-5: clause, literal, predicate, function, variable, constant |
| 1 | Arity | Number of arguments |
| 2 | Arg position | Position as argument to parent |
| 3 | Depth | Distance from clause root |
| 4 | Age | Clause age normalized to [0, 1] |
| 5 | Role | 0-4: axiom, hypothesis, definition, negated_conjecture, derived |
| 6 | Polarity | 1=positive literal, 0=negative |
| 7 | Is equality | 1 if equality predicate, 0 otherwise |

The model's feature embedding layer converts these to a richer representation using one-hot and sinusoidal encodings.

## Tests

```bash
cargo test                        # Rust tests
python -m pytest python/tests/    # Python tests
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

BSD 0-Clause License

## Citation

```bibtex
@software{proofatlas2024,
  title = {ProofAtlas: Neural Clause Selection for Theorem Proving},
  year = {2024},
  url = {https://github.com/lexpk/proofatlas}
}
```
