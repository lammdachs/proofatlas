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
# Build the Rust prover
cargo build --release

# Install Python package with ML dependencies
pip install -e ".[ml]"

# Setup TPTP problem library
python scripts/setup_tptp.py
```

## Usage

### Running the Prover

```bash
./target/release/prove .tptp/TPTP-v9.0.0/Problems/PUZ/PUZ001-1.p --timeout 30
```

### Benchmarking

```bash
python scripts/bench.py --prover proofatlas --preset quick
```

### Training ML Models

```bash
# Collect traces
python scripts/bench.py --prover proofatlas --trace

# Train a GCN selector
python scripts/train.py --traces steps_sel22 --model gcn
```

### Local Web Interface

```bash
# Export benchmark and training data to web/data/
proofatlas-export

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

ML selectors require the `torch` feature and use GPU-accelerated inference via tch-rs.

## Node Features

Each clause is converted to a graph with 13-dimensional node features:

| Index | Feature | Description |
|-------|---------|-------------|
| 0-5 | Node type | One-hot: clause, literal, predicate, function, variable, constant |
| 6 | Arity | Number of arguments |
| 7 | Arg position | Position as argument to parent |
| 8 | Depth | Distance from clause root |
| 9 | Age | Clause age normalized to [0, 1] |
| 10 | Role | axiom/hypothesis/negated_conjecture/derived |
| 11 | Polarity | positive/negative |
| 12 | Is equality | Whether predicate is equality |

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
