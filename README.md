# ProofAtlas: Neural Clause Selection for Theorem Proving

ProofAtlas is a high-performance theorem prover for first-order logic with equality, combining saturation-based superposition calculus with ML-guided clause selection. Written in Rust with Python bindings, it supports multiple neural architectures for learning clause selection strategies from proof traces.

## Live Demo

Try ProofAtlas in your browser: **[lammdachs.github.io/proofatlas](https://lammdachs.github.io/proofatlas)**

## Research Focus

**Can neural networks learn to select useful clauses during saturation-based proof search?**

- **Clause encoders**: Graph neural networks (GCN), sentence transformers (MiniLM), and clause feature vectors
- **Scoring architectures**: MLP, multi-head attention, transformer, and cross-attention heads
- **Pipelined inference**: Asynchronous ML scoring overlapped with proof search
- **Trace-based training**: Per-clause lifecycle arrays enable step-level supervision from proof search traces

## Project Structure

```
proofatlas/
├── crates/
│   ├── proofatlas/                # Core theorem prover (Rust)
│   │   └── src/
│   │       ├── logic/             # FOL types: core/, ordering/, unification/, interner, clause_manager
│   │       ├── simplifying/       # Tautology elimination, subsumption, demodulation
│   │       ├── generating/        # Resolution, superposition, factoring, equality rules
│   │       ├── index/             # SubsumptionChecker, SelectedLiteralIndex, DiscriminationTree
│   │       ├── prover/            # Saturation engine (prove/init/step/saturate)
│   │       ├── selection/         # Clause selection: age-weight, ML pipeline, scoring server
│   │       │   ├── ml/            # GCN, sentence, features encoders (tch-rs)
│   │       │   ├── pipeline/      # Backend compute service, ChannelSink
│   │       │   ├── network/       # Scoring protocol, RemoteSelector, ScoringServer
│   │       │   └── training/      # NPZ trace writer, proof trace extraction
│   │       ├── parser/            # TPTP parser with FOF→CNF conversion
│   │       ├── atlas.rs           # ProofAtlas orchestrator (reusable across problems)
│   │       ├── config.rs          # ProverConfig, LiteralSelectionStrategy
│   │       └── state.rs           # SaturationState, StateChange, ProofResult, inference traits
│   │
│   └── proofatlas-wasm/           # WebAssembly bindings for browser execution
│
├── python/proofatlas/             # Python package
│   ├── cli/                       # CLI tools: prove, bench, train, web
│   ├── ml/                        # Training loops, data loading, model export
│   └── selectors/                 # PyTorch models: GCN, sentence, features, scorers
│
├── web/                           # SvelteKit web frontend (Tailwind CSS, Chart.js)
├── configs/                       # JSON configs: presets, training, problem sets, embeddings, scorers
├── scripts/                       # Setup, benchmarking, training, experiment orchestration
├── .data/                         # Runtime data (gitignored)
│   ├── traces/                    # Proof search traces (NPZ)
│   └── runs/                      # Benchmark results
├── .tptp/                         # TPTP problem library (gitignored)
├── .weights/                      # Trained model weights (gitignored)
├── .vampire/                      # Vampire prover binary (gitignored)
└── .spass/                        # SPASS prover binary (gitignored)
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
proofatlas problem.p                              # Basic usage
proofatlas problem.p --config time                 # With preset (10s timeout)
proofatlas problem.p --timeout 30                  # Custom timeout
proofatlas problem.p --literal-selection 21         # Literal selection strategy
proofatlas problem.p --json output.json            # Export result to JSON
proofatlas --list                                  # List available presets
```

### Training ML Models

```bash
proofatlas-train --config gcn_mlp                  # Train GCN + MLP scorer
proofatlas-train --config sentence_attention        # Train sentence + attention scorer
proofatlas-train --config gcn_mlp --use-cuda       # Train on GPU
proofatlas-train --config gcn_mlp --max-epochs 4   # Short test run
```

### Benchmarking

```bash
proofatlas-bench                                   # Run all presets
proofatlas-bench --config gcn_mlp                  # Run specific preset
proofatlas-bench --config gcn_mlp --retrain        # Retrain before benchmarking
proofatlas-bench --status                          # Check running jobs
proofatlas-bench --kill                            # Stop running jobs
```

### Web Interface

```bash
proofatlas-web                                     # Start on default port
proofatlas-web --port 3000                         # Custom port
proofatlas-web --kill                              # Stop the server
```

## ML Architecture

ProofAtlas supports a modular encoder + scorer architecture. Models are trained in PyTorch, exported to TorchScript, and loaded at inference time via tch-rs.

### Encoders

| Encoder | Description |
|---------|-------------|
| `gcn` | Graph Convolutional Network over clause structure (node type, arity, depth, polarity, etc.) |
| `gcn_struct` | GCN with structural features only (no symbol embeddings) |
| `features` | 9-dimensional clause feature vector (sinusoidal + one-hot encoding → MLP) |
| `sentence` | Sentence transformer (MiniLM) encoding clause text as 384-D embeddings |

### Scorers

| Scorer | Description |
|--------|-------------|
| `mlp` | Feed-forward network |
| `attention` | Multi-head self/cross-attention with learnable sentinel |
| `transformer` | Full transformer block with cross-attention |
| `cross_attention` | Dot-product cross-attention |

Any encoder can be combined with any scorer. Presets are defined in `configs/proofatlas.json` and follow the pattern `{encoder}_{scorer}` (e.g., `gcn_mlp`, `sentence_attention`, `features_transformer`).

### Inference Pipeline

During proof search, clause scoring runs asynchronously in a pipelined architecture:

```
Prover → ChannelSink → Data Processing Thread → Backend (CPU/GPU model)
          (signals)      (embedding cache,         (TorchScript)
                          softmax sampling)
```

For multi-GPU setups, a socket-based scoring server is available via `--gpu-workers`.

## Superposition Calculus

ProofAtlas implements the superposition calculus for first-order logic with equality:

- **Generating rules**: Binary resolution, superposition (left/right), factoring, equality resolution, equality factoring
- **Simplifying rules**: Tautology elimination, forward/backward subsumption, forward/backward demodulation
- **Term ordering**: Knuth-Bendix Ordering (KBO)
- **Literal selection**: Four strategies (levels 0, 20, 21, 22) based on [Hoder et al. 2016](https://doi.org/10.1007/978-3-319-40229-1_18)
- **Indexing**: Feature vector index for subsumption, discrimination tree for demodulation, selected literal index for inference candidate filtering

## Tests

```bash
# Rust tests (requires PyTorch environment)
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=$(python -c "import torch; print(torch.__path__[0])")/lib
cargo test

# Python tests
pytest python/tests/ -v
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License

## Citation

```bibtex
@software{proofatlas,
  title = {ProofAtlas: Neural Clause Selection for Theorem Proving},
  author = {Pluska, Alexander},
  year = {2025},
  url = {https://github.com/lexpk/proofatlas}
}
```
