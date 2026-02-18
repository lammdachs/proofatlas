# Contributing to ProofAtlas

We welcome contributions to ProofAtlas! This document provides guidelines for contributing to the project.

## Research Focus

ProofAtlas is a high-performance theorem prover with ML-guided clause selection. Key areas of interest:

- **Clause selection strategies**: Improving the selection of clauses during saturation
- **Literal selection**: Strategies for selecting literals within clauses
- **ML-guided selection**: Graph neural networks, sentence transformers, and feature-based models for learning clause selection
- **Benchmark evaluation**: Comparing against other provers (Vampire, SPASS)

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/proofatlas.git`
3. Run setup: `python scripts/setup.py`
4. Create a new branch: `git checkout -b feature/your-feature-name`

## Prerequisites

- Python 3.10+
- Rust toolchain (install from https://rustup.rs/)
- PyTorch (for ML features)

## Installation

```bash
# Install Python package (builds Rust extension via maturin)
LIBTORCH_USE_PYTORCH=1 maturin develop

# Or install in editable mode
pip install -e ".[dev]"
```

## Project Structure

```
proofatlas/
├── crates/proofatlas/        # Core theorem prover (Rust)
│   └── src/
│       ├── logic/            # FOL types: core/, ordering/, unification/, interner, clause_manager
│       ├── simplifying/      # Tautology, subsumption, demodulation
│       ├── generating/       # Resolution, superposition, factoring, equality rules
│       ├── index/            # IndexRegistry, SubsumptionChecker, SelectedLiteralIndex, DiscriminationTree
│       ├── prover/           # Saturation engine (Prover struct)
│       ├── selection/        # Clause selection: age_weight, ML pipeline, scoring server
│       ├── parser/           # TPTP parser with FOF→CNF conversion
│       ├── atlas.rs          # ProofAtlas orchestrator
│       ├── state.rs          # SaturationState, StateChange, traits
│       └── config.rs         # ProverConfig
├── crates/proofatlas-wasm/   # WebAssembly bindings
├── python/proofatlas/        # Python package
│   ├── cli/                  # CLI tools: prove, bench, train, web
│   ├── ml/                   # Training loops, data loading, model export
│   └── selectors/            # PyTorch models: GCN, sentence, features, scorers
├── web/                      # SvelteKit web frontend
├── configs/                  # JSON configs: presets, training, problem sets
└── scripts/                  # Setup, benchmarking, training, experiment orchestration
```

## Development Process

### Code Style

**Rust:**
- Run `cargo fmt` before committing
- Run `cargo clippy` to check for common issues

**Python:**
- Use Black for formatting: `black python/`
- Use Ruff for linting: `ruff check python/`
- Use type hints where possible

### Testing

**Rust:**
```bash
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=$(python -c "import torch; print(torch.__path__[0])")/lib

cargo test                    # Run all tests
cargo test --test '*'         # Integration tests only
cargo test -- --nocapture     # With output
```

**Python:**
```bash
pytest python/tests/ -v
```

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb in present tense (e.g., "Add", "Fix", "Update")
- Reference issues when applicable: "Fix #123: Description"

## Pull Request Process

1. Ensure all tests pass (`cargo test` and `pytest`)
2. Add tests for new functionality
3. Request review from maintainers

## Adding New Features

### New Literal Selection Strategies

1. Add implementation in `crates/proofatlas/src/logic/literal_selection.rs`
2. Register in `LiteralSelectionStrategy` enum in `config.rs`
3. Add tests

### New Inference Rules

Generating rules implement `GeneratingInference` (in `state.rs`):

1. Add implementation in `crates/proofatlas/src/generating/`
2. Implement `generate(given_idx, &SaturationState, &mut ClauseManager, &IndexRegistry)`
3. Register in the prover's rule list in `crates/proofatlas/src/prover/mod.rs`
4. Add tests

Simplifying rules implement `SimplifyingInference` (in `state.rs`):

1. Add implementation in `crates/proofatlas/src/simplifying/`
2. Implement `simplify_forward` and/or `simplify_backward`
3. Register in the prover's rule list in `crates/proofatlas/src/prover/mod.rs`
4. Add tests

### Benchmark Configurations

1. Add preset to `configs/proofatlas.json`
2. Test with `proofatlas-bench --config your_preset`

## Benchmarking

Use the benchmark tool to evaluate changes:

```bash
# Run all configs
proofatlas-bench

# Run specific configuration
proofatlas-bench --config time

# Check job status
proofatlas-bench --status
```

Results are saved to `.data/runs/`.

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
