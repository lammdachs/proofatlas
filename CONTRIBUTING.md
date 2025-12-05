# Contributing to ProofAtlas

We welcome contributions to ProofAtlas! This document provides guidelines for contributing to the project.

## Research Focus

ProofAtlas is a high-performance theorem prover with ML-guided clause selection. Key areas of interest:

- **Clause selection strategies**: Improving the selection of clauses during saturation
- **Literal selection**: Strategies for selecting literals within clauses
- **Graph neural networks**: Learning clause representations for ML-guided selection
- **Benchmark evaluation**: Comparing against other provers (Vampire, SPASS)

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/proofatlas.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`

## Prerequisites

- Python 3.7+
- Rust toolchain (install from https://rustup.rs/)

## Installation

```bash
# Install Python package
pip install -e ".[dev]"

# Build Rust prover
cd rust && cargo build --release
```

## Project Structure

```
proofatlas/
├── rust/                     # Core theorem prover
│   ├── src/
│   │   ├── core/            # Terms, literals, clauses, substitutions
│   │   ├── inference/       # Resolution, superposition, demodulation
│   │   ├── saturation/      # Saturation loop, subsumption
│   │   ├── parser/          # TPTP parser
│   │   └── selection/       # Clause/literal selection strategies
│   └── tests/
├── python/                   # Python package
│   ├── proofatlas/
│   │   ├── cli/             # Command-line tools
│   │   └── ml/              # ML training infrastructure
│   └── tests/
├── configs/                  # Prover and benchmark configurations
├── scripts/                  # Utility scripts (bench.py, etc.)
└── .tptp/                    # TPTP problem library
```

## Development Process

### Code Style

**Rust:**
- Run `cargo fmt` before committing
- Run `cargo clippy` to check for common issues
- Write doc comments for public items

**Python:**
- Use Black for formatting: `black python/`
- Use Ruff for linting: `ruff check python/`
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
python -m pytest python/tests/ -v
```

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb in present tense (e.g., "Add", "Fix", "Update")
- Reference issues when applicable: "Fix #123: Description"

Example:
```
Add literal selection strategy 21

- Implement unique maximal literal selection
- Fall back to negative max-weight when no unique maximal
- Add tests for edge cases
```

## Pull Request Process

1. Ensure all tests pass (`cargo test` and `pytest`)
2. Update documentation if needed
3. Add tests for new functionality
4. Request review from maintainers

## Adding New Features

### New Literal Selection Strategies

1. Add implementation in `rust/src/selection/literal.rs`
2. Register in the CLI options
3. Add tests in `rust/tests/`
4. Document the strategy behavior

### New Inference Rules

1. Add implementation in `rust/src/inference/`
2. Create tests in `rust/tests/`
3. Integrate with saturation loop in `rust/src/saturation/state.rs`

### Benchmark Configurations

1. Add preset to `configs/proofatlas.json`
2. Test with `proofatlas-bench --preset your_preset`
3. Document the configuration

## Benchmarking

Use the benchmark tool to evaluate changes:

```bash
# Run all provers with all presets
proofatlas-bench --track

# Run specific configuration
proofatlas-bench --prover proofatlas --preset time_sel21 --track

# Check job status
proofatlas-bench --status
```

Results are saved to `.logs/eval_TIMESTAMP/`.

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
