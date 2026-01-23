# ProofAtlas Installation Guide

## Quick Start

```bash
# Clone the repository
git clone https://github.com/lexpk/proofatlas.git
cd proofatlas

# Install in development mode
pip install -e .

# Build the Rust prover
cargo build --release

# Verify installation
./target/release/prove .tptp/TPTP-v9.0.0/Problems/PUZ/PUZ001-1.p
```

## Prerequisites

- Python 3.7 or later
- Rust toolchain (install from https://rustup.rs/)

## Installation Options

### Basic Installation

```bash
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

This includes pytest, black, ruff, mypy for development.

## Building the Prover

The core theorem prover is written in Rust:

```bash
cargo build --release
```

The binary will be at `target/release/prove`.

### With ML Features (GCN/Sentence selectors)

To build with ML-based clause selection:

```bash
# Install PyTorch first
pip install torch==2.9.0

# Configure Cargo for tch-rs
python scripts/setup_cargo.py

# Install with ML features (--no-build-isolation required)
SETUPTOOLS_RUST_CARGO_FLAGS="--features ml" pip install --no-build-isolation -e ".[ml]"

# Build Rust with ML features
cargo build --release --features ml
```

The `--no-build-isolation` flag is needed because the Rust build uses your
installed PyTorch to find libtorch. Re-run `setup_cargo.py` if you change
Python environments.

## Running the Prover

```bash
# Run on a TPTP problem
./target/release/prove .tptp/TPTP-v9.0.0/Problems/PUZ/PUZ001-1.p

# With options
./target/release/prove problem.p --timeout 60 --literal-selection 21
```

## Benchmarking

After installation, use the benchmark tool:

```bash
# Start benchmark job (runs in background)
proofatlas-bench --prover proofatlas --problem-set test

# Start with live tracking
proofatlas-bench --track

# Check status of running job
proofatlas-bench --status
```

## Exporting Results

Export benchmark results for web display:

```bash
# Export all results (uses default problem set from tptp.json)
proofatlas-export --benchmarks

# Filter by prover and/or preset
proofatlas-export --benchmarks --prover proofatlas
proofatlas-export --benchmarks --prover vampire --preset time_sel21

# Filter by problem set (overrides default)
proofatlas-export --benchmarks --problem-set krs

# Skip learned selectors (only base configs)
proofatlas-export --benchmarks --base-only
```

The default problem set is configured in `configs/tptp.json` under `defaults.problem_set`.
Results are written to `web/data/benchmarks.json`.

## Common Issues

1. **Rust not found**
   - Install Rust from https://rustup.rs/

2. **setuptools-rust not found**
   - Run: `pip install setuptools-rust`

3. **Build fails on older Python**
   - Ensure Python 3.7+ is installed

## Project Structure

```
proofatlas/
├── crates/proofatlas/   # Core theorem prover (Rust)
├── python/              # Python package and bindings
├── configs/             # Prover and benchmark configurations
├── scripts/             # Utility scripts
└── .tptp/               # TPTP problem library (after setup)
```
