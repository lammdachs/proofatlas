# ProofAtlas Installation Guide

## Quick Start

```bash
# Clone the repository
git clone https://github.com/lexpk/proofatlas.git
cd proofatlas

# Install in development mode
pip install -e .

# Build the Rust prover
cd rust && cargo build --release && cd ..

# Verify installation
./rust/target/release/prove --help
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
cd rust
cargo build --release
```

The binary will be at `rust/target/release/prove`.

## Running the Prover

```bash
# Run on a TPTP problem
./rust/target/release/prove path/to/problem.p

# With options
./rust/target/release/prove problem.p --timeout 60 --literal-selection 21
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
proofatlas-export --benchmarks --prover vampire --preset time_sel22

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
├── rust/           # Core theorem prover (Rust)
├── python/         # Python package and bindings
├── configs/        # Prover and benchmark configurations
├── scripts/        # Utility scripts
└── .tptp/          # TPTP problem library (after setup)
```
