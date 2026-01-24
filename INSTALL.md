# ProofAtlas Installation Guide

## Prerequisites

- Python 3.8 or later
- Rust toolchain (install from https://rustup.rs/)

## Installation

### With GPU Support (Recommended)

Install PyTorch with CUDA first, then install proofatlas:

```bash
# Install PyTorch 2.9 with CUDA (adjust cuda version as needed)
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu124

# Clone and install proofatlas
git clone https://github.com/lexpk/proofatlas.git
cd proofatlas
pip install -e .

# Verify
proofatlas --list
```

### CPU Only

```bash
git clone https://github.com/lexpk/proofatlas.git
cd proofatlas
pip install -e .
```

This installs CPU-only PyTorch automatically.

### What happens during install

- Installs PyTorch 2.9.0 (if not already installed)
- Generates `.cargo/config.toml` with libtorch paths
- Builds Rust extension with ML features

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

This adds pytest, black, ruff, mypy for development.

## Running the Prover

```bash
# Run on a TPTP problem
proofatlas problem.p

# With a config
proofatlas problem.p --config time_sel21

# With options
proofatlas problem.p --timeout 60 --literal-selection 21

# List available configs
proofatlas --list

# Export result to JSON
proofatlas problem.p --json output.json
```

## Web Interface

The web interface runs the prover in the browser using WebAssembly:

```bash
# Install wasm-pack (if not installed)
cargo install wasm-pack

# Build the WASM package
cd crates/proofatlas-wasm
wasm-pack build --target web --out-dir ../../web/pkg

# Serve the web directory
cd ../../web
python -m http.server 8000
```

Then open http://localhost:8000 in your browser.

## Benchmarking

```bash
# Run all configs
proofatlas-bench

# Run specific config
proofatlas-bench --config gcn_mlp_sel21

# Retrain ML models
proofatlas-bench --config gcn_mlp_sel21 --retrain

# Check status of running job
proofatlas-bench --status

# Stop running job
proofatlas-bench --kill

# List available configs
proofatlas-bench --list
```

## Common Issues

1. **Rust not found**
   - Install Rust from https://rustup.rs/

2. **Build fails with libtorch errors**
   - Re-run `pip install -e .` to regenerate `.cargo/config.toml`
   - Ensure PyTorch 2.9.0 is installed

3. **Changed Python environment**
   - Re-run `pip install -e .` to update libtorch paths

## Project Structure

```
proofatlas/
├── crates/proofatlas/   # Core theorem prover (Rust)
├── python/              # Python package and bindings
├── configs/             # Prover and benchmark configurations
├── scripts/             # Utility scripts
└── .tptp/               # TPTP problem library (after setup)
```
