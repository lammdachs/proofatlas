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
maturin develop

# Verify
proofatlas --list
```

### CPU Only

```bash
git clone https://github.com/lexpk/proofatlas.git
cd proofatlas
maturin develop
```

This installs CPU-only PyTorch automatically.

### What happens during install

- Installs PyTorch 2.9.0 (if not already installed)
- Builds Rust extension with Python bindings and ML features (both are default Cargo features)
- At runtime, `proofatlas/__init__.py` preloads libtorch (CPU + CUDA if available) from your PyTorch installation

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

The web interface runs the prover in the browser using WebAssembly.
The WASM package is built automatically during installation.

```bash
# Start the web server
proofatlas-web

# Or on a custom port
proofatlas-web --port 3000

# Stop the server
proofatlas-web --kill
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

## Developer Setup

For development with `cargo test` and `cargo build`, you need `.cargo/config.toml` to point at your PyTorch installation so the linker can find libtorch at runtime:

```bash
python scripts/setup_cargo.py
```

Re-run this after switching Python environments. This is only needed for the `cargo` workflow; `pip install -e .` handles it automatically.

## Common Issues

1. **Rust not found**
   - Install Rust from https://rustup.rs/

2. **Build fails with libtorch errors**
   - Ensure PyTorch 2.9.0 is installed
   - Run `python scripts/setup_cargo.py` to regenerate `.cargo/config.toml`

3. **`cargo test` fails with `libtorch_cpu.so: cannot open shared object file`**
   - Run `python scripts/setup_cargo.py` to update the rpath in `.cargo/config.toml`

4. **Changed Python environment**
   - Run `python scripts/setup_cargo.py` to update libtorch paths
   - Re-run `pip install -e .` if using the Python package

## Project Structure

```
proofatlas/
├── crates/proofatlas/   # Core theorem prover (Rust)
├── python/              # Python package and bindings
├── configs/             # Prover and benchmark configurations
├── scripts/             # Utility scripts
└── .tptp/               # TPTP problem library (after setup)
```
