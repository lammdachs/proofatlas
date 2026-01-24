# ProofAtlas Installation Guide

## Quick Start

```bash
# Clone the repository
git clone https://github.com/lexpk/proofatlas.git
cd proofatlas

# Install in development mode
pip install -e .

# Verify installation
proofatlas --list
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

### With ML Features (GCN/Sentence selectors)

To use ML-based clause selection:

```bash
# Install PyTorch first
pip install torch==2.9.0

# Configure Cargo for tch-rs
python scripts/setup_cargo.py

# Build with ML features
cargo build --release --features python,ml -p proofatlas

# Copy the extension to the Python package
cp target/release/libproofatlas.so python/proofatlas/proofatlas.cpython-312-x86_64-linux-gnu.so

# Install Python dependencies
pip install -e ".[ml]"
```

Re-run `setup_cargo.py` if you change Python environments.

## Running the Prover

```bash
# Run on a TPTP problem
proofatlas problem.p

# With a preset
proofatlas problem.p --config time_sel21

# With options
proofatlas problem.p --timeout 60 --literal-selection 21

# List available presets
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

After installation, use the benchmark tool:

```bash
# Run all presets
proofatlas-bench

# Run specific preset
proofatlas-bench --config gcn_mlp_sel21

# Retrain ML models
proofatlas-bench --config gcn_mlp_sel21 --retrain

# Check status of running job
proofatlas-bench --status

# Stop running job
proofatlas-bench --kill

# List available presets
proofatlas-bench --list
```

## Common Issues

1. **Rust not found**
   - Install Rust from https://rustup.rs/

2. **setuptools-rust not found**
   - Run: `pip install setuptools-rust`

3. **Build fails on older Python**
   - Ensure Python 3.7+ is installed

4. **ML features not working**
   - Ensure PyTorch is installed before building
   - Run `python scripts/setup_cargo.py` to configure libtorch paths
   - Rebuild with `cargo build --release --features python,ml -p proofatlas`

## Project Structure

```
proofatlas/
├── crates/proofatlas/   # Core theorem prover (Rust)
├── python/              # Python package and bindings
├── configs/             # Prover and benchmark configurations
├── scripts/             # Utility scripts
└── .tptp/               # TPTP problem library (after setup)
```
