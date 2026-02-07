# ProofAtlas Installation Guide

## Prerequisites

- Python 3.10 or later

## Installation (pip)

Install PyTorch first (to choose your CUDA version), then install proofatlas:

```bash
# GPU (adjust CUDA version as needed, see https://pytorch.org/)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install proofatlas

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install proofatlas
```

## Installation (from source)

For development or if no pre-built wheel is available for your platform:

```bash
# Clone
git clone https://github.com/lexpk/proofatlas.git
cd proofatlas

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Run setup (installs deps, auto-detects CUDA, builds extension)
python scripts/setup.py

# Verify
proofatlas --list
```

The setup script handles everything: installs Rust (if missing), Python dependencies, configures libtorch, and builds the extension.

### Setup options

```bash
python scripts/setup.py                  # Full setup (auto-detects GPU)
python scripts/setup.py --cpu            # Force CPU-only torch
python scripts/setup.py --skip-external  # Skip TPTP/Vampire/SPASS downloads
python scripts/setup.py --skip-wasm      # Skip WASM build
```

## Running the Prover

```bash
# Run on a TPTP problem
proofatlas problem.p

# With a config
proofatlas problem.p --config time

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
proofatlas-bench --config gcn_mlp

# Retrain ML models
proofatlas-bench --config gcn_mlp --retrain

# Check status of running job
proofatlas-bench --status

# Stop running job
proofatlas-bench --kill

# List available configs
proofatlas-bench --list
```

## Common Issues

1. **No virtual environment**
   - The setup script requires an active venv. See instructions above.

2. **Rust not found**
   - Install Rust from https://rustup.rs/

3. **Build fails with libtorch errors**
   - Ensure PyTorch >=2.9 is installed
   - Re-run `python scripts/setup.py` to regenerate `.cargo/config.toml`

4. **`cargo test` fails with `libtorch_cpu.so: cannot open shared object file`**
   - Run `python scripts/setup_libtorch.py` to update the rpath in `.cargo/config.toml`

5. **Changed Python environment**
   - Run `python scripts/setup.py` in the new environment

## Project Structure

```
proofatlas/
├── crates/proofatlas/   # Core theorem prover (Rust)
├── python/              # Python package and bindings
├── configs/             # Prover and benchmark configurations
├── scripts/             # Utility scripts
└── .tptp/               # TPTP problem library (after setup)
```
