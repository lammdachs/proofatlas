# Optional Dependencies Guide

ProofAtlas is designed with a modular architecture where the core theorem proving functionality works without machine learning dependencies. This guide explains the optional components and when you might need them.

## Core Functionality (Always Installed)

The base installation provides a complete theorem proving system:

- **Saturation-based theorem prover**: Implementation of the given clause algorithm
- **Inference rules**: Resolution, factoring, subsumption checking
- **Proof tracking**: Complete proof history with rule applications
- **Basic clause selection**: FIFO and random selection strategies
- **Parser support**: TPTP problem format
- **Proof visualization**: Terminal-based proof navigator and printer
- **First-order logic**: Full support for variables, functions, and predicates

## Optional: PyTorch and Graph Neural Networks

### When You Need It

Install PyTorch and related packages if you want to:
- Use GNN-based clause selection (requires trained models)
- Train new clause selection models on proof data
- Experiment with neural proof guidance
- Research learned heuristics for theorem proving

### What It Includes

- **PyTorch**: Core deep learning framework
- **PyTorch Geometric (PyG)**: Graph neural network library
- **PyTorch Lightning**: Training framework with logging and checkpointing
- **CUDA support**: GPU acceleration for model training and inference

### Installation

PyTorch is included in the base dependencies but will install CPU version by default. For GPU support:

```bash
# Install PyTorch with CUDA support (example for CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CPU-only (if you need to reinstall)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Usage Example

```python
from proofatlas.selectors.gnn import GNNSelector

# Requires trained model file
selector = GNNSelector(model_path="path/to/model.pt")
loop = BasicLoop()
# Use GNN for clause selection during proof search
```


## Choosing What to Install

### For Theorem Proving Research (Core Only)
If you're researching classical automated theorem proving, inference rules, or proof search strategies, the core installation is sufficient.

### For Machine Learning Research (Core + PyTorch)
If you're researching neural theorem proving, learning proof guidance, or combining symbolic and neural methods, install PyTorch and related packages.

### For Development
Install the development dependencies with `pip install -e ".[dev]"` to get testing and code quality tools.

## System Requirements

### Core Installation
- **RAM**: 4GB minimum, 8GB recommended
- **Disk**: 2GB for environment and dependencies
- **GPU**: Not required

### With PyTorch
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 5GB for environment and dependencies
- **GPU**: NVIDIA GPU with 4GB+ VRAM recommended
- **CUDA**: Version 12.1 compatible drivers


## TPTP Library

The TPTP (Thousands of Problems for Theorem Provers) library is an optional but recommended resource containing thousands of theorem proving problems for testing and benchmarking.

### Installation

ProofAtlas includes a command to download TPTP:

```bash
# Download to default location (.data/problems/tptp)
proofatlas-download-tptp

# Download to custom location
proofatlas-download-tptp --data-dir /path/to/tptp
```

### Manual Installation

Alternatively, you can install TPTP manually:

```bash
# Download the latest version from
# https://www.tptp.org/TPTP/Distribution/

# Extract to your preferred location
tar -xzf TPTP-vX.Y.Z.tgz

# Update TPTP_PATH in your .env file
echo "TPTP_PATH=/path/to/tptp" >> .env
```

## Troubleshooting

### PyTorch Installation Issues

If CUDA version conflicts occur:
```bash
# Check your CUDA version
nvidia-smi

# Install CPU-only version if no GPU
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Import Errors

If you see `ImportError: No module named torch`:
- The code is trying to use GNN features without PyTorch installed
- Either install PyTorch or use a non-GNN selector (FIFO, Random)

