# Optional Dependencies Guide

ProofAtlas is designed with a modular architecture where the core theorem proving functionality works without machine learning dependencies. This guide explains the optional components and when you might need them.

## Core Functionality (Always Installed)

The base installation provides a complete theorem proving system:

- **Saturation-based theorem prover**: Implementation of the given clause algorithm
- **Inference rules**: Resolution, factoring, subsumption checking
- **Proof tracking**: Complete proof history with rule applications
- **Basic clause selection**: FIFO and random selection strategies
- **Parser support**: TPTP problem format and Vampire proof format
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

During setup, answer "yes" when prompted, or install manually:

```bash
conda activate proofatlas
conda install -y pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y pyg -c pyg
conda install -y pytorch-lightning torchmetrics -c conda-forge
```

### Usage Example

```python
from proofatlas.selectors.gnn import GNNSelector

# Requires trained model file
selector = GNNSelector(model_path="path/to/model.pt")
loop = BasicLoop()
# Use GNN for clause selection during proof search
```

## Optional: Claude CLI

### When You Need It

Install Claude CLI if you want:
- Interactive AI assistance while developing
- Help understanding and modifying the codebase
- Suggestions for proof strategies
- Code generation for new features

### What It Includes

- **Claude CLI**: Command-line interface to Claude AI
- **Node.js**: Required runtime for the CLI

### Installation

During setup, answer "yes" when prompted, or install manually:

```bash
# Activate the conda environment
conda activate proofatlas

# Install Node.js via conda
conda install -y nodejs -c conda-forge

# Install Claude CLI locally in the environment
npm install @anthropic-ai/claude-cli

# The CLI will be available when the environment is active
# Set up API key
export ANTHROPIC_API_KEY='your-api-key-here'
# Or run: claude login
```

Note: The setup script automatically configures the PATH so that locally installed npm packages are accessible when the conda environment is active.

### Usage Example

```bash
# Get help with theorem proving
claude "How do I implement a new inference rule in ProofAtlas?"

# Analyze a proof
claude --file proof.json "Explain this proof structure"
```

## Choosing What to Install

### For Theorem Proving Research (Core Only)
If you're researching classical automated theorem proving, inference rules, or proof search strategies, the core installation is sufficient.

### For Machine Learning Research (Core + PyTorch)
If you're researching neural theorem proving, learning proof guidance, or combining symbolic and neural methods, install PyTorch and related packages.

### For Development (Core + Claude)
If you're actively developing new features or need help understanding the codebase, Claude CLI can be a valuable assistant.

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

### With Claude CLI
- **Network**: Internet connection for API calls
- **API Key**: Anthropic API key required

## Troubleshooting

### PyTorch Installation Issues

If CUDA version conflicts occur:
```bash
# Check your CUDA version
nvidia-smi

# Install CPU-only version if no GPU
conda install pytorch cpuonly -c pytorch
```

### Import Errors

If you see `ImportError: No module named torch`:
- The code is trying to use GNN features without PyTorch installed
- Either install PyTorch or use a non-GNN selector (FIFO, Random)

### Claude CLI Issues

If `claude: command not found`:
- Ensure npm/Node.js is installed
- Check that npm global bin is in your PATH
- Try: `export PATH=$PATH:$(npm config get prefix)/bin`