# ProofAtlas: Visual Neural Guidance for First-Order Theorem Proving

ProofAtlas is a research framework for experimenting with neural guidance in automated theorem proving, with a strong focus on visualization of proof structures and search strategies. It combines graph neural networks (GNNs) and transformers to learn proof strategies from successful theorem proving attempts, while providing rich visual insights into the proving process.

## Overview

This project provides a flexible platform for:
- **Visual proof exploration**: Interactive visualization of proof structures, search spaces, and neural attention
- **Neural proof guidance**: Learning to guide theorem provers using deep learning with visual feedback
- **Multiple proof formats**: Supporting TPTP and other file formats
- **Real-time visualization**: Watch proof search strategies unfold in real-time

## Key Features

- **Interactive Visualization**:
  - Real-time proof search visualization
  - Graph-based formula structure display
  - Attention heatmaps for neural guidance
  - Proof tree exploration interface
  - Search space topology mapping
- **Parser Support**: Built-in parser for TPTP problem format
- **Graph Representations**: Convert logical formulas and proofs into graph structures
- **Neural Architectures**: 
  - Graph Neural Networks for formula structure understanding
  - Transformers for proof step sequencing
  - Hybrid models combining both approaches
- **Training Infrastructure**: PyTorch Lightning-based training with distributed support
- **Flexible Configuration**: Hydra/OmegaConf-based configuration system

## Project Structure

ProofAtlas is organized into Python and Rust components:

```
proofatlas/
├── python/         # Python implementation
│   ├── src/        # Source code
│   ├── tests/      # Test suite
│   └── scripts/    # Utility scripts
├── rust/           # Rust acceleration (optional)
│   ├── src/        # Rust source code
│   └── README.md   # Rust component docs
└── setup.sh        # Setup script
```

## Installation

### Prerequisites
- Python 3.11 or 3.12
- Rust toolchain (for building the acceleration module)

### Install via pip

ProofAtlas is distributed via pip. We recommend installing in a virtual environment:

```bash
# Create a virtual environment
python -m venv proofatlas-env
source proofatlas-env/bin/activate  # On Windows: proofatlas-env\Scripts\activate

# Install ProofAtlas
pip install proofatlas
```

#### PyTorch Setup

ProofAtlas requires PyTorch. If you already have PyTorch installed in your environment, ProofAtlas will use it. Otherwise, we'll install PyTorch with CPU support by default.

**For GPU support**, install PyTorch first with your CUDA version:
```bash
# Example for CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Example for CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**For CPU-only**, PyTorch will be installed automatically, or you can explicitly install:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Check your CUDA version:
```bash
nvidia-smi  # Shows your NVIDIA driver and CUDA version
```

### Development Installation

For development or to access the latest features:

```bash
# Clone the repository
git clone https://github.com/yourusername/proofatlas.git
cd proofatlas

# Install Rust (if not present)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all optional dependencies
pip install -e ".[dev]"
```


### Downloading TPTP Library

After installation, you can download the TPTP problem library:

```bash
# Download TPTP to default location (.data/problems/tptp)
proofatlas-download-tptp

# Download to custom location
proofatlas-download-tptp --data-dir /path/to/tptp

# Force re-download
proofatlas-download-tptp --force
```

The TPTP library contains thousands of theorem proving problems and is useful for testing and benchmarking.

## Project Structure

```
proofatlas/
├── src/proofatlas/
│   ├── core/             # First-order logic representations
│   ├── rules/            # Modular inference rules (resolution, factoring)
│   ├── proofs/           # Proof state and proof tracking
│   ├── fileformats/      # File format parsers (TPTP)
│   ├── dataformats/      # Data representations for selectors
│   ├── data/             # Dataset management and splitting
│   ├── loops/            # Given clause algorithm implementations
│   ├── selectors/        # Clause selection strategies
│   ├── navigator/        # Terminal-based proof visualization
│   └── utils/            # Utility functions
├── tests/                # Test suite (mirrors src structure)
│   ├── core/
│   ├── rules/
│   ├── proofs/
│   ├── loops/
│   ├── data/
│   ├── fileformats/
│   ├── navigator/
│   └── .data/            # Static test fixtures and example data
├── docs/                 # Documentation
│   ├── saturation_loop_design.md  # BasicLoop design and implementation
│   ├── directory_structure.md     # Data directory organization
│   └── optional_dependencies.md   # Guide to optional components
├── scripts/              # Utility scripts
│   ├── print_proof.py    # Print proofs in readable format
│   └── inspect_proof.py  # Interactive proof navigation
├── .data/                # Data directory (created by setup)
│   ├── problems/         # Theorem proving problems
│   ├── proofs/          # Generated proofs
│   ├── datasets/        # Prepared datasets
│   └── cache/           # Temporary files
├── .logs/               # Log files (created by setup)
└── .selectors/          # ML resources (if PyTorch installed)
```

## Usage

### Quick Start

```python
from proofatlas import *

# Create a simple problem
P = Predicate("P", 0)
Q = Predicate("Q", 0)

# Create clauses: P, P→Q, ¬Q
clause1 = Clause(Literal(P(), True))                    # P
clause2 = Clause(Literal(P(), False), Literal(Q(), True))  # ¬P ∨ Q
clause3 = Clause(Literal(Q(), False))                   # ¬Q

problem = Problem(clause1, clause2, clause3)

# Run saturation to find proof
proof = prove(problem)
print(f"Proof found: {proof.final_state.contains_empty_clause}")
```

### Examples

The `examples/` directory contains several examples demonstrating core functionality:

- **`basic_saturation.py`** - Basic saturation loop demonstration
- **`custom_problem.py`** - Creating problems programmatically  
- **`tptp_parsing.py`** - Parsing and working with TPTP files
- **`selector_comparison.py`** - Comparing different clause selection strategies

Run an example:
```bash
python examples/basic_saturation.py
```

### Data Preparation
```bash
# Select problems from a TPTP dataset
python scripts/data/select_problems.py --tptp-dir /path/to/tptp --output data/problems.json

# Create training dataset from solved problems
python scripts/data/create_dataset.py --problems data/problems.json --output data/dataset.pt
```

### Training Models
```bash
# Train a hybrid GNN-Transformer model
python scripts/train/train_model.py \
    --config configs/hybrid_model.yaml \
    --data data/dataset.pt \
    --name my_experiment

# Train with custom configuration
python scripts/train/train_model.py \
    --config configs/base.yaml \
    model.hidden_dim=512 \
    training.batch_size=32 \
    training.learning_rate=0.001
```

### Solving Problems
```bash
# Use trained model to guide theorem proving
python scripts/solve/solve_problems.py \
    --model checkpoints/best_model.ckpt \
    --problems data/test_problems.json \
    --prover vampire \
    --timeout 300
```

### Running the Saturation Loop
```bash
# Example of using BasicLoop (see tests/loops/test_basic_loop_save_proofs.py for full examples)
from proofatlas.loops.basic import BasicLoop
from proofatlas.proofs import Proof
from proofatlas.proofs.state import ProofState

# Create initial state with clauses
initial_state = ProofState(processed=[], unprocessed=[clause1, clause2, ...])
proof = Proof(initial_state)

# Run saturation steps
loop = BasicLoop(max_clause_size=50, forward_simplify=True)
proof = loop.step(proof, given_clause=0)  # Process first clause
```

### Visualizing Proofs
```bash
# Print a proof in readable format
python scripts/print_proof.py path/to/proof.json

# Print specific step only
python scripts/print_proof.py path/to/proof.json --step 5

# Navigate through a proof interactively
python scripts/inspect_proof.py path/to/proof.json

# Using the built-in navigator
python -m proofatlas.navigator proof.json
```

The proof visualization tools provide:
- Two-column layout showing PROCESSED and UNPROCESSED clauses
- Given clause highlighting with arrow (→)
- Rule application display with parent clause indices
- Clause generation tracking
- Simple keyboard navigation (n/next, p/prev, q/quit, h/help)

## Configuration

The project uses Hydra for configuration management. Key configuration areas:

- **Model Architecture**: GNN layers, transformer heads, embedding dimensions
- **Training**: Learning rate, batch size, optimization strategy
- **Data**: Problem selection criteria, preprocessing options
- **Inference**: Search strategies, guidance parameters

Example configuration override:
```bash
python scripts/train/train_model.py \
    model=transformer \
    model.num_heads=8 \
    model.num_layers=6 \
    training.epochs=100
```

## Extending the Framework

### Adding New Inference Rules
1. Create a new rule class in `src/proofatlas/rules/`
2. Inherit from the `Rule` abstract base class
3. Implement `name` property and `apply` method
4. Return `RuleApplication` objects with generated clauses

Example:
```python
from proofatlas.rules.base import Rule, RuleApplication

class MyRule(Rule):
    @property
    def name(self) -> str:
        return "my_rule"
    
    def apply(self, state, clause_indices):
        # Rule implementation
        return RuleApplication(
            rule_name=self.name,
            parents=clause_indices,
            generated_clauses=new_clauses
        )
```

### Adding New File Format Parsers
1. Create parser in `src/proofatlas/fileformats/`
2. Implement the `FileFormat` interface
3. Register in the format registry

### Implementing New Selectors
1. Add selector to `src/proofatlas/selectors/`
2. Inherit from `Selector` base class
3. Implement `select()` method for clause selection

### Custom Proof Strategies
1. Implement new loop in `src/proofatlas/loops/`
2. Use the modular rule system for inferences
3. Track rule applications in proof steps
4. Integrate with the solving pipeline

## Research Applications

This framework has been designed for research in:
- **Proof guidance learning**: Learning from successful proofs to guide future attempts
- **Formula embeddings**: Learning meaningful representations of logical formulas
- **Proof search strategies**: Combining neural guidance with traditional ATP techniques
- **Transfer learning**: Applying learned strategies across different problem domains

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use ProofAtlas in your research, please cite:
```bibtex
@software{proofatlas2024,
  title = {ProofAtlas: Visual Neural Guidance for First-Order Theorem Proving},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/proofatlas}
}
```

## License

This project is licensed under the BSD 0-Clause License - see [LICENSE](LICENSE) for details.

## Troubleshooting

### CUDA/GPU Errors
If you encounter errors like `libcupti.so.11.8: cannot open shared object file`:
- **Cause**: PyTorch was installed with GPU support but CUDA runtime libraries are missing
- **Solution 1** (Recommended): Reinstall PyTorch with CPU-only support:
  ```bash
  conda activate proofatlas
  conda remove pytorch pyg pytorch-lightning
  conda install -y pytorch cpuonly pyg cpuonly pytorch-lightning -c pytorch -c pyg -c conda-forge
  ```
- **Solution 2**: Install CUDA drivers matching your PyTorch version (see [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads))

### Import Errors
If you get import errors when PyTorch is not installed:
- The codebase is designed to work without PyTorch for core functionality
- GNN-based selectors require PyTorch and will be unavailable without it
- All other features (resolution, factoring, basic selectors) work without PyTorch

### Test Failures
Run tests from the `src` directory:
```bash
cd src
python -m pytest ../tests/ -v
```

## Acknowledgments

- Built on PyTorch and PyTorch Geometric
- Inspired by recent advances in neural theorem proving
- TPTP problem library for standardized benchmarks