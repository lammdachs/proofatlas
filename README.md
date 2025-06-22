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
- **Parser Support**: Built-in parsers for TPTP and Vampire output formats
- **Graph Representations**: Convert logical formulas and proofs into graph structures
- **Neural Architectures**: 
  - Graph Neural Networks for formula structure understanding
  - Transformers for proof step sequencing
  - Hybrid models combining both approaches
- **Training Infrastructure**: PyTorch Lightning-based training with distributed support
- **Flexible Configuration**: Hydra/OmegaConf-based configuration system

## Installation

### Prerequisites
- Conda (Anaconda or Miniconda)
- CUDA-capable GPU (optional, for GNN-based clause selection)

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/proofatlas.git
cd proofatlas

# Run the setup script
./setup.sh
```

The setup script will:
1. Create a conda environment with core dependencies from `environment.yml`
2. Optionally install PyTorch and GNN packages (for neural clause selection)
3. Optionally install Claude CLI (for AI assistance)
4. Set up the project in development mode
5. Create necessary directories and configuration files

### Installation Options

#### Core Installation (Default)
The basic installation includes all theorem proving functionality:
- Saturation-based theorem prover with given clause algorithm
- Resolution, factoring, and subsumption inference rules
- TPTP and Vampire proof format parsers
- Basic clause selection strategies (FIFO, Random)
- Proof visualization and exploration tools

#### Optional: PyTorch and Graph Neural Networks
Required for advanced clause selection strategies:
- GNN-based clause selection
- Learned proof guidance
- Neural premise selection

To install: Answer "yes" when prompted during setup, or run:
```bash
conda activate proofatlas
conda install -y pytorch pytorch-cuda=12.1 pyg pytorch-lightning -c pytorch -c nvidia -c pyg
```

#### Optional: Claude CLI
For interactive AI assistance with theorem proving:
- Code generation and explanation
- Proof strategy suggestions
- Interactive debugging help

To install: Answer "yes" when prompted during setup, or run:
```bash
conda activate proofatlas
conda install -y nodejs -c conda-forge
npm install @anthropic-ai/claude-cli
export ANTHROPIC_API_KEY='your-api-key-here'
```

### Manual Setup
If you prefer manual installation:
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate proofatlas

# Install the package
pip install -e .
```

## Project Structure

```
proofatlas/
├── src/proofatlas/
│   ├── core/             # First-order logic representations
│   ├── rules/            # Modular inference rules (resolution, factoring)
│   ├── proofs/           # Proof state and proof tracking
│   ├── fileformats/      # File format parsers (TPTP, Vampire)
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
│   ├── loops/            # Tests for saturation loops
│   ├── data/
│   ├── fileformats/
│   ├── navigator/
│   └── test_data/        # Test data and example proofs
├── docs/                 # Documentation
│   └── saturation_loop_design.md  # BasicLoop design and implementation
├── scripts/              # Utility scripts
│   ├── print_proof.py    # Print proofs in readable format
│   └── inspect_proof.py  # Interactive proof navigation
└── configs/              # Configuration files
```

## Usage

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

If you use Foreduce in your research, please cite:
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

## Acknowledgments

- Built on PyTorch and PyTorch Geometric
- Inspired by recent advances in neural theorem proving
- TPTP problem library for standardized benchmarks