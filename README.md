# ProofAtlas: Visual Neural Guidance for First-Order Theorem Proving

ProofAtlas is a research framework for experimenting with neural guidance in automated theorem proving, with a strong focus on visualization of proof structures and search strategies. It combines graph neural networks (GNNs) and transformers to learn proof strategies from successful theorem proving attempts, while providing rich visual insights into the proving process.

## Overview

This project provides a flexible platform for:
- **Visual proof exploration**: Interactive visualization of proof structures, search spaces, and neural attention
- **Neural proof guidance**: Learning to guide theorem provers using deep learning with visual feedback
- **Hybrid architectures**: Combining GNNs for structural reasoning with transformers for sequential reasoning
- **Multiple proof formats**: Supporting TPTP, Vampire, and other theorem prover formats
- **Real-time visualization**: Watch proof search strategies unfold in real-time
- **Experiment tracking**: Integration with Weights & Biases for comprehensive experiment management

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
- CUDA-capable GPU (recommended)

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/proofatlas.git
cd proofatlas

# Run the setup script
./setup.sh
```

The setup script will:
1. Create a conda environment with all dependencies
2. Install PyTorch with CUDA support
3. Set up the project in development mode
4. Create necessary directories and configuration files

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
│   ├── rules/            # Modular inference rules
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
│   ├── data/
│   ├── fileformats/
│   └── navigator/
├── docs/                 # Documentation
├── scripts/              # Utility scripts
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

### Visualizing Proofs
```bash
# Navigate through a proof interactively
python -m proofatlas.navigator proof.json

# Navigate with problem context
python -m proofatlas.navigator proof.json --problem problem.json
```

The proof navigator provides:
- Two-column layout showing PROCESSED and UNPROCESSED clauses
- Given clause highlighting with arrow (→)
- Rule application display
- Simple keyboard navigation (n/next, p/prev, q/quit)

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

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Built on PyTorch and PyTorch Geometric
- Inspired by recent advances in neural theorem proving
- TPTP problem library for standardized benchmarks