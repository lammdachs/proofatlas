# ProofAtlas Project Structure

This project is organized into separate Python and Rust components:

## Directory Layout

```
proofatlas/
├── python/         # Python implementation
│   ├── src/        # Python source code
│   │   └── proofatlas/
│   │       ├── core/         # Core logic (Term, Literal, Clause, Problem)
│   │       ├── fileformats/  # File parsers (TPTP)
│   │       ├── rules/        # Inference rules
│   │       ├── proofs/       # Proof tracking
│   │       ├── loops/        # Saturation loops
│   │       ├── selectors/    # Clause selection strategies
│   │       ├── data/         # Dataset management
│   │       └── navigator/    # Proof visualization
│   ├── tests/      # Python tests
│   ├── scripts/    # Utility scripts
│   └── setup.py    # Python package setup
│
├── rust/           # Rust acceleration (optional)
│   ├── src/        # Rust source code
│   │   ├── core/           # Core types (mirrors Python)
│   │   ├── parser/         # High-performance TPTP parser
│   │   ├── fileformats/    # File format handlers
│   │   └── python/         # PyO3 Python bindings
│   ├── Cargo.toml  # Rust package config
│   └── README.md   # Rust component documentation
│
├── pyproject.toml  # Combined Python/Rust build config
├── setup.sh        # Environment setup script
├── environment.yml # Conda environment specification
└── README.md       # Main project documentation
```

## Development Workflow

### Python Development
```bash
cd python
python -m pytest tests/  # Run tests
python scripts/parse_tptp.py ALG001-1.p  # Run scripts
```

### Rust Development (Optional)
```bash
cd rust
cargo test              # Run Rust tests
maturin develop         # Build Python bindings
python test_parser.py   # Test integration
```

### Building Everything
```bash
# From project root
maturin develop  # Builds both Python and Rust components
```

## Key Design Principles

1. **Gradual Migration**: Start with Python, optionally accelerate with Rust
2. **Drop-in Compatibility**: Rust components return Python objects
3. **Modular Structure**: Each component can be developed independently
4. **Performance Focus**: Critical paths can be ported to Rust as needed

## Import Paths

From Python code:
```python
# Python components
from proofatlas.core.logic import Problem, Clause, Literal
from proofatlas.fileformats.tptp import TPTPFormat

# Rust components (if built)
import proofatlas_rust
parser = proofatlas_rust.parser.RustTPTPParser()
```