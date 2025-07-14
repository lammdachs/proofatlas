# ProofAtlas Installation Guide

## Quick Start

From the top-level directory:

```bash
# Install in development mode
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"

# Or with example dependencies
pip install -e ".[examples]"
```

## Prerequisites

- Python 3.7 or later
- Rust toolchain (install from https://rustup.rs/)
- C compiler (for Python extensions)

## Build from Source

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/proofatlas.git
   cd proofatlas
   ```

2. Install build dependencies:
   ```bash
   pip install setuptools wheel setuptools-rust
   ```

3. Install the package:
   ```bash
   pip install -e .
   ```

## Verify Installation

Run the test script:
```bash
python test_install.py
```

Or test manually:
```python
from proofatlas import ProofState
state = ProofState()
print("ProofAtlas is working!")
```

## Development Setup

For development, install with extra dependencies:
```bash
pip install -e ".[dev]"
```

This includes:
- pytest (testing)
- black (code formatting)
- ruff (linting)
- mypy (type checking)

## Common Issues

1. **ImportError: No module named 'proofatlas.proofatlas'**
   - The Rust extension hasn't been built. Run: `python setup.py build_ext --inplace`

2. **Rust not found**
   - Install Rust from https://rustup.rs/

3. **setuptools-rust not found**
   - Install it: `pip install setuptools-rust`

## Project Structure

- `proofatlas/` - Main Python package (in `python/proofatlas/`)
- `rust/` - Rust implementation with Python bindings
- `tests/` - Test suite (in `python/tests/`)
- `examples/` - Example scripts (in `python/examples/`)

## Using Make

Common development tasks:
```bash
make install      # Install in development mode
make test         # Run tests
make format       # Format code
make lint         # Run linters
make clean        # Clean build artifacts
```