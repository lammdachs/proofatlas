# ProofAtlas Rust Components

High-performance Rust implementations of ProofAtlas components with Python bindings.

## Project Structure

This project is designed to support gradual migration of ProofAtlas components from Python to Rust:

```
src/
├── core/           # Core logic types (Term, Literal, Clause, Problem)
├── parser/         # TPTP parser implementation
├── fileformats/    # File format handlers
├── python/         # PyO3 Python bindings
└── lib.rs          # Main library entry point

Future modules:
├── rules/          # Inference rules (resolution, factoring, etc.)
├── proofs/         # Proof state and tracking
└── loops/          # Saturation loop implementations
```

## Building

### Prerequisites

1. Install Rust: https://rustup.rs/
2. Install Maturin: `pip install maturin`

### Development Build

```bash
cd rust_tptp_parser
maturin develop
```

### Release Build

```bash
maturin build --release
```

## Usage

The Rust parser returns Python `Problem` objects that are fully compatible with the existing Python codebase:

```python
import proofatlas_rust

# Create parser with include path
parser = proofatlas_rust.parser.RustTPTPParser(include_path="/path/to/tptp/")

# Parse file - returns a proofatlas.core.logic.Problem object
problem = parser.parse_file("ALG001-1.p")
print(f"Parsed {len(problem.clauses)} clauses")

# Or use the functional API
problem = proofatlas_rust.parser.parse_file("ALG001-1.p", include_path="/path/to/tptp/")

# Pre-scan for performance (avoid parsing large files)
literal_count, is_exact = proofatlas_rust.parser.prescan_file("ALG001-1.p")
if literal_count <= 1000:  # Only parse if reasonable size
    problem = parser.parse_file("ALG001-1.p")

# For JSON serialization, use parse_file_to_dict
problem_dict = parser.parse_file_to_dict("ALG001-1.p")
```

## Performance

The Rust parser offers significant performance improvements:
- 10-100x faster parsing for large files
- Efficient pre-scanning to avoid parsing oversized problems
- Zero-copy parsing where possible
- Parallel processing support (future)

## Integration with Existing Code

The parser is designed as a drop-in replacement:

```python
# Old Python code:
from proofatlas.fileformats.tptp import TPTPFormat
tptp = TPTPFormat()
problem = tptp.parse_file(Path("example.p"))

# New Rust-accelerated code:
import proofatlas_rust
parser = proofatlas_rust.parser.RustTPTPParser()
problem = parser.parse_file("example.p")
```

## Future Components

This structure supports gradually porting more components:

1. **Rules Module**: Port inference rules for faster proof search
2. **Saturation Loops**: Parallel saturation with Rayon
3. **Proof Compression**: Efficient proof storage and manipulation
4. **Indexing**: Fast subsumption and unification indexes

Each component can be developed independently and integrated via PyO3.