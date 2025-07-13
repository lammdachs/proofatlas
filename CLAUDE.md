# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important: Saturation and Completeness

When working with theorem provers, remember:
- If a problem is unsatisfiable (a theorem) and the prover saturates without finding a proof, this indicates **incompleteness** in the implementation
- A complete theorem prover will NEVER saturate on an unsatisfiable problem - it will either find a proof or run out of resources
- Do NOT claim the implementation is "complete in principle" or "correct" when it saturates on problems known to be theorems

## Environment Setup and Commands

### Initial Setup
```bash
# Create virtual environment and install
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Optional: Download TPTP library
proofatlas-download-tptp
```

### Running Tests
```bash
# Run all tests from python directory
cd python && python -m pytest tests/ -v

# Run specific test category
cd python && python -m pytest tests/core/ -v
cd python && python -m pytest tests/rules/ -v

# Run with coverage
cd python && python -m pytest tests/ --cov=proofatlas
```

### Test Structure

#### Python Tests
Tests are organized to mirror the source structure:
```
python/tests/
├── core/                    # Tests for core.logic, core.state  
├── proofs/                  # Tests for proofs module
├── rules/                   # Tests for rules module
├── loops/                   # Tests for saturation loops
├── data/                    # Tests for data module
├── fileformats/             # Tests for file format parsers
├── navigator/               # Tests for proof navigator
├── selectors/               # Tests for clause selectors
├── integration/             # Integration tests
├── test_data/               # Test fixtures, TPTP files, reference proofs
├── test_serialization.py    # Comprehensive serialization tests
└── test_rust_bindings.py    # Tests for Rust Python bindings
```

#### Rust Tests
Rust tests follow a different pattern with tests distributed across the codebase:
```
rust/
├── src/
│   ├── core/               # Core module with comprehensive tests
│   │   ├── problem_tests.rs     # ArrayProblem tests (Box<[T]> arrays)
│   │   ├── symbol_table_tests.rs # Symbol interning tests
│   │   ├── builder_tests.rs     # ArrayBuilder tests
│   │   └── proof_tests.rs       # Proof tracking tests
│   ├── rules/
│   │   ├── tests.rs        # Inference rule tests
│   │   ├── variable_sharing_tests.rs # Variable handling tests
│   │   └── failure_tests.rs     # Error case tests
│   └── saturation/         # Saturation loop implementation
└── tests/                   # Integration tests
    ├── README.md           # Test organization guide
    └── integration_test.rs # Cross-module tests
```

**Important**: Unit tests are colocated with source in `*_tests.rs` files.

To run Rust tests:
```bash
# Build and run with Python bindings
cd rust && pip install -e .

# Run all tests
cargo test

# Run only unit tests
cargo test --lib

# Run specific module tests
cargo test core
cargo test rules
cargo test saturation

# Run only integration tests  
cargo test --test '*'
```


## Codebase Architecture

### Module Dependency Order
The codebase follows a strict dependency hierarchy. When modifying code, respect this order:
1. `core.logic` (standalone - pure FOL)
2. `proofs.state` depends on `core.logic`
3. `rules` depends on `core`, `proofs.state`
4. `proofs.proof` depends on `core`, `rules`, `proofs.state`
5. `fileformats` depends on `core`
6. `dataformats` depends on `core`
7. `loops` depends on `core`, `rules`, `proofs`
8. `selectors` depends on `core`, `dataformats`
9. `data` depends on all above

### Core Module (`./rust/src/proofatlas/core/`)
The foundation of the system, implementing pure first-order logic:
- **logic.py**: Immutable FOL objects (Variable, Constant, Function, Predicate, Term, Literal, Clause, Problem)
  - Variables and Constants don't take arity parameter in constructor
  - Use `Problem(*clauses)` not `Problem([clauses])`
  - Clause constructor takes `*literals` (unpacked), not a list

### Rules Module (`src/proofatlas/rules/`)
Modular inference rules that operate on proof states:
- **base.py**: Abstract `Rule` class and `RuleApplication` dataclass
- **resolution.py**: Binary resolution rule
- **factoring.py**: Factoring rule
- **subsumption.py**: Subsumption elimination rule

Rules return `RuleApplication` objects containing:
- `rule_name`: Name of the rule
- `parents`: Indices of parent clauses
- `generated_clauses`: New clauses produced
- `deleted_clause_indices`: Clauses to remove
- `metadata`: Additional information

### Proofs Module (`src/proofatlas/proofs/`)
- **state.py**: `ProofState` tracks processed/unprocessed clauses during proof search
- **proof.py**: `Proof` stores sequence of `ProofStep` objects
  - ProofStep contains: state, selected_clause, applied_rules, metadata
  - Proof tracks both steps list and final_state
  - Applied rules are stored as RuleApplication objects
- **serialization.py**: JSON serialization for proof objects including ProofState

### Current Project Structure

```
proofatlas/
├── python/                    # Python implementation
│   ├── proofatlas/
│   │   ├── core/             # FOL logic core (array-based)
│   │   │   ├── array_builder.py      # Array construction utilities
│   │   │   ├── array_from_dict.py    # Dict to array conversion
│   │   │   ├── array_logic.py        # Array-based FOL implementation
│   │   │   ├── logic_compat.py       # Compatibility layer
│   │   │   ├── zero_copy_arrays.py   # Zero-copy array utilities
│   │   │   └── zero_copy_wrapper.py  # Rust array wrapper
│   │   ├── rules/            # Inference rules
│   │   │   └── array_rules.py        # Array-based inference rules
│   │   ├── loops/            # Saturation loops
│   │   │   └── array_loop.py         # Array-based saturation
│   │   └── selectors/        # Clause selection
│   │       └── array_selectors.py    # Array-based selectors
│   ├── tests/                # Test suite
│   │   ├── core/             # Core module tests
│   │   ├── test_array_*.py   # Array implementation tests
│   │   └── test_rust_*.py    # Rust binding tests
│   └── examples/             # Usage examples
│       ├── array_logic_demo.py
│       ├── array_selection_demo.py
│       └── csr_to_coo_conversion.py
│
├── rust/                     # Rust implementation
│   ├── src/
│   │   ├── core/            # Core data structures
│   │   │   ├── mod.rs       # Module exports
│   │   │   ├── problem.rs   # Problem struct (Box<[T]> arrays)
│   │   │   ├── builder.rs   # Builder for array construction
│   │   │   ├── ordering.rs  # KBO term ordering
│   │   │   └── symbol_table.rs # Symbol interning
│   │   ├── rules/           # Inference rules
│   │   │   ├── mod.rs       # Rule exports
│   │   │   ├── common.rs    # Common rule utilities
│   │   │   ├── resolution.rs        # Binary resolution
│   │   │   ├── factoring.rs         # Factoring
│   │   │   ├── superposition.rs     # Superposition calculus
│   │   │   ├── equality_resolution.rs
│   │   │   ├── equality_factoring.rs
│   │   │   └── equality_symmetry.rs # Equality axioms
│   │   ├── saturation/      # Saturation loop
│   │   │   ├── mod.rs       # Loop implementation
│   │   │   ├── loop.rs      # Main saturation algorithm
│   │   │   ├── subsumption.rs # Subsumption checking
│   │   │   ├── unification.rs # Unification algorithm
│   │   │   └── indexing/    # Indexing structures
│   │   ├── parsing/         # TPTP parser
│   │   │   └── tptp_parser.rs
│   │   └── bindings/        # Python bindings
│   │       └── python/      # PyO3 bindings
│   ├── tests/               # Integration tests
│   ├── examples/            # Rust examples
│   └── bin/                 # Binary executables
│       ├── interactive_saturation.rs  # Interactive prover
│       └── [various debug tools]
│
├── docs/                    # Documentation
│   ├── calculus_quick_reference.md   # Inference rules reference
│   ├── array_native_architecture.md  # Array design docs
│   └── ARRAY_IMPLEMENTATION_UPDATE_2024.md
│
└── CLAUDE.md               # This file
```

### Navigator Module (`src/proofatlas/navigator/`)
Interactive terminal-based proof visualization:
- **proof_navigator.py**: Navigate through proof steps with keyboard controls
  - Uses box-drawing characters for clean terminal UI
  - Two-column layout showing PROCESSED and UNPROCESSED clauses
  - Highlights the given clause with an arrow (→)
  - Shows rule applications and metadata
  - Keyboard controls: n/p (next/prev), q (quit), h (help)

### Loops Module (`src/proofatlas/loops/`)
Implements the given clause algorithm:
- **base.py**: Abstract `Loop` class with helper methods (tautology/subsumption checking)
- **basic.py**: `BasicLoop` - complete implementation with:
  - Resolution and factoring inference rules
  - Forward simplification (tautology deletion, subsumption checking)
  - Redundancy filtering (duplicate removal)
  - Clause size limits
  - Full proof tracking with rule applications
  - **Note**: Backward simplification is TODO

### Key Implementation Notes

1. **Import Structure**: Tests run from `python/` directory, so imports use `proofatlas.module.submodule`

2. **Rule Architecture**:
   - All rules inherit from `Rule` abstract base class
   - Rules implement `apply(state: ProofState, clause_indices: List[int]) -> Optional[RuleApplication]`
   - Rules don't modify state - they return RuleApplication objects
   - Loops are responsible for applying RuleApplications to create new states
   - **Important**: Import ProofState from `proofatlas.proofs.state`, not from `proofatlas.proofs` to avoid circular imports

3. **Loop Architecture**:
   - BasicLoop records the state BEFORE processing each given clause
   - Rule applications are filtered to only show those producing non-redundant clauses
   - Parent indices in rule applications show only the processed clause (given clause is implicit)
   - Steps are 0-indexed in both JSON and CLI display

4. **Selector Architecture**: 
   - Base class in `selectors/base.py`
   - GNN selector includes the model implementation directly
   - Selectors have `select()`, `run()`, and `train()` methods

5. **Parser Integration**: TPTP parser implemented in `fileformats/tptp.py`

6. **No Models Directory**: GNN and other models are integrated into their usage points (e.g., GNN in selectors/gnn.py)

7. **Testing Pattern**: Tests mirror source structure under `python/tests/` directory

8. **Proof Inspection**: Use python/scripts/print_proof.py or python/scripts/inspect_proof.py to examine generated proofs

9. **Rust Components**: High-performance array-based implementation in `rust/` directory
   - Uses CSR (Compressed Sparse Row) format with **Box<[T]> arrays** for stable memory
   - Pre-allocated arrays with capacity tracking to prevent overflows
   - Three main modules: `core/` (data structures), `rules/` (inference), `saturation/` (proof search)
   - Python bindings via PyO3 with array-based interface
   - See `rust/README.md` for detailed documentation
   
   **Key Changes (2024)**:
   - Replaced Vec<T> with Box<[T]> for zero-copy potential
   - All inference rules updated to use array indexing (no push/insert)
   - Added PyArrayProblem with freeze mechanism for safety
   - Parser can return pre-allocated arrays with user-specified capacity

### Working with the Rust Array Interface

```python
# Using the array-based parser with pre-allocated capacity
from proofatlas_rust.parser import parse_string_to_array, parse_file_to_array

# Parse with capacity suitable for saturation
problem = parse_string_to_array(
    tptp_content,
    max_nodes=1000000,    # 1M nodes
    max_clauses=100000,   # 100k clauses  
    max_edges=5000000     # 5M edges
)

# Access arrays (currently copies, zero-copy planned)
node_types, symbols, polarities, arities = problem.get_node_arrays()

# Arrays are numpy arrays
print(f"Node types shape: {node_types.shape}")
print(f"Memory owned by array: {node_types.flags['OWNDATA']}")  # True = copy

# Freeze to prevent modifications (required for future zero-copy)
problem.freeze()
```

### Working with the Refactored Codebase

When implementing new features:
1. Check if it belongs in existing modules before creating new ones
2. Follow the established patterns (e.g., metadata dict for extensibility)
3. Maintain the dependency hierarchy
4. Add tests for any new functionality in the appropriate test file
5. For Rust changes: ensure all array operations use indexing, not push/insert

The refactoring prioritized:
- Flatter directory structure
- Clear module boundaries  
- Flexible metadata-based extensibility
- Testability and maintainability
- Zero-copy ready array infrastructure

### Key Bug Fixes (2024)

1. **Equality Canonicalization**: The equality_symmetry rule now uses KBO ordering to prevent generating redundant clauses (e.g., won't generate `e = mult(X,X)` from `mult(X,X) = e`)

2. **Empty Clause Generation**: Fixed superposition rule to allow empty clause generation (previously skipped with warning). Empty clauses represent contradictions and are essential for refutation proofs.

### Interactive Saturation Tool

Located at `rust/src/bin/interactive_saturation.rs`, this tool allows manual clause selection for debugging saturation:
- Shows processed and unprocessed clauses at each step
- Allows manual selection of the given clause
- Useful for understanding why certain proofs are/aren't found
- Run with: `cargo run --bin interactive_saturation`