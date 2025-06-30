# ProofAtlas Rust Implementation

This directory contains the high-performance Rust implementation of the ProofAtlas theorem prover, featuring an efficient array-based representation for first-order logic formulas and modern superposition calculus inference rules.

## Architecture Overview

The Rust implementation uses a Compressed Sparse Row (CSR) array representation for logical formulas, providing excellent cache locality and performance. The architecture is divided into three main components:

### Core Module (`src/core/`)
Foundation data structures and types:
- **`problem.rs`** - `ArrayProblem` struct using CSR representation for formulas
- **`symbol_table.rs`** - String interning for efficient symbol storage
- **`builder.rs`** - `ArrayBuilder` for constructing array representations
- **`proof.rs`** - Proof tracking types (`ProofStep`, `Proof`, `InferenceRule`)
- **`parser_convert.rs`** - Conversion from parse types to array representation

### Rules Module (`src/rules/`)
Modular inference rules for superposition calculus:
- **`common.rs`** - Shared types and utilities (`InferenceResult`)
- **`resolution.rs`** - Binary resolution rule
- **`factoring.rs`** - Factoring rule  
- **`superposition.rs`** - Superposition rule (equality handling)
- **`equality_resolution.rs`** - Equality resolution rule
- **`equality_factoring.rs`** - Equality factoring rule

### Saturation Module (`src/saturation/`)
Given-clause saturation loop and supporting algorithms:
- **`loop.rs`** - Main saturation algorithm implementation
- **`literal_selection.rs`** - Strategies for constraining inference
- **`clause_selection.rs`** - Clause selection for given-clause algorithm
- **`subsumption.rs`** - Subsumption checking and indexing
- **`unification.rs`** - Fast array-based unification

## Array Representation

The core uses a CSR (Compressed Sparse Row) format with **pre-allocated Box<[T]> arrays** for zero-copy Python interoperability:

```rust
pub struct ArrayProblem {
    // Node data - stored as primitives for zero-copy Python interface
    pub node_types: Box<[u8]>,         // NodeType as u8 (0=Variable, 1=Constant, etc.)
    pub node_symbols: Box<[u32]>,      // Symbol table indices
    pub node_polarities: Box<[i8]>,    // For literals: 1 (positive), -1 (negative), 0 (n/a)
    pub node_arities: Box<[u32]>,      // Number of arguments/children
    pub node_selected: Box<[bool]>,    // For literal selection
    
    // Edge data (CSR format)
    pub edge_row_offsets: Box<[usize]>, // Start index for each node's edges
    pub edge_col_indices: Box<[u32]>,   // Target node indices
    
    // Hierarchical structure
    pub clause_boundaries: Box<[usize]>, // Start/end indices for clauses
    pub clause_types: Box<[u8]>,        // ClauseType as u8
    pub literal_boundaries: Box<[usize]>, // Start/end indices for literals
    
    // Symbol table (remains as SymbolTable for flexibility)
    pub symbols: SymbolTable,
    
    // Metadata - tracks actual usage within pre-allocated arrays
    pub num_nodes: usize,
    pub num_clauses: usize,
    pub num_literals: usize,
    pub num_edges: usize,
    
    // Capacity tracking
    pub max_nodes: usize,
    pub max_clauses: usize,
    pub max_edges: usize,
}
```

### Key Design Decisions:

1. **Box<[T]> instead of Vec<T>**: Provides stable memory addresses, enabling future zero-copy access from Python
2. **Pre-allocation**: Arrays are allocated upfront with capacity for saturation
3. **Primitive types**: Enums stored as u8/u32 for numpy compatibility
4. **Capacity tracking**: Prevents buffer overflows with CapacityError

## Building and Testing

### Prerequisites
- Rust 1.70 or later
- Cargo

### Build
```bash
cargo build --release
```

### Run Tests
```bash
# Run all tests
cargo test

# Run only unit tests
cargo test --lib

# Run only integration tests  
cargo test --test '*'

# Run specific module tests
cargo test core
cargo test rules
cargo test saturation
```

### Test Organization
- Unit tests are located within each module in `*_tests.rs` files
- Integration tests are in the `tests/` directory
- See `tests/README.md` for integration test details

## Usage

### As a Library
```rust
use proofatlas_rust::core::{ArrayProblem, ArrayBuilder};
use proofatlas_rust::saturation::{saturate, SaturationConfig};
use proofatlas_rust::parsing::tptp::TPTPFormat;

// Parse a TPTP file
let parser = TPTPFormat::new();
let problem = parser.parse_file(Path::new("problem.p"))?;

// Run saturation
let config = SaturationConfig::default();
let result = saturate(problem, config);

match result {
    SaturationResult::Proof(proof) => println!("Found proof!"),
    SaturationResult::Saturated => println!("Saturated without proof"),
    SaturationResult::ResourceLimit => println!("Resource limit exceeded"),
}
```

### Python Bindings
The Rust implementation provides Python bindings via PyO3 with array-based interface:

```python
from proofatlas_rust.parser import parse_string_to_array
from proofatlas_rust.array_repr import ArrayProblem

# Parse TPTP to pre-allocated array problem
problem = parse_string_to_array(
    tptp_content,
    max_nodes=1000000,    # Pre-allocate for 1M nodes
    max_clauses=100000,   # Pre-allocate for 100k clauses
    max_edges=5000000     # Pre-allocate for 5M edges
)

# Access arrays (currently copies, zero-copy planned)
node_types, symbols, polarities, arities = problem.get_node_arrays()

# Freeze to prevent modifications
problem.freeze()

# Run saturation (only on unfrozen problems)
found_proof, num_clauses, num_steps = problem.saturate(max_clauses=10000)
```

## Performance Considerations

1. **Array Representation**: The CSR format provides excellent cache locality for traversing formula structures
2. **String Interning**: All symbols are interned in a `SymbolTable` for fast comparison
3. **Literal Selection**: Constrains the search space by limiting which literals participate in inference
4. **Subsumption Checking**: Eliminates redundant clauses (currently uses linear search, discrimination trees planned)

## Future Improvements

- [ ] Implement term ordering (KBO or LPO) for orienting equations
- [ ] Add discrimination trees for efficient subsumption checking
- [ ] Implement backward simplification in the saturation loop
- [ ] Add support for AC (associative-commutative) theories
- [ ] Optimize unification with occurs check

## Contributing

When adding new features:
1. Follow the existing module structure (core/rules/saturation)
2. Add comprehensive unit tests in `*_tests.rs` files
3. Update documentation as needed
4. Run `cargo fmt` and `cargo clippy` before submitting

## License

This is part of the ProofAtlas project. See the main project LICENSE file for details.