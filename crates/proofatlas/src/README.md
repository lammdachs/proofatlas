# ProofAtlas Rust Source Code Organization

This directory contains the source code for the ProofAtlas Rust implementation, organized into logical modules.

## Module Structure

```
src/
├── core/               # Core data structures and array representation
│   ├── problem.rs      # ArrayProblem - CSR representation of formulas
│   ├── symbol_table.rs # String interning for symbols
│   ├── builder.rs      # ArrayBuilder for constructing problems
│   ├── proof.rs        # Proof tracking types
│   └── parser_convert.rs # Parse type to array conversion
│
├── rules/              # Inference rules for superposition calculus
│   ├── common.rs       # Shared types (InferenceResult)
│   ├── resolution.rs   # Binary resolution
│   ├── factoring.rs    # Factoring
│   ├── superposition.rs # Superposition (equality)
│   ├── equality_resolution.rs # Equality resolution
│   └── equality_factoring.rs  # Equality factoring
│
├── saturation/         # Saturation loop and algorithms
│   ├── loop.rs         # Main given-clause algorithm
│   ├── literal_selection.rs # Literal selection strategies
│   ├── clause_selection.rs  # Clause selection strategies
│   ├── subsumption.rs  # Subsumption checking
│   └── unification.rs  # Array-based unification
│
├── parsing/            # File format parsers
│   ├── tptp_parser.rs  # TPTP parser implementation
│   ├── tptp.rs         # TPTP high-level API
│   ├── parse_types.rs  # Intermediate parse representations
│   ├── fof.rs          # First-order formula parsing
│   └── prescan.rs      # Fast file prescanning
│
├── python/             # Python bindings via PyO3
│   ├── bindings.rs     # Main Python module definition
│   ├── parser.rs       # Parser Python bindings
│   └── types.rs        # Type conversions
│
├── fileformats/        # File format handlers
│   └── tptp.rs         # TPTP format handler
│
└── lib.rs              # Library root
```

## Module Dependencies

The modules follow a strict dependency hierarchy:

1. **Core** - No dependencies on other modules
2. **Rules** - Depends on Core
3. **Saturation** - Depends on Core and Rules
4. **Parsing** - Depends on Core
5. **Python** - Depends on all modules (for bindings)

## Key Design Decisions

### Array Representation
The core uses a Compressed Sparse Row (CSR) format for representing logical formulas. This provides:
- Excellent cache locality for traversal
- Minimal memory overhead
- Fast iteration over node children
- Efficient parallel processing potential

### String Interning
All symbols (predicate names, function names, constants) are interned in a `SymbolTable`. This allows:
- O(1) symbol comparison
- Reduced memory usage
- Cache-friendly symbol access

### Modular Rules
Each inference rule is implemented as a separate module with a common interface. This allows:
- Easy addition of new rules
- Clear separation of concerns
- Independent testing of each rule
- Potential for parallel rule application

### Literal Selection
The saturation module supports various literal selection strategies to constrain the search space:
- SelectPositive - Select only positive literals  
- SelectFirst - Select first literal
- SelectFirstNegative - Select first negative, or first if none
- SelectAll - No restriction

## Testing Strategy

Each module has comprehensive unit tests in `*_tests.rs` files:
- `core/*_tests.rs` - Test data structures and builders
- `rules/tests.rs` - Test inference rules
- `saturation/*_tests.rs` - Test saturation components
- `parsing/*_tests.rs` - Test parsers

Integration tests in `tests/` verify cross-module functionality.

## Performance Notes

1. **Zero-copy where possible** - The parser avoids unnecessary allocations
2. **Cache-friendly access** - CSR format keeps related data together
3. **Minimal indirection** - Direct array indexing instead of pointer chasing
4. **Pre-allocated buffers** - ArrayBuilder reuses allocations

## Future Enhancements

- Parallel saturation using Rayon
- SIMD operations for unification
- Memory-mapped file support for large problems
- GPU acceleration for subsumption checking