# Box<[T]> Array Implementation (2024)

## Overview

This document describes the major architectural change made to the Rust implementation in 2024: replacing `Vec<T>` with `Box<[T]>` arrays for zero-copy Python interoperability.

## Motivation

The original implementation used `Vec<T>` for dynamic arrays, which could reallocate and move in memory during growth. This prevented safe zero-copy access from Python. The new implementation uses pre-allocated `Box<[T]>` arrays with stable memory addresses.

## Key Changes

### 1. Data Structure Update

**Before:**
```rust
pub struct ArrayProblem {
    pub node_types: Vec<NodeType>,
    pub node_symbols: Vec<u32>,
    // ... other Vec fields
}
```

**After:**
```rust
pub struct ArrayProblem {
    pub node_types: Box<[u8]>,      // NodeType as u8
    pub node_symbols: Box<[u32]>,
    // ... other Box<[T]> fields
    
    // Capacity tracking
    pub max_nodes: usize,
    pub max_clauses: usize,
    pub max_edges: usize,
}
```

### 2. Pre-allocation

Arrays are now pre-allocated with user-specified capacity:

```rust
impl ArrayProblem {
    pub fn with_capacity(max_nodes: usize, max_clauses: usize, max_edges: usize) -> Self {
        ArrayProblem {
            node_types: vec![0; max_nodes].into_boxed_slice(),
            // ... other pre-allocated arrays
        }
    }
}
```

### 3. Array Operations

All `push()` and `insert()` operations were replaced with array indexing:

**Before:**
```rust
problem.node_types.push(NodeType::Clause as u8);
problem.edge_col_indices.insert(pos, target);
```

**After:**
```rust
// Check capacity first
if problem.num_nodes >= problem.max_nodes {
    return Err(CapacityError { ... });
}

// Use array indexing
problem.node_types[problem.num_nodes] = NodeType::Clause as u8;
problem.edge_col_indices[problem.num_edges] = target;
```

### 4. Python Interface

New array-based parser functions with capacity control:

```python
# Parse with pre-allocated capacity
problem = parse_string_to_array(
    tptp_content,
    max_nodes=1000000,
    max_clauses=100000,
    max_edges=5000000
)

# Access arrays
node_types, symbols, polarities, arities = problem.get_node_arrays()

# Freeze to prevent modifications
problem.freeze()
```

## Updated Files

### Core Changes:
- `src/core/problem.rs` - Box<[T]> arrays with capacity
- `src/core/builder.rs` - Array indexing with bounds checks
- `src/core/parser_convert.rs` - Result types for capacity errors

### Rule Updates:
- `src/rules/common.rs` - All copy functions use array indexing
- `src/rules/resolution.rs` - build_resolvent uses arrays
- `src/rules/factoring.rs` - build_factored uses arrays
- `src/rules/equality_factoring.rs` - Full array conversion
- `src/rules/equality_resolution.rs` - Array-based operations
- `src/rules/superposition.rs` - Complete array update

### Python Bindings:
- `src/bindings/python/array_bindings.rs` - PyArrayProblem with freeze
- `src/bindings/python/parser.rs` - Array conversion for PyO3

## Removed Files

During cleanup, the following superfluous files were removed:
- `src/bindings/python/zero_copy_array.rs` - Earlier Vec-based attempt
- `src/bindings/python/zero_copy_bindings.rs` - Arc<RwLock> approach
- `src/rules/superposition_debug.rs` - Debug version
- `src/test_basic_parser.rs` - Misplaced test file
- `src/saturation/test_superposition.rs` - Test in wrong location

## Benefits

1. **Stable Memory**: Arrays never move after allocation
2. **Zero-Copy Ready**: Foundation for numpy views without copying
3. **Type Safety**: Capacity checks prevent buffer overflows
4. **Performance**: Pre-allocation reduces allocation overhead
5. **Python Integration**: Direct array access for ML/analysis

## Future Work

- Implement true zero-copy using numpy's unsafe APIs
- Add PyBuffer protocol support
- Benchmark performance improvements
- Consider memory mapping for very large problems