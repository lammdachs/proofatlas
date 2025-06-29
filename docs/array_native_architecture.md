# Array-Native Architecture for ML-Based Theorem Proving

## Table of Contents
1. [Introduction](#introduction)
2. [Motivation](#motivation)
3. [Core Concepts](#core-concepts)
4. [Data Structures](#data-structures)
5. [Algorithm Adaptations](#algorithm-adaptations)
6. [ML Integration](#ml-integration)
7. [Implementation Strategy](#implementation-strategy)
8. [Performance Considerations](#performance-considerations)
9. [Migration Guide](#migration-guide)

## Introduction

This document describes a novel array-native architecture for theorem proving, designed from the ground up to support machine learning-based clause and literal selection. Instead of traditional object-oriented representations of logical formulas, all data is stored in flat arrays suitable for efficient ML processing and zero-copy transfer between Rust and Python.

## Motivation

Traditional theorem provers represent logical formulas as nested object structures:
```rust
struct Literal {
    polarity: bool,
    predicate: Predicate,
}

struct Predicate {
    name: String,
    args: Vec<Term>,
}

enum Term {
    Variable(String),
    Constant(String), 
    Function { name: String, args: Vec<Term> },
}
```

While intuitive, this representation has several drawbacks for ML-based approaches:

1. **Conversion Overhead**: Converting to ML-friendly formats (graphs, matrices) is expensive
2. **Memory Fragmentation**: Nested structures lead to poor cache locality
3. **No Vectorization**: Cannot leverage SIMD/GPU operations
4. **Python Interface**: Complex serialization for cross-language transfer

The array-native architecture addresses these issues by storing everything as flat arrays from the start.

## Core Concepts

### Graph Representation of Logic

First-order logic formulas naturally form graphs:
- **Nodes**: Variables, constants, functions, predicates, literals, clauses
- **Edges**: Argument relationships, literal membership, predicate applications

Example: The clause `P(x, f(a)) ∨ ¬Q(x)` becomes:
```
Clause
├─ Literal[+]
│  └─ Predicate[P]
│     ├─ Variable[x]
│     └─ Function[f]
│        └─ Constant[a]
└─ Literal[-]
   └─ Predicate[Q]
      └─ Variable[x]
```

### Array Storage

Instead of objects, we store:
1. **Node Arrays**: Type, symbol ID, features
2. **Edge Arrays**: Sparse adjacency matrix (CSR format)
3. **Hierarchy Arrays**: Clause→literal, literal→term mappings
4. **Symbol Table**: String↔ID mappings

## Data Structures

### Core Array Types

```rust
/// Global problem representation
pub struct ArrayProblem {
    // Node data
    pub node_types: Vec<NodeType>,      // Enum: Variable, Constant, Function, etc.
    pub node_symbols: Vec<u32>,         // Symbol table indices
    pub node_polarities: Vec<i8>,       // For literals: 1, -1, or 0
    pub node_arities: Vec<u32>,         // Number of arguments
    
    // Edge data (CSR format)
    pub edge_row_offsets: Vec<usize>,   // Start of each node's edges
    pub edge_col_indices: Vec<u32>,     // Target nodes
    pub edge_types: Vec<EdgeType>,      // HasArg, InClause, etc.
    
    // Hierarchy
    pub clause_boundaries: Vec<usize>,   // Node indices where clauses start/end
    pub literal_boundaries: Vec<usize>,  // Node indices where literals start/end
    
    // Symbol table
    pub symbols: SymbolTable,
    
    // Metadata
    pub num_nodes: usize,
    pub num_clauses: usize,
    pub num_literals: usize,
}

/// Node types
#[repr(u8)]
#[derive(Copy, Clone, Debug)]
pub enum NodeType {
    Variable = 0,
    Constant = 1,
    Function = 2,
    Predicate = 3,
    Literal = 4,
    Clause = 5,
}

/// Edge types
#[repr(u8)]
#[derive(Copy, Clone, Debug)]
pub enum EdgeType {
    HasArgument = 0,    // Function/predicate → term
    HasLiteral = 1,     // Clause → literal
    HasPredicate = 2,   // Literal → predicate
    HasPolarity = 3,    // Literal → polarity marker
}

/// Symbol table for string mappings
pub struct SymbolTable {
    symbols: Vec<String>,
    symbol_to_id: HashMap<String, u32>,
    next_id: u32,
}
```

### Index Structures

For efficient access, we maintain index structures:

```rust
/// Maps high-level concepts to array indices
pub struct IndexMap {
    // Clause ID → node index
    clause_nodes: Vec<usize>,
    
    // Literal ID → node index  
    literal_nodes: Vec<usize>,
    
    // Variable occurrence tracking
    var_occurrences: HashMap<u32, Vec<usize>>,  // Symbol ID → node indices
}
```

### View Types for Convenient Access

While data is stored in arrays, we provide zero-cost view types:

```rust
/// Lightweight view into a clause
pub struct ClauseView<'a> {
    problem: &'a ArrayProblem,
    clause_idx: usize,
    start_node: usize,
    end_node: usize,
}

impl<'a> ClauseView<'a> {
    pub fn literals(&self) -> impl Iterator<Item = LiteralView<'a>> {
        // Iterate over literal nodes in this clause's range
    }
    
    pub fn num_literals(&self) -> usize {
        // Count literal nodes
    }
}

/// View into a literal
pub struct LiteralView<'a> {
    problem: &'a ArrayProblem,
    literal_node: usize,
}

impl<'a> LiteralView<'a> {
    pub fn polarity(&self) -> bool {
        self.problem.node_polarities[self.literal_node] > 0
    }
    
    pub fn predicate(&self) -> PredicateView<'a> {
        // Follow edge to predicate node
    }
}
```

## Algorithm Adaptations

### Array-Based Unification

Traditional unification recursively traverses term structures. Array-based unification uses index-based traversal:

```rust
/// Unification result as array of substitutions
pub struct ArraySubstitution {
    var_indices: Vec<usize>,    // Variable node indices
    term_indices: Vec<usize>,   // What they map to
}

/// Unify two terms given by node indices
pub fn unify_nodes(
    problem: &ArrayProblem,
    node1: usize,
    node2: usize,
    subst: &mut ArraySubstitution,
) -> bool {
    let type1 = problem.node_types[node1];
    let type2 = problem.node_types[node2];
    
    match (type1, type2) {
        (NodeType::Variable, _) => {
            // Check if already bound
            if let Some(pos) = subst.var_indices.iter().position(|&v| v == node1) {
                unify_nodes(problem, subst.term_indices[pos], node2, subst)
            } else {
                // Bind variable
                subst.var_indices.push(node1);
                subst.term_indices.push(node2);
                true
            }
        }
        (_, NodeType::Variable) => {
            unify_nodes(problem, node2, node1, subst)
        }
        (NodeType::Constant, NodeType::Constant) => {
            problem.node_symbols[node1] == problem.node_symbols[node2]
        }
        (NodeType::Function, NodeType::Function) => {
            // Check symbol and arity
            if problem.node_symbols[node1] != problem.node_symbols[node2] ||
               problem.node_arities[node1] != problem.node_arities[node2] {
                return false;
            }
            
            // Unify arguments using edge traversal
            let args1 = get_argument_nodes(problem, node1);
            let args2 = get_argument_nodes(problem, node2);
            
            args1.iter().zip(args2.iter()).all(|(&a1, &a2)| {
                unify_nodes(problem, a1, a2, subst)
            })
        }
        _ => false,
    }
}

/// Get argument nodes using CSR edge format
fn get_argument_nodes(problem: &ArrayProblem, node: usize) -> Vec<usize> {
    let start = problem.edge_row_offsets[node];
    let end = problem.edge_row_offsets[node + 1];
    
    problem.edge_col_indices[start..end]
        .iter()
        .enumerate()
        .filter(|(i, _)| problem.edge_types[start + i] == EdgeType::HasArgument)
        .map(|(_, &target)| target as usize)
        .collect()
}
```

### Array-Based Resolution

Resolution creates new clauses by combining parent clauses:

```rust
/// Resolve two clauses on complementary literals
pub fn resolve_clauses(
    problem: &mut ArrayProblem,
    clause1_idx: usize,
    clause2_idx: usize,
    lit1_idx: usize,
    lit2_idx: usize,
) -> Option<usize> {
    // Get literal nodes
    let lit1_node = problem.literal_nodes[lit1_idx];
    let lit2_node = problem.literal_nodes[lit2_idx];
    
    // Check complementary
    if problem.node_polarities[lit1_node] == problem.node_polarities[lit2_node] {
        return None;
    }
    
    // Unify predicates
    let pred1 = get_predicate_node(problem, lit1_node);
    let pred2 = get_predicate_node(problem, lit2_node);
    
    let mut subst = ArraySubstitution::new();
    if !unify_nodes(problem, pred1, pred2, &mut subst) {
        return None;
    }
    
    // Build resolvent by copying parent literals (except resolved ones)
    let new_clause_node = problem.num_nodes;
    problem.node_types.push(NodeType::Clause);
    problem.node_symbols.push(0);
    problem.node_polarities.push(0);
    problem.node_arities.push(0);
    
    // Copy literals from clause1 (except lit1)
    copy_literals_except(problem, clause1_idx, lit1_idx, new_clause_node, &subst);
    
    // Copy literals from clause2 (except lit2)
    copy_literals_except(problem, clause2_idx, lit2_idx, new_clause_node, &subst);
    
    problem.num_nodes = problem.node_types.len();
    problem.num_clauses += 1;
    
    Some(problem.num_clauses - 1)
}
```

### Array-Based Subsumption

Subsumption checking becomes graph matching:

```rust
/// Check if clause1 subsumes clause2
pub fn subsumes(
    problem: &ArrayProblem,
    clause1_idx: usize,
    clause2_idx: usize,
) -> bool {
    let lits1 = get_clause_literals(problem, clause1_idx);
    let lits2 = get_clause_literals(problem, clause2_idx);
    
    // Every literal in clause1 must match some literal in clause2
    lits1.iter().all(|&lit1| {
        lits2.iter().any(|&lit2| {
            literals_match(problem, lit1, lit2)
        })
    })
}

/// Check if two literals match (same polarity, unifiable predicates)
fn literals_match(
    problem: &ArrayProblem,
    lit1_node: usize,
    lit2_node: usize,
) -> bool {
    // Check polarity
    if problem.node_polarities[lit1_node] != problem.node_polarities[lit2_node] {
        return false;
    }
    
    // Try to unify predicates
    let pred1 = get_predicate_node(problem, lit1_node);
    let pred2 = get_predicate_node(problem, lit2_node);
    
    let mut subst = ArraySubstitution::new();
    unify_nodes(problem, pred1, pred2, &mut subst)
}
```

## ML Integration

### Zero-Copy Python Interface

Using the `numpy` crate, we can expose arrays directly to Python:

```rust
use numpy::{PyArray1, PyArray2, IntoPyArray};
use pyo3::prelude::*;

#[pyclass]
pub struct PyArrayProblem {
    inner: ArrayProblem,
}

#[pymethods]
impl PyArrayProblem {
    /// Get node features as NumPy arrays
    fn get_node_arrays(&self, py: Python) -> (
        &PyArray1<u8>,      // node_types
        &PyArray1<u32>,     // node_symbols
        &PyArray1<i8>,      // node_polarities
        &PyArray1<u32>,     // node_arities
    ) {
        (
            self.inner.node_types.as_slice().into_pyarray(py),
            self.inner.node_symbols.as_slice().into_pyarray(py),
            self.inner.node_polarities.as_slice().into_pyarray(py),
            self.inner.node_arities.as_slice().into_pyarray(py),
        )
    }
    
    /// Get edge arrays (CSR format)
    fn get_edge_arrays(&self, py: Python) -> (
        &PyArray1<usize>,   // row_offsets
        &PyArray1<u32>,     // col_indices
        &PyArray1<u8>,      // edge_types
    ) {
        (
            self.inner.edge_row_offsets.as_slice().into_pyarray(py),
            self.inner.edge_col_indices.as_slice().into_pyarray(py),
            self.inner.edge_types.as_slice().into_pyarray(py),
        )
    }
    
    /// Get clause boundaries for indexing
    fn get_clause_info(&self, py: Python) -> (
        &PyArray1<usize>,   // clause_boundaries
        &PyArray1<usize>,   // literal_boundaries
    ) {
        (
            self.inner.clause_boundaries.as_slice().into_pyarray(py),
            self.inner.literal_boundaries.as_slice().into_pyarray(py),
        )
    }
}
```

### Python ML Pipeline

```python
import numpy as np
import scipy.sparse as sp
from proofatlas_rust import PyArrayProblem

class ArrayGraphNN:
    def __init__(self, hidden_dim=64):
        self.hidden_dim = hidden_dim
        # Initialize GNN layers
        
    def score_clauses(self, problem: PyArrayProblem) -> np.ndarray:
        # Get arrays from Rust (zero-copy)
        node_types, node_symbols, polarities, arities = problem.get_node_arrays()
        row_offsets, col_indices, edge_types = problem.get_edge_arrays()
        clause_bounds, literal_bounds = problem.get_clause_info()
        
        # Build sparse adjacency matrix
        num_nodes = len(node_types)
        adjacency = sp.csr_matrix(
            (np.ones_like(col_indices), col_indices, row_offsets),
            shape=(num_nodes, num_nodes)
        )
        
        # Create node features
        node_features = np.column_stack([
            node_types,
            polarities,
            arities,
            # Add more features as needed
        ])
        
        # Run GNN
        embeddings = self.gnn_forward(adjacency, node_features)
        
        # Extract clause embeddings
        clause_embeddings = []
        for i in range(len(clause_bounds) - 1):
            start, end = clause_bounds[i], clause_bounds[i+1]
            # Pool over nodes in clause
            clause_emb = embeddings[start:end].mean(axis=0)
            clause_embeddings.append(clause_emb)
        
        # Score clauses
        return self.score_head(np.array(clause_embeddings))
```

### Literal Selection Integration

```python
class ArrayLiteralSelector:
    def __init__(self, model):
        self.model = model
        
    def select_literals(self, problem: PyArrayProblem, clause_idx: int) -> List[int]:
        # Get arrays
        node_types, _, polarities, _ = problem.get_node_arrays()
        clause_bounds, literal_bounds = problem.get_clause_info()
        
        # Find literals in clause
        clause_start = clause_bounds[clause_idx]
        clause_end = clause_bounds[clause_idx + 1]
        
        literal_indices = []
        for node_idx in range(clause_start, clause_end):
            if node_types[node_idx] == NodeType.Literal:
                literal_indices.append(node_idx)
        
        # Score literals with ML model
        scores = self.model.score_literals(problem, literal_indices)
        
        # Return selected literals
        return np.argsort(scores)[::-1]  # Highest scores first
```

## Implementation Strategy

### Phase 1: Basic Infrastructure
1. Implement core array types and symbol table
2. Create basic node/edge builders
3. Add view types for convenient access
4. Test with simple propositional examples

### Phase 2: First-Order Logic
1. Implement array-based unification
2. Add term structure support
3. Create array-based substitution
4. Test with first-order problems

### Phase 3: Inference Rules
1. Port resolution to arrays
2. Implement factoring
3. Add subsumption checking
4. Integrate with saturation loop

### Phase 4: Python Integration
1. Add numpy bindings
2. Create Python array access
3. Implement example ML models
4. Benchmark against traditional approach

### Phase 5: Optimizations
1. Memory pool allocation
2. Parallel array operations
3. GPU acceleration support
4. Advanced indexing structures

## Performance Considerations

### Memory Layout
- **Cache Locality**: Arrays are contiguous, improving cache performance
- **Vectorization**: Operations on arrays can use SIMD instructions
- **Memory Pools**: Pre-allocate arrays to reduce allocation overhead

### Computational Complexity
- **Unification**: O(n) array traversal vs O(n) tree traversal, but better constants
- **Subsumption**: O(n²) literal matching, but parallelizable
- **Resolution**: O(n) clause construction with array copying

### Benchmarking Strategy
Compare array-native vs traditional on:
1. Parse time
2. Inference rule performance
3. ML feature extraction time
4. End-to-end proof search
5. Memory usage

## Migration Guide

### For Existing Code

1. **Parser Migration**:
   ```rust
   // Old
   fn parse_literal(input: &str) -> Literal { ... }
   
   // New
   fn parse_literal_to_array(
       input: &str,
       problem: &mut ArrayProblem,
   ) -> usize { ... }  // Returns node index
   ```

2. **Rule Migration**:
   ```rust
   // Old
   fn resolve(lit1: &Literal, lit2: &Literal) -> Option<Clause> { ... }
   
   // New  
   fn resolve_array(
       problem: &mut ArrayProblem,
       lit1_idx: usize,
       lit2_idx: usize,
   ) -> Option<usize> { ... }
   ```

3. **Selector Migration**:
   ```python
   # Old
   def select(self, clause: Clause) -> int:
       features = extract_features(clause)
       return self.model.predict(features)
   
   # New
   def select(self, problem: PyArrayProblem, clause_idx: int) -> int:
       # Direct array access, no conversion
       scores = self.model.score_from_arrays(problem, clause_idx)
       return scores.argmax()
   ```

### Compatibility Layer

During migration, maintain both representations:

```rust
pub struct HybridProblem {
    traditional: Problem,
    array_repr: ArrayProblem,
}

impl HybridProblem {
    fn add_clause(&mut self, clause: Clause) {
        let array_idx = self.array_repr.add_clause_from_traditional(&clause);
        self.traditional.clauses.push(clause);
    }
}
```

## Conclusion

The array-native architecture represents a fundamental shift in theorem prover design, optimizing for ML integration from the ground up. While requiring significant implementation effort, it promises substantial performance improvements for ML-based proof search strategies. The key insight is that logical formulas are graphs, and graphs are best represented as sparse matrices for computational efficiency.