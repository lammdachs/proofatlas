# Graph Representation for Logical Clauses

## Overview

This document describes how we represent first-order logic clauses as graphs suitable for Graph Neural Networks (GNNs). The representation uses PyTorch sparse tensors for efficient storage and computation.

**Goal**: Enable neural networks to learn patterns in logical formulas by treating clauses as graphs where:
- Structure encodes syntactic relationships (argument positions, term nesting)
- Features encode semantic properties (symbol identity, polarity, types)
- Sparse tensors minimize memory for large clause sets

## Why Graphs for Clauses?

Traditional representations (flat vectors, sequences) lose structural information:
- `P(f(x), g(y))` vs `P(g(y), f(x))` → same bag-of-symbols, different structure
- Nested terms like `f(g(h(a)))` have depth that matters
- Argument order matters: `mult(x,y)` ≠ `mult(y,x)` in general

Graph representation preserves:
- Term nesting (parent-child relationships)
- Argument positions (edge labels)
- Literal relationships (disjunction order, polarities)

---

## Graph Structure

### Design Principle

A clause is a **directed graph** with typed nodes and homogeneous edges:

```
Clause: P(x, f(a)) | ~Q(g(x), b)

Graph Structure:
                  [Clause Root]
                   /          \
            [Literal 0]     [Literal 1]
             (pos)            (neg)
               |                |
           [Pred: P]        [Pred: Q]
             /    \            /    \
        [Var:x] [Func:f]  [Func:g] [Const:b]
                   |          |
               [Const:a]  [Var:x]
```

### Node Types

We distinguish 6 node types:

| Type ID | Name       | Description                          | Example       |
|---------|------------|--------------------------------------|---------------|
| 0       | CLAUSE     | Root node representing whole clause  | (implicit)    |
| 1       | LITERAL    | A literal (atom with polarity)       | P(...), ~Q(...) |
| 2       | PREDICATE  | Predicate symbol                     | P, Q, =       |
| 3       | FUNCTION   | Function symbol                      | f, g, mult    |
| 4       | VARIABLE   | Variable term                        | X, Y, Z       |
| 5       | CONSTANT   | Constant term                        | a, b, e       |

**Rationale**:
- CLAUSE node allows representing empty clauses (⊥) and clause-level features
- LITERAL separates polarity from predicate (same predicate can be pos/neg)
- Separate VARIABLE/CONSTANT allows GNN to learn variable-specific patterns
- PREDICATE/FUNCTION separation respects FOL distinction

### Edge Structure (Homogeneous)

**Design Choice**: We use **homogeneous edges** - all edges represent the same type of relationship: "structural containment/connection".

Edges are **directed** and represent parent-child relationships in the syntax tree:
- CLAUSE → LITERAL (clause contains literal)
- LITERAL → PREDICATE (literal has predicate)
- PREDICATE → TERM (predicate has argument)
- FUNCTION → TERM (function has argument)

**No edge types** - the GNN learns to distinguish different relationships from:
1. Node types (what kind of nodes are connected)
2. Graph topology (position in the tree)
3. Node features (arity, depth, etc.)

**Rationale for homogeneous edges**:
- **Simpler implementation**: Single adjacency matrix, no edge type bookkeeping
- **Fewer hyperparameters**: No need to tune edge-type-specific weights
- **Better generalization**: GNN learns relationships from structure, not hard-coded types
- **Sufficient expressiveness**: Node types + topology encode most information
- **Faster computation**: Single sparse matrix multiply per layer (vs one per edge type)

---

## PyTorch Sparse Tensor Representation

### Why Sparse?

Clauses are sparse graphs:
- Typical clause: 10-50 nodes, 15-100 edges
- Dense adjacency matrix: 50×50 = 2,500 entries (95%+ zeros)
- Sparse format: Store only ~100 non-zero entries

**Memory savings**: 25× for typical clauses, 100×+ for large clauses

### Sparse Tensor Formats

PyTorch supports two sparse formats:

#### 1. COO (Coordinate) Format - Best for Construction

```python
import torch

# Example: 4 nodes, 3 edges: 0→1, 1→2, 0→3
indices = torch.tensor([
    [0, 1, 0],  # source nodes
    [1, 2, 3]   # target nodes
])
values = torch.ones(3)  # All edges have weight 1.0
adjacency = torch.sparse_coo_tensor(indices, values, size=(4, 4))

# That's it! No edge types needed.
```

**Use cases**:
- Building graphs from parsed clauses
- Incremental graph construction
- Converting from other formats

**Pros**: Easy to construct, efficient for small graphs
**Cons**: Slow for matrix operations (multiply, indexing)

#### 2. CSR (Compressed Sparse Row) Format - Best for Computation

```python
# Convert COO → CSR for efficient operations
adjacency_csr = adjacency.to_sparse_csr()

# CSR stores:
# - crow_indices: [0, 2, 3, 3, 4] (row pointer array)
# - col_indices: [1, 3, 2, ...] (column indices)
# - values: [1, 1, 1, ...] (edge weights)

# Fast row slicing (get all neighbors of node i)
neighbors = adjacency_csr[i]

# Fast matrix multiplication (GNN message passing)
messages = adjacency_csr @ node_features
```

**Use cases**:
- GNN message passing
- Fast neighbor lookups
- Graph traversal

**Pros**: Fast matrix ops, cache-friendly
**Cons**: Harder to modify, needs conversion from COO

### Recommended Workflow

```python
# 1. Parse clause and build graph in COO format
indices, values, edge_types = parse_clause_to_coo(clause_str)
adjacency_coo = torch.sparse_coo_tensor(indices, values, size=(n, n))

# 2. Convert to CSR for GNN processing
adjacency_csr = adjacency_coo.to_sparse_csr()

# 3. Use CSR in GNN forward pass
h = gnn(node_features, adjacency_csr)
```

---

## Node Features

Each node has a feature vector. Design balances:
- **Expressiveness**: Enough info for learning
- **Generalization**: Not memorizing specific symbols
- **Efficiency**: Small vectors for speed

### Feature Design (v1 - Simple)

**Feature vector size**: 16 dimensions

```python
# Node feature vector [16 dims]
features = [
    # Type encoding (one-hot, 6 dims)
    1.0 if type == CLAUSE else 0.0,
    1.0 if type == LITERAL else 0.0,
    1.0 if type == PREDICATE else 0.0,
    1.0 if type == FUNCTION else 0.0,
    1.0 if type == VARIABLE else 0.0,
    1.0 if type == CONSTANT else 0.0,

    # Structural properties (4 dims)
    arity,           # Number of arguments (0 for terms)
    depth,           # Distance from root
    num_children,    # Outgoing edges
    is_equality,     # 1.0 if predicate is '='

    # Symbol encoding (6 dims)
    symbol_hash_1,   # hash(symbol_name) % 1000 / 1000.0
    symbol_hash_2,   # hash(symbol_name + "2") % 1000 / 1000.0
    symbol_freq,     # Frequency in training set (learned embedding)
    polarity,        # 1.0 for positive literals, 0.0 for negative
    is_variable,     # Redundant with type, but helps gradient flow
    is_unit_clause,  # 1.0 if clause has only one literal
]
```

**Rationale**:
- One-hot type encoding: Explicit type info, easy for GNN to distinguish
- Arity: Critical for understanding term structure
- Depth: Helps GNN learn depth-dependent patterns (deep terms often less important)
- Symbol hashing: Allows some symbol-specific learning without huge vocabulary
- Frequency: Common symbols (like 'e' for identity) get learned embeddings
- Polarity: Essential for resolution (need complementary literals)

### Feature Design (v2 - With Learned Embeddings)

For better symbol handling, use learned embeddings:

```python
# Symbol vocabulary
vocab = {
    'mult': 1,
    'inv': 2,
    'e': 3,
    ...
}

# Embedding layer
symbol_embedding = nn.Embedding(vocab_size, embedding_dim=8)

# Node feature for predicate 'mult'
features = torch.cat([
    type_onehot,              # [6 dims]
    structural_features,      # [4 dims]
    symbol_embedding[vocab['mult']],  # [8 dims]
    polarity_etc              # [2 dims]
])  # Total: 20 dims
```

**Trade-off**: Better performance, but requires vocabulary management and can overfit to training symbols.

---

## Detailed Examples

### Example 1: Simple Clause

**Clause**: `P(x) | ~Q(a)`

**Graph Structure**:
```
[0] CLAUSE
 ├─[1] LITERAL (polarity=1.0)
 │   └─[2] PREDICATE "P" (arity=1)
 │       └─[3] VARIABLE "x"
 └─[4] LITERAL (polarity=0.0)
     └─[5] PREDICATE "Q" (arity=1)
         └─[6] CONSTANT "a"
```

**Adjacency Matrix (COO format)**:
```python
indices = torch.tensor([
    [0, 0, 1, 2, 4, 5],  # source
    [1, 4, 2, 3, 5, 6]   # target
])
values = torch.ones(6)  # All edges have weight 1.0
adjacency = torch.sparse_coo_tensor(indices, values, size=(7, 7))

# Edges: 0→1, 0→4 (clause to literals)
#        1→2, 4→5 (literals to predicates)
#        2→3, 5→6 (predicates to terms)
```

**Node Features** (simplified):
```python
features = torch.tensor([
    [1, 0, 0, 0, 0, 0, 0, 0, ...],  # [0] CLAUSE
    [0, 1, 0, 0, 0, 0, 1, 1, ...],  # [1] LITERAL (positive)
    [0, 0, 1, 0, 0, 0, 1, 2, ...],  # [2] PREDICATE "P"
    [0, 0, 0, 0, 1, 0, 0, 3, ...],  # [3] VARIABLE "x"
    [0, 1, 0, 0, 0, 0, 0, 1, ...],  # [4] LITERAL (negative)
    [0, 0, 1, 0, 0, 0, 1, 2, ...],  # [5] PREDICATE "Q"
    [0, 0, 0, 0, 0, 1, 0, 3, ...],  # [6] CONSTANT "a"
])
```

### Example 2: Nested Functions

**Clause**: `mult(mult(x, y), z) = mult(x, mult(y, z))`

This is an equality literal, so structure is:

```
[0] CLAUSE
 └─[1] LITERAL (equality)
     └─[2] PREDICATE "="
         ├─[3] FUNCTION "mult" (LHS)
         │   ├─[4] FUNCTION "mult"
         │   │   ├─[5] VAR "x"
         │   │   └─[6] VAR "y"
         │   └─[7] VAR "z"
         └─[8] FUNCTION "mult" (RHS)
             ├─[9] VAR "x"
             └─[10] FUNCTION "mult"
                 ├─[11] VAR "y"
                 └─[12] VAR "z"
```

**Key observations**:
- Deep nesting (depth 4 for innermost variables)
- Shared variable names (x, y, z appear twice)
- Equality gets special edge types (EQUALITY_LHS, EQUALITY_RHS)

### Example 3: Unit Clause

**Clause**: `~P(a, b)`

**Graph Structure**:
```
[0] CLAUSE
 └─[1] LITERAL (polarity=0.0, is_unit=1.0)
     └─[2] PREDICATE "P" (arity=2)
         ├─[3] CONSTANT "a"
         └─[4] CONSTANT "b"
```

**Feature highlights**:
- Clause node has `is_unit_clause=1.0`
- Literal node has `polarity=0.0` (negative)
- Predicate has `arity=2`

---

## Batching Multiple Clauses

GNNs need to process multiple clauses in parallel. PyTorch Geometric style batching:

### Approach 1: Batch as Disconnected Graph

Concatenate multiple clause graphs into one large graph:

```python
# Clause 1: 10 nodes, 15 edges
# Clause 2: 8 nodes, 12 edges
# Batch: 18 nodes, 27 edges

# Adjacency matrix for batch
indices_batch = torch.cat([
    indices_clause1,
    indices_clause2 + 10  # Offset by nodes in clause 1
], dim=1)

features_batch = torch.cat([
    features_clause1,
    features_clause2
], dim=0)

# Track which nodes belong to which clause
batch_indices = torch.tensor([
    0, 0, ..., 0,  # 10 nodes from clause 1
    1, 1, ..., 1   # 8 nodes from clause 2
])
```

**Global pooling** to get clause-level embeddings:

```python
# After GNN, pool node embeddings to clause level
clause_embeddings = scatter_mean(
    node_embeddings,
    batch_indices,
    dim=0
)  # Shape: (batch_size, hidden_dim)
```

### Approach 2: Padding to Fixed Size

Alternative for small clauses:

```python
max_nodes = 50

# Pad adjacency and features
adjacency_padded = torch.sparse_coo_tensor(
    indices, values, size=(batch_size, max_nodes, max_nodes)
)
features_padded = torch.zeros(batch_size, max_nodes, feature_dim)
features_padded[:, :actual_nodes] = features

# Use mask to ignore padding in pooling
mask = torch.zeros(batch_size, max_nodes)
mask[:, :actual_nodes] = 1
```

**Trade-off**: Simpler code, but wastes memory on padding.

**Recommendation**: Use Approach 1 (disconnected graph) for variable-size clauses, Approach 2 for fixed-size small clauses.

---

## GNN Architecture Considerations

### Message Passing

Standard GNN message passing with sparse tensors:

```python
class ClauseGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.message_net = nn.Linear(input_dim, hidden_dim)
        self.update_net = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, features, adjacency):
        # features: (num_nodes, input_dim)
        # adjacency: (num_nodes, num_nodes) sparse CSR

        # Compute messages
        messages = self.message_net(features)  # (num_nodes, hidden_dim)

        # Aggregate messages via sparse matrix multiply
        aggregated = torch.sparse.mm(adjacency, messages)  # (num_nodes, hidden_dim)

        # Update node representations
        combined = torch.cat([features, aggregated], dim=-1)
        updated = self.update_net(combined)

        return updated
```

### Multi-Layer Message Passing

Stack multiple GNN layers to capture long-range dependencies:

```python
class MultiLayerGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            ClauseGNN(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

    def forward(self, features, adjacency):
        h = features
        for layer in self.layers:
            h = layer(h, adjacency)
        return h
```

**With residual connections** for better gradient flow:

```python
def forward(self, features, adjacency):
    h = features
    for layer in self.layers:
        h_new = layer(h, adjacency)
        h = h + h_new  # Residual connection
    return h
```

---

## Implementation Roadmap

### Phase 1: Basic Representation
- [ ] Implement clause → graph conversion (COO format)
- [ ] Define node types and edge types
- [ ] Simple feature extraction (type + arity + depth)
- [ ] Test on hand-crafted examples

### Phase 2: PyTorch Integration
- [ ] Convert COO → CSR for efficiency
- [ ] Implement batching (disconnected graph approach)
- [ ] Test sparse matrix operations
- [ ] Benchmark memory usage

### Phase 3: Feature Engineering
- [ ] Add symbol hashing
- [ ] Compute structural features (depth, arity, etc.)
- [ ] Extract clause-level metadata
- [ ] Create feature normalization

### Phase 4: GNN Model
- [ ] Implement basic message passing GNN
- [ ] Add edge-type aware message passing
- [ ] Implement global pooling for clause embeddings
- [ ] Test on real TPTP clauses

### Phase 5: Integration
- [ ] Parse TPTP clauses to graphs
- [ ] Create dataset class for batching
- [ ] Integrate with Rust prover (export clause strings)
- [ ] End-to-end test: clause → graph → GNN → embedding

---

## Design Decisions & Alternatives

### Decision 1: Clause Root Node vs Direct Literal Nodes

**Chosen**: Include explicit CLAUSE root node

**Alternative**: Start graph at literals, no root

**Rationale**:
- Allows clause-level features (is_unit, is_horn, etc.)
- Consistent structure across all clauses (including empty clause ⊥)
- Makes global pooling more natural (pool from root)

**Trade-off**: One extra node per clause (~10% overhead)

### Decision 2: Separate PREDICATE and FUNCTION Types

**Chosen**: Separate types

**Alternative**: Unified SYMBOL type

**Rationale**:
- FOL distinguishes predicates (return bool) from functions (return terms)
- Different semantic roles in inference rules
- Predicate arguments are terms; function arguments are terms

**Trade-off**: More node types to handle

### Decision 3: Homogeneous vs Heterogeneous Edges

**Chosen**: Homogeneous (single edge type)

**Alternative**: Heterogeneous edges (HAS_LITERAL, HAS_ARG, EQUALITY_LHS, etc.)

**Rationale**:
- **Simpler**: Single adjacency matrix, no edge type management
- **Faster**: One sparse matmul per layer instead of multiple
- **Sufficient**: Node types + topology encode relationship information
- **Better generalization**: GNN learns from structure, not hard-coded types
- **Easier batching**: No need to align edge types across batch

**Trade-off**:
- GNN must learn to distinguish edge semantics from context
- Might need more layers to capture same information

**Evidence**: Recent work (Paliwal et al. 2020) shows homogeneous graphs work well for theorem proving when node features are rich.

### Decision 4: Argument Order Encoding

**Chosen**: Implicit via graph structure (directed edges) + optional position features

**Alternative**: Edge labels for positions

**Rationale**:
- Graph structure (parent-child order) implicitly encodes position
- Can add explicit `arg_position` feature if needed
- Keeps edges homogeneous

**Trade-off**: Position info might be harder for GNN to learn

**Future**: Add `arg_position` to node features if experiments show benefit

---

## Testing Strategy

### Unit Tests

```python
def test_simple_clause():
    """Test: P(x) | Q(a)"""
    graph = clause_to_graph("P(x) | Q(a)")
    assert graph.num_nodes == 7  # CLAUSE + 2 LIT + 2 PRED + 2 TERM
    assert graph.num_edges == 6  # All parent-child edges

def test_nested_function():
    """Test: P(f(g(x)))"""
    graph = clause_to_graph("P(f(g(x)))")
    assert graph.get_depth() == 4  # CLAUSE → LIT → PRED → f → g → x

def test_equality():
    """Test: x = f(x)"""
    graph = clause_to_graph("x = f(x)")
    # Check equality predicate node exists
    pred_node = graph.get_node_by_name("=")
    assert pred_node.type == NODE_TYPE_PREDICATE
    assert pred_node.features[13] == 1.0  # is_equality feature
```

### Integration Tests

```python
def test_tptp_clause_batch():
    """Test batching real TPTP clauses"""
    clauses = [
        "mult(e, X) = X",
        "mult(X, e) = X",
        "mult(inv(X), X) = e"
    ]

    graphs = [clause_to_graph(c) for c in clauses]
    batch = batch_graphs(graphs)

    # Run through GNN
    embeddings = gnn(batch.features, batch.adjacency)
    assert embeddings.shape == (3, hidden_dim)  # 3 clause embeddings
```

### Visualization Tests

Use matplotlib/networkx to visualize:

```python
def test_visualize_graph():
    graph = clause_to_graph("P(f(x), a) | ~Q(x)")
    visualize_graph(graph, save_path="test_graph.png")
    # Manually inspect image for correctness
```

---

## Performance Considerations

### Memory Efficiency

**Dense vs Sparse** for typical clause:
- 20 nodes, 30 edges
- Dense: 20×20 = 400 floats = 1.6 KB
- Sparse COO: 30 edges × 2 indices + 30 values = 90 ints + 30 floats = 480 bytes
- **Savings**: 3.3× for small clause

For batch of 100 clauses (avg 20 nodes each):
- Dense: 100 × 1.6 KB = 160 KB
- Sparse: 100 × 480 bytes = 48 KB
- **Savings**: 3.3×

For large clauses (100 nodes, 150 edges):
- Dense: 100×100 = 10,000 floats = 40 KB
- Sparse: 150 edges × (90+30) = 600 bytes
- **Savings**: 66×!

**Conclusion**: Sparse representation is 3-70× more memory efficient.

### Computation Speed

**Sparse Matrix Multiply** performance:
- Dense: O(n³) for n×n matrices
- Sparse: O(nnz × n) where nnz = number of non-zero entries

For clause graph with n=20, nnz=30:
- Dense: 20³ = 8,000 ops
- Sparse: 30 × 20 = 600 ops
- **Speedup**: 13×

**Caveat**: PyTorch sparse ops have overhead. Only worth it for:
- Large graphs (n > 50)
- Very sparse graphs (nnz << n²)

### Recommendations

1. **Use CSR for GNN forward pass** (fastest for matmul)
2. **Use COO for construction** (easiest to build)
3. **Convert once**: COO → CSR at batch creation time
4. **Cache conversions**: Don't reconvert every epoch
5. **Profile first**: Measure actual performance before optimizing

---


## Appendix: Complete Feature Specification

### Node Features (20 dimensions)

```python
node_features = [
    # Type (one-hot, 6 dims)
    is_clause,      # [0]
    is_literal,     # [1]
    is_predicate,   # [2]
    is_function,    # [3]
    is_variable,    # [4]
    is_constant,    # [5]

    # Structural (6 dims)
    arity,          # [6] Number of children
    depth,          # [7] Distance from root
    num_variables,  # [8] Variables in subtree
    num_constants,  # [9] Constants in subtree
    max_depth,      # [10] Max depth of subtree
    subtree_size,   # [11] Number of nodes in subtree

    # Semantic (5 dims)
    polarity,       # [12] 1.0=pos, 0.0=neg (literals only)
    is_equality,    # [13] Is equality predicate
    is_unit,        # [14] Is unit clause
    is_horn,        # [15] Is horn clause
    is_ground,      # [16] Is ground (no variables)

    # Symbol (3 dims)
    symbol_hash_1,  # [17] hash(name) % 1000 / 1000
    symbol_hash_2,  # [18] hash(name+"_2") % 1000 / 1000
    symbol_freq,    # [19] Frequency in dataset (if known)
]
```

### Edge Features (Not Used)

**Design choice**: We use **homogeneous edges without edge features**.

All information is encoded in:
1. **Node features** (type, arity, position in tree, etc.)
2. **Graph topology** (which nodes connect to which)
3. **Edge direction** (parent → child relationships)

**Alternative considered**: Edge features for argument positions
```python
# Not used in our design
edge_features = [
    arg_position,    # 0-indexed argument position
    is_last_arg,     # 1.0 if last argument
]
```

**Why not edge features?**
- Adds complexity to batching
- Requires edge-feature-aware GNN layers
- Positional info can be encoded in node features if needed
- Simpler = easier to debug and iterate

**Future**: Can add edge features later if experiments show benefit.

---

## Appendix: Example PyTorch Code

### Minimal Working Example

```python
import torch

# Simple clause: P(x) | Q(a)
# Nodes: [CLAUSE, LIT1, PRED_P, VAR_x, LIT2, PRED_Q, CONST_a]

# Adjacency (COO format)
indices = torch.tensor([
    [0, 0, 1, 2, 4, 5],  # source
    [1, 4, 2, 3, 5, 6]   # target
])
values = torch.ones(6)
adjacency = torch.sparse_coo_tensor(indices, values, size=(7, 7))

# Convert to CSR for GNN
adjacency_csr = adjacency.to_sparse_csr()

# Node features (simplified: type one-hot + arity)
features = torch.tensor([
    [1, 0, 0, 0, 0, 0, 0],  # CLAUSE
    [0, 1, 0, 0, 0, 0, 1],  # LIT1 (polarity=1)
    [0, 0, 1, 0, 0, 0, 1],  # PRED_P (arity=1)
    [0, 0, 0, 0, 1, 0, 0],  # VAR_x
    [0, 1, 0, 0, 0, 0, 0],  # LIT2 (polarity=0)
    [0, 0, 1, 0, 0, 0, 1],  # PRED_Q (arity=1)
    [0, 0, 0, 0, 0, 1, 0],  # CONST_a
], dtype=torch.float32)

# Simple GNN layer
class SimpleGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, hidden_dim)

    def forward(self, features, adjacency):
        # Message passing: aggregate neighbor features
        messages = torch.sparse.mm(adjacency, features)
        # Transform
        return torch.relu(self.linear(messages))

# Forward pass
gnn = SimpleGNN(input_dim=7, hidden_dim=16)
embeddings = gnn(features, adjacency_csr)
print(f"Node embeddings shape: {embeddings.shape}")  # (7, 16)

# Global pooling for clause embedding
clause_embedding = embeddings[0]  # Take root node
print(f"Clause embedding shape: {clause_embedding.shape}")  # (16,)
```

This example demonstrates the core components working together.
