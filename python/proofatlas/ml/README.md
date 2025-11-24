# ProofAtlas Machine Learning Module

PyTorch utilities for converting clause graphs to tensors and training GNNs for clause selection.

## Installation

```bash
# Basic installation (NumPy only)
pip install proofatlas

# With PyTorch support
pip install proofatlas torch

# With PyTorch Geometric support (recommended for GNN training)
pip install proofatlas torch torch-geometric
```

## Quick Start

```python
from proofatlas import ProofState
from proofatlas.ml import to_torch_tensors, batch_graphs

# Create clause graphs
state = ProofState()
clause_ids = state.add_clauses_from_tptp("cnf(test, axiom, p(X)).")
graph = state.clause_to_graph(clause_ids[0])

# Convert to PyTorch tensors
tensors = to_torch_tensors(graph)
print(tensors['edge_index'].shape)  # (2, num_edges)
print(tensors['x'].shape)           # (num_nodes, 20)

# Batch multiple graphs
graphs = [state.clause_to_graph(id) for id in clause_ids]
batched = batch_graphs(graphs, labels=[0, 1, 1])
```

## API Reference

### Tensor Conversion

#### `to_torch_tensors(graph, device='cpu')`

Convert ClauseGraphData to PyTorch tensors.

**Returns:**
- `edge_index`: LongTensor (2, num_edges) - Edge connectivity
- `x`: FloatTensor (num_nodes, 20) - Node features
- `node_types`: ByteTensor (num_nodes,) - Node type indices
- `num_nodes`: int
- `num_edges`: int

#### `to_torch_geometric(graph, y=None, device='cpu')`

Convert to PyTorch Geometric Data object.

**Args:**
- `graph`: ClauseGraphData from Rust
- `y`: Optional label/target
- `device`: PyTorch device

**Returns:** PyTorch Geometric `Data` object

#### `to_sparse_adjacency(graph, format='coo', device='cpu')`

Convert to sparse adjacency matrix.

**Args:**
- `format`: 'coo' or 'csr'

**Returns:** Sparse tensor (num_nodes, num_nodes)

### Batching

#### `batch_graphs(graphs, labels=None, device='cpu')`

Batch multiple graphs into single disconnected graph.

**Returns:**
- `edge_index`: Combined edges
- `x`: Combined features
- `node_types`: Combined types
- `batch`: Graph assignment for each node
- `num_graphs`: Number of graphs
- `y`: Labels (if provided)

#### `batch_graphs_geometric(graphs, labels=None, device='cpu')`

Batch using PyTorch Geometric's Batch.

**Returns:** PyTorch Geometric `Batch` object

#### `create_dataloader(graphs, labels=None, batch_size=32, shuffle=True, **kwargs)`

Create PyTorch DataLoader for graphs.

**Returns:** PyTorch Geometric `DataLoader`

### Embedding Extraction

#### `extract_graph_embeddings(node_embeddings, batch, method='mean')`

Extract graph-level embeddings from node embeddings.

**Args:**
- `node_embeddings`: Node embeddings (num_nodes, embedding_dim)
- `batch`: Batch assignment (num_nodes,)
- `method`: 'mean', 'sum', 'max', or 'root'

**Returns:** Graph embeddings (num_graphs, embedding_dim)

### Utilities

#### `get_node_type_masks(node_types)`

Create boolean masks for each node type.

**Returns:** Dict mapping type names to boolean masks

#### `compute_graph_statistics(graph)`

Compute statistics about a clause graph.

**Returns:** Dict with node counts, edge counts, max depth, etc.

## Graph Structure

### Node Types

- `CLAUSE` (0): Clause root node
- `LITERAL` (1): Literal node
- `PREDICATE` (2): Predicate/function symbol
- `FUNCTION` (3): Function symbol
- `VARIABLE` (4): Variable
- `CONSTANT` (5): Constant

### Node Features (20 dimensions)

| Index | Feature | Description |
|-------|---------|-------------|
| 0-5 | Type one-hot | Node type encoding |
| 6 | Arity | Function/predicate arity |
| 7 | Depth | Depth in syntax tree |
| 8-11 | Structural | Reserved for future use |
| 12 | Polarity | Literal polarity (1=positive, 0=negative) |
| 13 | Is equality | Equality predicate flag |
| 14 | Is unit | Unit clause flag |
| 15 | Is Horn | Horn clause flag |
| 16 | Is ground | Ground clause flag (no variables) |
| 17-18 | Symbol hash | Hash values for symbol identity |
| 19 | Reserved | Future use |

### Edge Structure

- Edges represent parent-child relationships in the syntax tree
- Graph is a tree rooted at the clause node
- Edges are directed (parent → child)
- Stored in COO format: shape (2, num_edges)

## Usage Examples

### Example 1: Simple Conversion

```python
from proofatlas import ProofState
from proofatlas.ml import to_torch_tensors

state = ProofState()
clause_ids = state.add_clauses_from_tptp("cnf(test, axiom, p(X, a)).")
graph = state.clause_to_graph(clause_ids[0])

tensors = to_torch_tensors(graph)
print(f"Nodes: {tensors['num_nodes']}")
print(f"Edges: {tensors['num_edges']}")
```

### Example 2: Batching for Training

```python
from proofatlas.ml import batch_graphs

# Create multiple graphs
graphs = [state.clause_to_graph(id) for id in clause_ids]
labels = [0, 1, 1, 0]  # Binary classification labels

# Batch together
batched = batch_graphs(graphs, labels=labels)

# Use in training
edge_index = batched['edge_index']
node_features = batched['x']
batch_assignment = batched['batch']
targets = batched['y']
```

### Example 3: PyTorch Geometric Integration

```python
from proofatlas.ml import create_dataloader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn

# Create DataLoader
loader = create_dataloader(
    graphs,
    labels,
    batch_size=32,
    shuffle=True
)

# Define GNN model
class ClauseGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(20, 64)
        self.conv2 = GCNConv(64, 64)
        self.classifier = nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GNN layers
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()

        # Global pooling
        x = global_mean_pool(x, batch)

        # Classification
        return self.classifier(x)

# Training loop
model = ClauseGNN()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

for epoch in range(10):
    for batch in loader:
        optimizer.zero_grad()
        predictions = model(batch)
        loss = criterion(predictions.squeeze(), batch.y)
        loss.backward()
        optimizer.step()
```

### Example 4: Custom Pooling

```python
from proofatlas.ml import extract_graph_embeddings

# After GNN forward pass
node_embeddings = gnn(batch.x, batch.edge_index)

# Different pooling strategies
mean_emb = extract_graph_embeddings(node_embeddings, batch.batch, method='mean')
sum_emb = extract_graph_embeddings(node_embeddings, batch.batch, method='sum')
max_emb = extract_graph_embeddings(node_embeddings, batch.batch, method='max')
root_emb = extract_graph_embeddings(node_embeddings, batch.batch, method='root')
```

### Example 5: Node Type Filtering

```python
from proofatlas.ml import get_node_type_masks

# Get type-specific masks
masks = get_node_type_masks(tensors['node_types'])

# Extract only variable nodes
variable_features = tensors['x'][masks['variable']]

# Count node types
for type_name, mask in masks.items():
    count = mask.sum().item()
    print(f"{type_name}: {count}")
```

## Performance

- **Sparse representation**: Efficient memory usage for large graphs
- **Batch processing**: Combine multiple graphs for GPU efficiency
- **Zero-copy transfer**: Direct NumPy → PyTorch conversion
- **Fast aggregation**: Optimized pooling operations

## Testing

```bash
# Run all ML tests
pytest python/tests/ml/ -v

# Run specific test file
pytest python/tests/ml/test_graph_export.py -v
pytest python/tests/ml/test_graph_utils.py -v
```

## Complete Example

See `python/examples/pytorch_graph_usage.py` for a comprehensive example demonstrating all features.

## Next Steps

1. **Data Collection**: Instrument saturation loop to collect training data
2. **Baseline Model**: Implement GCN/GAT for clause selection
3. **Training Pipeline**: Set up training with validation
4. **Integration**: Use trained model in prover

## References

- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Graph Neural Networks](https://distill.pub/2021/gnn-intro/)
