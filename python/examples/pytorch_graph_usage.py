#!/usr/bin/env python3
"""
Example: Using PyTorch utilities with clause graphs

This example demonstrates how to:
1. Convert clause graphs to PyTorch tensors
2. Batch multiple graphs for training
3. Create DataLoaders
4. Extract graph-level embeddings
5. Use with PyTorch Geometric
"""

import torch
from proofatlas import ProofState
from proofatlas.ml import (
    to_torch_tensors,
    to_torch_geometric,
    batch_graphs,
    create_dataloader,
    extract_graph_embeddings,
    get_node_type_masks,
    compute_graph_statistics,
)

print("=" * 70)
print("PyTorch Graph Utilities - Usage Examples")
print("=" * 70)

# ============================================================================
# Example 1: Convert Single Graph to PyTorch Tensors
# ============================================================================
print("\n" + "=" * 70)
print("Example 1: Converting Single Graph to PyTorch Tensors")
print("=" * 70)

state = ProofState()
tptp = "cnf(test, axiom, (p(X) | q(f(X))))."
clause_ids = state.add_clauses_from_tptp(tptp)

graph = state.clause_to_graph(clause_ids[0])
print(f"Clause: {state.clause_to_string(clause_ids[0])}")

# Convert to PyTorch tensors
tensors = to_torch_tensors(graph, device='cpu')

print(f"\nTensor shapes:")
print(f"  edge_index: {tensors['edge_index'].shape}")
print(f"  x (features): {tensors['x'].shape}")
print(f"  node_types: {tensors['node_types'].shape}")
print(f"\nData types:")
print(f"  edge_index: {tensors['edge_index'].dtype}")
print(f"  x: {tensors['x'].dtype}")
print(f"  node_types: {tensors['node_types'].dtype}")

# ============================================================================
# Example 2: PyTorch Geometric Data Object
# ============================================================================
print("\n" + "=" * 70)
print("Example 2: PyTorch Geometric Data Object")
print("=" * 70)

try:
    # Convert with label (1 = selected clause, 0 = not selected)
    data = to_torch_geometric(graph, y=1, device='cpu')
    print(f"Data object: {data}")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")
    print(f"  Label: {data.y.item()}")
except ImportError:
    print("PyTorch Geometric not installed. Skipping this example.")

# ============================================================================
# Example 3: Batching Multiple Graphs
# ============================================================================
print("\n" + "=" * 70)
print("Example 3: Batching Multiple Graphs")
print("=" * 70)

# Create multiple clauses
tptp_batch = """
cnf(c1, axiom, p(X)).
cnf(c2, axiom, (p(X) | q(Y))).
cnf(c3, axiom, (~p(a) | r(b))).
"""
state2 = ProofState()
clause_ids2 = state2.add_clauses_from_tptp(tptp_batch)

graphs = [state2.clause_to_graph(id) for id in clause_ids2]
labels = [0, 1, 0]  # Example labels: clause 2 was selected

batched = batch_graphs(graphs, labels=labels, device='cpu')

print(f"Batched {batched['num_graphs']} graphs:")
print(f"  Total nodes: {batched['x'].shape[0]}")
print(f"  Total edges: {batched['edge_index'].shape[1]}")
print(f"  Batch indices shape: {batched['batch'].shape}")
print(f"  Labels: {batched['y']}")

# Show which nodes belong to which graph
print(f"\nBatch assignment (first 15 nodes):")
print(f"  {batched['batch'][:15]}")

# ============================================================================
# Example 4: DataLoader for Training
# ============================================================================
print("\n" + "=" * 70)
print("Example 4: Creating DataLoader for Training")
print("=" * 70)

try:
    # Create more clauses for a larger dataset
    tptp_dataset = """
    cnf(c1, axiom, p(a)).
    cnf(c2, axiom, (p(X) | q(X))).
    cnf(c3, axiom, (~p(X) | r(X))).
    cnf(c4, axiom, (~q(X) | s(X))).
    cnf(c5, axiom, (t(X, Y) | u(Y))).
    """
    state3 = ProofState()
    clause_ids3 = state3.add_clauses_from_tptp(tptp_dataset)

    dataset_graphs = [state3.clause_to_graph(id) for id in clause_ids3]
    dataset_labels = [0, 1, 1, 0, 1]  # Binary labels for classification

    # Create DataLoader
    loader = create_dataloader(
        dataset_graphs,
        dataset_labels,
        batch_size=2,
        shuffle=True
    )

    print(f"Created DataLoader:")
    print(f"  Dataset size: {len(dataset_graphs)}")
    print(f"  Batch size: 2")
    print(f"  Number of batches: {len(loader)}")

    # Iterate through batches
    print(f"\nFirst batch:")
    for i, batch in enumerate(loader):
        if i == 0:
            print(f"  Batch nodes: {batch.x.shape[0]}")
            print(f"  Batch edges: {batch.edge_index.shape[1]}")
            print(f"  Batch labels: {batch.y}")
        break

except ImportError:
    print("PyTorch Geometric not installed. Skipping DataLoader example.")

# ============================================================================
# Example 5: Extracting Graph-Level Embeddings
# ============================================================================
print("\n" + "=" * 70)
print("Example 5: Extracting Graph-Level Embeddings")
print("=" * 70)

# Simulate node embeddings from a GNN
num_total_nodes = batched['x'].shape[0]
embedding_dim = 64
node_embeddings = torch.randn(num_total_nodes, embedding_dim)

print(f"Simulated node embeddings: {node_embeddings.shape}")

# Extract graph-level embeddings using different aggregation methods
for method in ['mean', 'sum', 'max', 'root']:
    graph_emb = extract_graph_embeddings(
        node_embeddings,
        batched['batch'],
        method=method
    )
    print(f"  {method:8s} pooling: {graph_emb.shape}")

# ============================================================================
# Example 6: Node Type Masking
# ============================================================================
print("\n" + "=" * 70)
print("Example 6: Node Type Masking")
print("=" * 70)

# Get node type masks
masks = get_node_type_masks(tensors['node_types'])

print("Node type counts in first graph:")
for node_type, mask in masks.items():
    count = mask.sum().item()
    if count > 0:
        print(f"  {node_type:10s}: {count}")

# Use masks to select specific node types
print(f"\nExample: Extract only variable node features:")
variable_features = tensors['x'][masks['variable']]
print(f"  Shape: {variable_features.shape}")

# ============================================================================
# Example 7: Graph Statistics
# ============================================================================
print("\n" + "=" * 70)
print("Example 7: Computing Graph Statistics")
print("=" * 70)

stats = compute_graph_statistics(graph)

print("Graph statistics:")
for key, value in sorted(stats.items()):
    print(f"  {key:20s}: {value}")

# ============================================================================
# Example 8: Complete Training Pipeline Skeleton
# ============================================================================
print("\n" + "=" * 70)
print("Example 8: Complete Training Pipeline Skeleton")
print("=" * 70)

print("""
# Pseudo-code for complete training pipeline:

# 1. Load data
state = ProofState()
clause_ids = state.add_clauses_from_tptp(tptp_content)

# 2. Create graphs with labels
graphs = [state.clause_to_graph(id) for id in clause_ids]
labels = get_labels_from_saturation_run()  # Your labeling function

# 3. Create DataLoader
train_loader = create_dataloader(
    graphs[:800],  # 80% train
    labels[:800],
    batch_size=32,
    shuffle=True
)

val_loader = create_dataloader(
    graphs[800:],  # 20% val
    labels[800:],
    batch_size=32,
    shuffle=False
)

# 4. Training loop
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        # Forward pass
        node_emb = gnn(batch.x, batch.edge_index)
        graph_emb = extract_graph_embeddings(node_emb, batch.batch, method='mean')
        predictions = classifier(graph_emb)

        # Compute loss
        loss = criterion(predictions, batch.y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = evaluate(model, val_loader)

    print(f"Epoch {epoch}: train_loss={loss:.4f}, val_loss={val_loss:.4f}")

# 5. Use trained model for clause selection
def score_clause(clause_id):
    graph = state.clause_to_graph(clause_id)
    data = to_torch_geometric(graph)
    with torch.no_grad():
        node_emb = gnn(data.x, data.edge_index)
        graph_emb = extract_graph_embeddings(
            node_emb,
            torch.zeros(data.num_nodes, dtype=torch.long),
            method='mean'
        )
        score = classifier(graph_emb)
    return score.item()
""")

print("\n" + "=" * 70)
print("All examples completed successfully!")
print("=" * 70)
