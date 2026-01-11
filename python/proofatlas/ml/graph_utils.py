"""PyTorch utilities for converting clause graphs to tensors

This module provides functions to convert ClauseGraphData objects to PyTorch
tensors and batch multiple graphs.
"""

from typing import List, Dict, Optional
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _check_torch():
    """Check if PyTorch is available"""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for graph utilities. "
            "Install with: pip install torch"
        )


def to_torch_tensors(graph, device: str = "cpu") -> Dict[str, "torch.Tensor"]:
    """Convert ClauseGraphData to PyTorch tensors

    Args:
        graph: ClauseGraphData object from Rust
        device: PyTorch device ('cpu', 'cuda', 'cuda:0', etc.)

    Returns:
        Dictionary with keys:
            - 'edge_index': Long tensor of shape (2, num_edges)
            - 'x': Float tensor of shape (num_nodes, feature_dim) - node features
            - 'node_types': Byte tensor of shape (num_nodes,)
            - 'num_nodes': int
            - 'num_edges': int

    Example:
        >>> state = ProofState()
        >>> clause_ids = state.add_clauses_from_tptp("cnf(test, axiom, p(X)).")
        >>> graph = state.clause_to_graph(clause_ids[0])
        >>> tensors = to_torch_tensors(graph)
        >>> tensors['edge_index'].shape
        torch.Size([2, 3])
    """
    _check_torch()

    # Get numpy arrays from Rust
    edge_indices = graph.edge_indices()  # (2, num_edges)
    node_features = graph.node_features()  # (num_nodes, feature_dim)
    node_types = graph.node_types()  # (num_nodes,)

    # Convert to PyTorch tensors
    edge_index = torch.from_numpy(edge_indices).long().to(device)
    x = torch.from_numpy(node_features).float().to(device)
    node_type = torch.from_numpy(node_types).byte().to(device)

    return {
        'edge_index': edge_index,
        'x': x,
        'node_types': node_type,
        'num_nodes': graph.num_nodes(),
        'num_edges': graph.num_edges(),
    }


def to_sparse_adjacency(
    graph,
    format: str = "coo",
    device: str = "cpu"
) -> "torch.Tensor":
    """Convert graph to PyTorch sparse adjacency matrix

    Args:
        graph: ClauseGraphData object
        format: Sparse format - 'coo' or 'csr'
        device: PyTorch device

    Returns:
        Sparse tensor of shape (num_nodes, num_nodes)

    Example:
        >>> adj_coo = to_sparse_adjacency(graph, format='coo')
        >>> adj_csr = to_sparse_adjacency(graph, format='csr')
    """
    _check_torch()

    edge_indices = graph.edge_indices()
    num_nodes = graph.num_nodes()
    num_edges = graph.num_edges()

    # Convert to PyTorch
    indices = torch.from_numpy(edge_indices).long().to(device)
    values = torch.ones(num_edges, dtype=torch.float32, device=device)

    # Create COO sparse tensor
    adj = torch.sparse_coo_tensor(
        indices,
        values,
        size=(num_nodes, num_nodes),
        device=device
    )

    # Convert to CSR if requested
    if format == "csr":
        adj = adj.to_sparse_csr()
    elif format != "coo":
        raise ValueError(f"Unknown format: {format}. Use 'coo' or 'csr'")

    return adj


def batch_graphs(
    graphs: List,
    labels: Optional[List] = None,
    device: str = "cpu"
) -> Dict[str, "torch.Tensor"]:
    """Batch multiple clause graphs into a single disconnected graph

    This creates a single large graph where each input graph is a disconnected
    component. Node and edge indices are adjusted accordingly.

    Args:
        graphs: List of ClauseGraphData objects
        labels: Optional list of labels (one per graph)
        device: PyTorch device

    Returns:
        Dictionary with:
            - 'edge_index': Combined edge indices (2, total_edges)
            - 'x': Combined node features (total_nodes, feature_dim)
            - 'node_types': Combined node types (total_nodes,)
            - 'batch': Batch assignment for each node (total_nodes,)
            - 'num_graphs': Number of graphs in batch
            - 'y': Labels if provided (num_graphs,)

    Example:
        >>> graphs = [state.clause_to_graph(id) for id in clause_ids]
        >>> batched = batch_graphs(graphs, labels=[0, 1, 1])
        >>> batched['batch']  # Which graph each node belongs to
        tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    """
    _check_torch()

    if len(graphs) == 0:
        raise ValueError("Cannot batch empty list of graphs")

    # Collect all data
    all_edges = []
    all_features = []
    all_types = []
    batch_indices = []

    node_offset = 0

    for graph_idx, graph in enumerate(graphs):
        # Get numpy arrays
        edges = graph.edge_indices()  # (2, num_edges)
        features = graph.node_features()  # (num_nodes, feature_dim)
        types = graph.node_types()  # (num_nodes,)

        num_nodes = graph.num_nodes()

        # Adjust edge indices by offset
        adjusted_edges = edges + node_offset

        all_edges.append(adjusted_edges)
        all_features.append(features)
        all_types.append(types)

        # Track which graph each node belongs to
        batch_indices.extend([graph_idx] * num_nodes)

        node_offset += num_nodes

    # Concatenate everything
    edge_index = torch.from_numpy(
        np.concatenate(all_edges, axis=1)
    ).long().to(device)

    x = torch.from_numpy(
        np.concatenate(all_features, axis=0)
    ).float().to(device)

    node_types = torch.from_numpy(
        np.concatenate(all_types, axis=0)
    ).byte().to(device)

    batch = torch.tensor(batch_indices, dtype=torch.long, device=device)

    result = {
        'edge_index': edge_index,
        'x': x,
        'node_types': node_types,
        'batch': batch,
        'num_graphs': len(graphs),
    }

    # Add labels if provided
    if labels is not None:
        if len(labels) != len(graphs):
            raise ValueError(
                f"Number of labels ({len(labels)}) must match "
                f"number of graphs ({len(graphs)})"
            )
        result['y'] = torch.tensor(labels, dtype=torch.float, device=device)

    return result


def extract_graph_embeddings(
    node_embeddings: "torch.Tensor",
    batch: "torch.Tensor",
    method: str = "mean"
) -> "torch.Tensor":
    """Extract graph-level embeddings from node embeddings

    Args:
        node_embeddings: Node embeddings (num_nodes, embedding_dim)
        batch: Batch assignment (num_nodes,)
        method: Aggregation method - 'mean', 'sum', 'max', or 'root'

    Returns:
        Graph embeddings (num_graphs, embedding_dim)

    Example:
        >>> # After GNN forward pass
        >>> node_emb = gnn(batched['x'], batched['edge_index'])
        >>> graph_emb = extract_graph_embeddings(node_emb, batched['batch'])
    """
    _check_torch()

    num_graphs = batch.max().item() + 1
    embedding_dim = node_embeddings.shape[1]

    if method == "mean":
        # Average pooling
        graph_emb = torch.zeros(num_graphs, embedding_dim, device=node_embeddings.device)
        for i in range(num_graphs):
            mask = batch == i
            graph_emb[i] = node_embeddings[mask].mean(dim=0)

    elif method == "sum":
        # Sum pooling
        graph_emb = torch.zeros(num_graphs, embedding_dim, device=node_embeddings.device)
        for i in range(num_graphs):
            mask = batch == i
            graph_emb[i] = node_embeddings[mask].sum(dim=0)

    elif method == "max":
        # Max pooling
        graph_emb = torch.zeros(num_graphs, embedding_dim, device=node_embeddings.device)
        for i in range(num_graphs):
            mask = batch == i
            graph_emb[i] = node_embeddings[mask].max(dim=0)[0]

    elif method == "root":
        # Use root node (node 0 of each graph)
        # Find first node of each graph
        root_indices = []
        for i in range(num_graphs):
            mask = batch == i
            root_idx = torch.where(mask)[0][0]  # First node in this graph
            root_indices.append(root_idx)
        root_indices = torch.tensor(root_indices, device=node_embeddings.device)
        graph_emb = node_embeddings[root_indices]

    else:
        raise ValueError(
            f"Unknown aggregation method: {method}. "
            f"Use 'mean', 'sum', 'max', or 'root'"
        )

    return graph_emb


def get_node_type_masks(
    node_types: "torch.Tensor",
) -> Dict[str, "torch.Tensor"]:
    """Create boolean masks for each node type

    Args:
        node_types: Node type indices (num_nodes,)

    Returns:
        Dictionary mapping type names to boolean masks

    Example:
        >>> masks = get_node_type_masks(node_types)
        >>> clause_nodes = tensors['x'][masks['clause']]
        >>> literal_nodes = tensors['x'][masks['literal']]
    """
    _check_torch()

    # Node type constants
    TYPE_CLAUSE = 0
    TYPE_LITERAL = 1
    TYPE_PREDICATE = 2
    TYPE_FUNCTION = 3
    TYPE_VARIABLE = 4
    TYPE_CONSTANT = 5

    return {
        'clause': node_types == TYPE_CLAUSE,
        'literal': node_types == TYPE_LITERAL,
        'predicate': node_types == TYPE_PREDICATE,
        'function': node_types == TYPE_FUNCTION,
        'variable': node_types == TYPE_VARIABLE,
        'constant': node_types == TYPE_CONSTANT,
    }


def compute_graph_statistics(graph) -> Dict[str, int]:
    """Compute statistics about a clause graph

    Args:
        graph: ClauseGraphData object

    Returns:
        Dictionary with statistics

    Example:
        >>> stats = compute_graph_statistics(graph)
        >>> print(f"Variables: {stats['num_variables']}")
    """
    node_types = graph.node_types()

    # Count each node type
    unique, counts = np.unique(node_types, return_counts=True)
    type_counts = dict(zip(unique.tolist(), counts.tolist()))

    # Get default values
    TYPE_NAMES = ['clause', 'literal', 'predicate', 'function', 'variable', 'constant']

    stats = {
        'num_nodes': graph.num_nodes(),
        'num_edges': graph.num_edges(),
        'feature_dim': graph.feature_dim(),
    }

    # Add type-specific counts
    for i, name in enumerate(TYPE_NAMES):
        stats[f'num_{name}s'] = type_counts.get(i, 0)

    # Compute depth (depth is at feature index 3 in the 8-dim format)
    node_features = graph.node_features()
    stats['max_depth'] = int(node_features[:, 3].max())

    return stats
