"""PyTorch utilities for converting clause graphs to tensors

This module provides functions to convert ClauseGraphData objects to PyTorch
tensors, batch multiple graphs, and integrate with PyTorch Geometric.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False


def _check_torch():
    """Check if PyTorch is available"""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for graph utilities. "
            "Install with: pip install torch"
        )


def _check_torch_geometric():
    """Check if PyTorch Geometric is available"""
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError(
            "PyTorch Geometric is required for this function. "
            "Install with: pip install torch-geometric"
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


def to_torch_geometric(
    graph,
    y: Optional[Any] = None,
    device: str = "cpu"
) -> "Data":
    """Convert ClauseGraphData to PyTorch Geometric Data object

    Args:
        graph: ClauseGraphData object from Rust
        y: Optional label/target (for supervised learning)
        device: PyTorch device

    Returns:
        PyTorch Geometric Data object with:
            - data.edge_index: Edge connectivity (2, num_edges)
            - data.x: Node features (num_nodes, feature_dim)
            - data.node_types: Node type indices (num_nodes,)
            - data.y: Label (if provided)
            - data.num_nodes: Number of nodes

    Example:
        >>> graph = state.clause_to_graph(clause_id)
        >>> data = to_torch_geometric(graph, y=1)  # Label 1 = selected
        >>> print(data)
        Data(x=[4, 20], edge_index=[2, 3], node_types=[4], y=1)
    """
    _check_torch()
    _check_torch_geometric()

    tensors = to_torch_tensors(graph, device=device)

    data = Data(
        x=tensors['x'],
        edge_index=tensors['edge_index'],
        node_types=tensors['node_types'],
        num_nodes=tensors['num_nodes'],
    )

    if y is not None:
        if isinstance(y, (int, float)):
            data.y = torch.tensor([y], dtype=torch.float).to(device)
        else:
            data.y = torch.tensor(y).to(device)

    return data


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


def batch_graphs_geometric(
    graphs: List,
    labels: Optional[List] = None,
    device: str = "cpu"
) -> "Batch":
    """Batch graphs using PyTorch Geometric's Batch

    Args:
        graphs: List of ClauseGraphData objects
        labels: Optional list of labels
        device: PyTorch device

    Returns:
        PyTorch Geometric Batch object

    Example:
        >>> graphs = [state.clause_to_graph(id) for id in clause_ids]
        >>> batch = batch_graphs_geometric(graphs, labels=[0, 1, 1])
        >>> batch.num_graphs
        3
    """
    _check_torch()
    _check_torch_geometric()

    # Convert each graph to PyTorch Geometric Data
    data_list = []
    for i, graph in enumerate(graphs):
        y = labels[i] if labels is not None else None
        data = to_torch_geometric(graph, y=y, device=device)
        data_list.append(data)

    # Use PyG's batching
    return Batch.from_data_list(data_list)


def create_dataloader(
    graphs: List,
    labels: Optional[List] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    **kwargs
) -> "torch.utils.data.DataLoader":
    """Create a PyTorch DataLoader for clause graphs

    Args:
        graphs: List of ClauseGraphData objects
        labels: Optional list of labels
        batch_size: Batch size
        shuffle: Whether to shuffle data
        **kwargs: Additional arguments for DataLoader

    Returns:
        PyTorch DataLoader that yields batched graphs

    Example:
        >>> graphs = [state.clause_to_graph(id) for id in clause_ids]
        >>> labels = [0, 1, 1, 0, 1]
        >>> loader = create_dataloader(graphs, labels, batch_size=2)
        >>> for batch in loader:
        ...     print(batch['x'].shape)  # Node features
        ...     print(batch['y'].shape)  # Labels
    """
    _check_torch()
    _check_torch_geometric()

    from torch_geometric.loader import DataLoader

    # Convert to PyTorch Geometric Data objects
    data_list = []
    for i, graph in enumerate(graphs):
        y = labels[i] if labels is not None else None
        data = to_torch_geometric(graph, y=y)
        data_list.append(data)

    # Create DataLoader
    return DataLoader(
        data_list,
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs
    )


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

    # Compute depth
    node_features = graph.node_features()
    stats['max_depth'] = int(node_features[:, 7].max())

    return stats
