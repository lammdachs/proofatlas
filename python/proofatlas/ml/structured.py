"""Convert structured clause data to graphs and strings.

This module provides converters for the model-independent structured trace format.
Traces are stored as JSON with full clause structure, and converted to graphs
or strings at training time.

Example structured clause:
    {
        "literals": [
            {"polarity": true, "atom": {"predicate": "=", "args": [
                {"type": "Function", "name": "mult", "args": [
                    {"type": "Variable", "name": "X"},
                    {"type": "Variable", "name": "Y"}
                ]},
                {"type": "Variable", "name": "Z"}
            ]}}
        ],
        "label": 1,
        "age": 0,
        "role": "axiom"
    }
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Node type constants (matching Rust GraphBuilder)
TYPE_CLAUSE = 0
TYPE_LITERAL = 1
TYPE_PREDICATE = 2
TYPE_FUNCTION = 3
TYPE_VARIABLE = 4
TYPE_CONSTANT = 5

# Node feature indices (3 features per node for GCN encoder)
FEAT_NODE_TYPE = 0
FEAT_ARITY = 1
FEAT_ARG_POSITION = 2

# Clause feature indices (3 features per clause for scorer, sinusoidal encoded)
CLAUSE_FEAT_AGE = 0
CLAUSE_FEAT_ROLE = 1
CLAUSE_FEAT_SIZE = 2

# Role encoding
ROLE_MAP = {
    "axiom": 0,
    "hypothesis": 1,
    "definition": 2,
    "negated_conjecture": 3,
    "derived": 4,
}


def clause_to_string(clause: Dict[str, Any]) -> str:
    """Convert structured clause to TPTP-style string.

    Args:
        clause: Structured clause dict with literals

    Returns:
        String representation like "mult(X, Y) = Z | ~p(X)"
    """
    literals = clause.get("literals", [])
    if not literals:
        return "[]"

    lit_strs = []
    for lit in literals:
        atom = lit["atom"]
        pred = atom["predicate"]
        args = atom.get("args", [])

        # Build atom string
        if pred == "=" and len(args) == 2:
            # Equality: show as t1 = t2
            atom_str = f"{_term_to_string(args[0])} = {_term_to_string(args[1])}"
        elif args:
            args_str = ", ".join(_term_to_string(a) for a in args)
            atom_str = f"{pred}({args_str})"
        else:
            atom_str = pred

        # Add polarity
        if lit["polarity"]:
            lit_strs.append(atom_str)
        else:
            lit_strs.append(f"~{atom_str}")

    return " | ".join(lit_strs)


def _term_to_string(term: Dict[str, Any]) -> str:
    """Convert structured term to string."""
    term_type = term["type"]

    if term_type == "Variable":
        return term["name"]
    elif term_type == "Constant":
        return term["name"]
    elif term_type == "Function":
        name = term["name"]
        args = term.get("args", [])
        if args:
            args_str = ", ".join(_term_to_string(a) for a in args)
            return f"{name}({args_str})"
        else:
            return name
    else:
        return f"?{term_type}"


def clause_to_graph(
    clause: Dict[str, Any],
    max_age: int = 1000,
) -> Dict[str, Any]:
    """Convert structured clause to graph (numpy arrays).

    The new architecture separates node-level and clause-level features:
    - Node features (3d): type, arity, arg_pos (for GCN encoder)
    - Clause features (3d): age, role, size (for scorer, sinusoidal encoded)

    Returns numpy arrays for efficient batching. Converted to torch tensors
    in batch_graphs().

    Args:
        clause: Structured clause dict
        max_age: Maximum age for normalization

    Returns:
        Dictionary with numpy arrays:
            edge_index: [2, num_edges] edge indices
            x: [num_nodes, 3] node features (type, arity, arg_pos)
            node_types: [num_nodes] node type indices
            num_nodes: int
            num_edges: int
            clause_features: [3] raw clause features (age, role, size)
    """
    builder = _GraphBuilder(max_age)
    builder.build_clause(clause)

    return builder.to_numpy(clause)


class _GraphBuilder:
    """Build graph representation from structured clause.

    New architecture (from IJCAR26 plan):
    - Node features (3d): type, arity, arg_pos (for GCN encoder)
    - Clause features (3d): age, role, size (for scorer, sinusoidal encoded)
    """

    # Node feature dimension (type, arity, arg_pos)
    NODE_FEATURE_DIM = 3

    # Clause feature dimension (age, role, size)
    CLAUSE_FEATURE_DIM = 3

    def __init__(self, max_age: int = 1000):
        self.max_age = max_age
        # Store as flat lists for efficiency (avoid numpy array per node)
        self.node_features: List[Tuple[float, float, float]] = []
        self.node_types: List[int] = []
        self.node_names: List[str] = []
        self.edge_src: List[int] = []
        self.edge_dst: List[int] = []

    def build_clause(self, clause: Dict[str, Any]) -> int:
        """Build graph for clause, return root node index."""
        literals = clause.get("literals", [])

        # Create clause node
        clause_idx = self._add_node(TYPE_CLAUSE, len(literals), 0, name="CLAUSE")

        # Add literals
        for lit_pos, lit in enumerate(literals):
            lit_idx = self._build_literal(lit, arg_pos=lit_pos)
            self.edge_src.append(clause_idx)
            self.edge_dst.append(lit_idx)

        return clause_idx

    def _build_literal(self, lit: Dict[str, Any], arg_pos: int) -> int:
        """Build graph for literal."""
        atom = lit["atom"]
        args = atom.get("args", [])

        # Create literal node (arity=1: one child which is the predicate)
        lit_idx = self._add_node(TYPE_LITERAL, 1, arg_pos, name="LIT")

        # Create predicate node
        pred_idx = self._add_node(
            node_type=TYPE_PREDICATE,
            arity=len(args),
            arg_pos=0,
            name=atom["predicate"],
        )
        self.edge_src.append(lit_idx)
        self.edge_dst.append(pred_idx)

        # Add arguments
        for child_pos, arg in enumerate(args):
            arg_idx = self._build_term(arg, arg_pos=child_pos)
            self.edge_src.append(pred_idx)
            self.edge_dst.append(arg_idx)

        return lit_idx

    def _build_term(self, term: Dict[str, Any], arg_pos: int) -> int:
        """Build graph for term."""
        term_type = term["type"]

        if term_type == "Variable":
            return self._add_node(TYPE_VARIABLE, 0, arg_pos, name="VAR")
        elif term_type == "Constant":
            return self._add_node(TYPE_CONSTANT, 0, arg_pos, name=term["name"])
        elif term_type == "Function":
            args = term.get("args", [])
            func_idx = self._add_node(TYPE_FUNCTION, len(args), arg_pos, name=term["name"])

            for child_pos, child in enumerate(args):
                child_idx = self._build_term(child, arg_pos=child_pos)
                self.edge_src.append(func_idx)
                self.edge_dst.append(child_idx)

            return func_idx
        else:
            # Unknown term type, treat as constant
            return self._add_node(TYPE_CONSTANT, 0, arg_pos, name=term.get("name", "?"))

    def _add_node(self, node_type: int, arity: int = 0, arg_pos: int = 0, name: str = "") -> int:
        """Add a node with 3-dim feature vector (type, arity, arg_pos) and symbol name."""
        idx = len(self.node_features)
        self.node_features.append((float(node_type), float(arity), float(arg_pos)))
        self.node_types.append(node_type)
        self.node_names.append(name)
        return idx

    def to_numpy(self, clause: Dict[str, Any]) -> Dict[str, Any]:
        """Convert to numpy arrays (for efficient batching later).

        Args:
            clause: Original clause dict (for extracting clause-level features)

        Returns:
            Dictionary with numpy arrays (converted to torch tensors during batching)
        """
        num_nodes = len(self.node_features)
        num_edges = len(self.edge_src)

        # Node features
        if self.node_features:
            x = np.array(self.node_features, dtype=np.float32)
        else:
            x = np.zeros((0, self.NODE_FEATURE_DIM), dtype=np.float32)

        # Edge index
        if self.edge_src:
            edge_index = np.array([self.edge_src, self.edge_dst], dtype=np.int64)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)

        # Node types
        node_types = np.array(self.node_types, dtype=np.uint8)

        # Clause-level features (raw values, will be sinusoidal encoded by scorer)
        age = clause.get("age", 0)
        role = clause.get("role", "derived")
        size = len(clause.get("literals", []))

        # Normalize age by max_age for consistency
        normalized_age = float(age) / float(max(self.max_age, 1))
        role_idx = float(ROLE_MAP.get(role, 4))

        clause_features = np.array(
            [normalized_age, role_idx, float(size)], dtype=np.float32
        )

        return {
            'edge_index': edge_index,
            'x': x,
            'node_types': node_types,
            'node_names': self.node_names,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'clause_features': clause_features,
        }


def clauses_to_strings(clauses: List[Dict[str, Any]]) -> List[str]:
    """Convert list of structured clauses to strings.

    Args:
        clauses: List of structured clause dicts

    Returns:
        List of string representations
    """
    return [clause_to_string(c) for c in clauses]


def clauses_to_graphs(
    clauses: List[Dict[str, Any]],
    max_age: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Convert list of structured clauses to graph numpy arrays.

    Args:
        clauses: List of structured clause dicts
        max_age: Maximum age for normalization (default: len(clauses))

    Returns:
        List of graph dicts with numpy arrays
    """
    if max_age is None:
        max_age = len(clauses)

    return [clause_to_graph(c, max_age) for c in clauses]


def load_structured_trace(path: str) -> Dict[str, Any]:
    """Load a structured trace from JSON file.

    Args:
        path: Path to .json trace file

    Returns:
        Dict with proof_found, time_seconds, clauses
    """
    import json
    with open(path) as f:
        return json.load(f)


def batch_graphs(
    graphs: List[Dict[str, Any]],
    labels: Optional[List[int]] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Batch multiple clause graphs with sparse adjacency and pool matrices.

    Args:
        graphs: List of graph dicts from clause_to_graph (numpy arrays)
        labels: Optional list of labels (one per graph)
        device: PyTorch device

    Returns:
        Dict with:
            x: [num_nodes, 3] node features
            adj: Sparse normalized adjacency matrix
            pool_matrix: Sparse pooling matrix [num_clauses, num_nodes]
            clause_features: [num_clauses, 3] clause features
            labels: [num_clauses] if provided
            batch: [num_nodes] clause index for each node
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for batching")

    import torch

    if len(graphs) == 0:
        raise ValueError("Cannot batch empty list of graphs")

    all_x = []
    all_edges = []
    all_clause_features = []
    batch_indices = []

    node_offset = 0

    for i, g in enumerate(graphs):
        all_x.append(g['x'])
        all_edges.append(g['edge_index'] + node_offset)
        batch_indices.extend([i] * g['num_nodes'])

        if 'clause_features' in g:
            all_clause_features.append(g['clause_features'])

        node_offset += g['num_nodes']

    # Concatenate node features
    x = torch.from_numpy(np.concatenate(all_x, axis=0)).to(device)
    num_nodes = x.size(0)
    num_clauses = len(graphs)

    # Build sparse adjacency with self-loops
    edge_index = torch.from_numpy(np.concatenate(all_edges, axis=1)).to(device)
    self_loops = torch.arange(num_nodes, device=device)
    all_indices = torch.cat([edge_index, torch.stack([self_loops, self_loops])], dim=1)
    all_values = torch.ones(all_indices.size(1), device=device)
    adj = torch.sparse_coo_tensor(all_indices, all_values, (num_nodes, num_nodes)).coalesce()

    # Normalize adjacency: D^{-1/2} A D^{-1/2}
    indices = adj.indices()
    values = adj.values()
    deg = torch.zeros(num_nodes, device=device)
    deg.scatter_add_(0, indices[0], values)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm_values = deg_inv_sqrt[indices[0]] * values * deg_inv_sqrt[indices[1]]
    adj_norm = torch.sparse_coo_tensor(indices, norm_values, (num_nodes, num_nodes)).coalesce()

    # Build sparse pool matrix
    batch = torch.tensor(batch_indices, dtype=torch.long, device=device)
    counts = torch.bincount(batch, minlength=num_clauses).float()
    node_indices = torch.arange(num_nodes, device=device)
    pool_indices = torch.stack([batch, node_indices])
    pool_values = 1.0 / counts[batch].sqrt()
    pool_matrix = torch.sparse_coo_tensor(pool_indices, pool_values, (num_clauses, num_nodes)).coalesce()

    result = {
        'x': x,
        'adj': adj_norm,
        'pool_matrix': pool_matrix,
        'batch': batch,
        'num_graphs': num_clauses,
    }

    if all_clause_features:
        result['clause_features'] = torch.from_numpy(np.stack(all_clause_features, axis=0)).to(device)

    if labels is not None:
        result['y'] = torch.tensor(labels, dtype=torch.float, device=device)

    return result
