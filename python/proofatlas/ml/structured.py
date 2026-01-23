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
        self.edge_src: List[int] = []
        self.edge_dst: List[int] = []

    def build_clause(self, clause: Dict[str, Any]) -> int:
        """Build graph for clause, return root node index."""
        literals = clause.get("literals", [])

        # Create clause node
        clause_idx = self._add_node(TYPE_CLAUSE, len(literals), 0)

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
        lit_idx = self._add_node(TYPE_LITERAL, 1, arg_pos)

        # Create predicate node
        pred_idx = self._add_node(
            node_type=TYPE_PREDICATE,
            arity=len(args),
            arg_pos=0,
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
            return self._add_node(TYPE_VARIABLE, 0, arg_pos)
        elif term_type == "Constant":
            return self._add_node(TYPE_CONSTANT, 0, arg_pos)
        elif term_type == "Function":
            args = term.get("args", [])
            func_idx = self._add_node(TYPE_FUNCTION, len(args), arg_pos)

            for child_pos, child in enumerate(args):
                child_idx = self._build_term(child, arg_pos=child_pos)
                self.edge_src.append(func_idx)
                self.edge_dst.append(child_idx)

            return func_idx
        else:
            # Unknown term type, treat as constant
            return self._add_node(TYPE_CONSTANT, 0, arg_pos)

    def _add_node(self, node_type: int, arity: int = 0, arg_pos: int = 0) -> int:
        """Add a node with 3-dim feature vector (type, arity, arg_pos)."""
        idx = len(self.node_features)
        self.node_features.append((float(node_type), float(arity), float(arg_pos)))
        self.node_types.append(node_type)
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


def clauses_to_batch(
    clauses: List[Dict[str, Any]],
    labels: Optional[List[int]] = None,
    device: str = "cpu",
) -> Dict[str, "torch.Tensor"]:
    """Convert clauses directly to batched torch tensors (optimized path).

    This function pre-allocates arrays and builds all graphs in a single pass,
    avoiding intermediate per-clause allocations. ~25% faster than
    clause_to_graph + batch_graphs.

    Args:
        clauses: List of structured clause dicts
        labels: Optional list of labels (one per clause)
        device: PyTorch device

    Returns:
        Batched graph dict with torch tensors
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")

    import torch

    if not clauses:
        raise ValueError("Cannot batch empty list of clauses")

    # Count total nodes and edges
    total_nodes = 0
    total_edges = 0
    for c in clauses:
        n, e = _count_clause(c)
        total_nodes += n
        total_edges += e

    # Pre-allocate numpy arrays
    x = np.zeros((total_nodes, 3), dtype=np.float32)
    node_types = np.zeros(total_nodes, dtype=np.uint8)
    edge_index = np.zeros((2, total_edges), dtype=np.int64)
    batch = np.zeros(total_nodes, dtype=np.int64)
    clause_features = np.zeros((len(clauses), 3), dtype=np.float32)

    node_idx = 0
    edge_idx = 0
    max_age = len(clauses)

    for clause_i, clause in enumerate(clauses):
        # Clause-level features
        age = clause.get("age", 0)
        role = clause.get("role", "derived")
        size = len(clause.get("literals", []))
        clause_features[clause_i, 0] = float(age) / float(max(max_age, 1))
        clause_features[clause_i, 1] = float(ROLE_MAP.get(role, 4))
        clause_features[clause_i, 2] = float(size)

        # Build graph nodes and edges
        def add_node(t, arity, pos):
            nonlocal node_idx
            idx = node_idx
            x[idx, 0] = t
            x[idx, 1] = arity
            x[idx, 2] = pos
            node_types[idx] = t
            batch[idx] = clause_i
            node_idx += 1
            return idx

        def add_edge(src, dst):
            nonlocal edge_idx
            edge_index[0, edge_idx] = src
            edge_index[1, edge_idx] = dst
            edge_idx += 1

        def build_term(term, pos):
            tt = term["type"]
            if tt == "Variable":
                return add_node(TYPE_VARIABLE, 0, pos)
            elif tt == "Constant":
                return add_node(TYPE_CONSTANT, 0, pos)
            else:  # Function
                args = term.get("args", [])
                idx = add_node(TYPE_FUNCTION, len(args), pos)
                for i, a in enumerate(args):
                    child = build_term(a, i)
                    add_edge(idx, child)
                return idx

        literals = clause.get("literals", [])
        clause_node = add_node(TYPE_CLAUSE, len(literals), 0)

        for lit_pos, lit in enumerate(literals):
            lit_node = add_node(TYPE_LITERAL, 1, lit_pos)
            add_edge(clause_node, lit_node)

            args = lit["atom"].get("args", [])
            pred_node = add_node(TYPE_PREDICATE, len(args), 0)
            add_edge(lit_node, pred_node)

            for i, arg in enumerate(args):
                child = build_term(arg, i)
                add_edge(pred_node, child)

    # Single conversion to torch (from_numpy shares memory, no copy)
    result = {
        "x": torch.from_numpy(x).to(device),
        "edge_index": torch.from_numpy(edge_index).to(device),
        "node_types": torch.from_numpy(node_types).to(device),
        "batch": torch.from_numpy(batch).to(device),
        "clause_features": torch.from_numpy(clause_features).to(device),
        "num_graphs": len(clauses),
    }

    if labels is not None:
        result["y"] = torch.tensor(labels, dtype=torch.float, device=device)

    return result


def _count_clause(clause: Dict[str, Any]) -> Tuple[int, int]:
    """Count nodes and edges in a clause (for pre-allocation)."""
    nodes = 1  # clause node
    edges = 0
    for lit in clause.get("literals", []):
        nodes += 2  # literal + predicate
        edges += 2  # clause->lit, lit->pred
        args = lit["atom"].get("args", [])
        n, e = _count_args(args)
        nodes += n
        edges += e + len(args)  # pred->args edges
    return nodes, edges


def _count_args(args: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Count nodes and edges in term arguments."""
    nodes = 0
    edges = 0
    for arg in args:
        nodes += 1
        if arg["type"] == "Function":
            child_args = arg.get("args", [])
            n, e = _count_args(child_args)
            nodes += n
            edges += e + len(child_args)
    return nodes, edges


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
    """Batch multiple graphs into a single disconnected graph.

    Args:
        graphs: List of graph dicts from clause_to_graph (numpy arrays)
        labels: Optional list of labels (one per graph)
        device: PyTorch device

    Returns:
        Batched graph dict with torch tensors and batch indices
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for batching")

    import torch

    if len(graphs) == 0:
        raise ValueError("Cannot batch empty list of graphs")

    all_x = []
    all_edges = []
    all_types = []
    all_clause_features = []
    batch_indices = []

    node_offset = 0

    for i, g in enumerate(graphs):
        x = g['x']
        edges = g['edge_index']
        types = g['node_types']
        num_nodes = g['num_nodes']

        all_x.append(x)
        # Offset edge indices (numpy addition)
        all_edges.append(edges + node_offset)
        all_types.append(types)
        batch_indices.extend([i] * num_nodes)

        # Collect clause features if present
        if 'clause_features' in g:
            all_clause_features.append(g['clause_features'])

        node_offset += num_nodes

    # Convert numpy arrays to tensors in one shot (much faster than per-graph)
    result = {
        'x': torch.from_numpy(np.concatenate(all_x, axis=0)).to(device) if all_x else torch.zeros(0, _GraphBuilder.NODE_FEATURE_DIM, device=device),
        'edge_index': torch.from_numpy(np.concatenate(all_edges, axis=1)).to(device) if all_edges else torch.zeros(2, 0, dtype=torch.long, device=device),
        'node_types': torch.from_numpy(np.concatenate(all_types, axis=0)).to(device) if all_types else torch.zeros(0, dtype=torch.uint8, device=device),
        'batch': torch.tensor(batch_indices, dtype=torch.long, device=device),
        'num_graphs': len(graphs),
    }

    # Stack clause features if present
    if all_clause_features:
        result['clause_features'] = torch.from_numpy(np.stack(all_clause_features, axis=0)).to(device)

    if labels is not None:
        result['y'] = torch.tensor(labels, dtype=torch.float, device=device)

    return result
