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
    device: str = "cpu"
) -> Dict[str, Any]:
    """Convert structured clause to graph tensors.

    The new architecture separates node-level and clause-level features:
    - Node features (3d): type, arity, arg_pos (for GCN encoder)
    - Clause features (3d): age, role, size (for scorer, sinusoidal encoded)

    Args:
        clause: Structured clause dict
        max_age: Maximum age for normalization
        device: PyTorch device

    Returns:
        Dictionary with:
            edge_index: [2, num_edges] edge indices
            x: [num_nodes, 3] node features (type, arity, arg_pos)
            node_types: [num_nodes] node type indices
            num_nodes: int
            num_edges: int
            clause_features: [3] raw clause features (age, role, size)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for graph conversion")

    builder = _GraphBuilder(max_age)
    builder.build_clause(clause)

    return builder.to_tensors(clause, device)


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
        self.nodes: List[np.ndarray] = []  # Node features
        self.node_types: List[int] = []
        self.edges: List[Tuple[int, int]] = []
        self.node_names: List[str] = []

    def build_clause(self, clause: Dict[str, Any]) -> int:
        """Build graph for clause, return root node index."""
        literals = clause.get("literals", [])

        # Create clause node
        clause_idx = self._add_node(
            node_type=TYPE_CLAUSE,
            arity=len(literals),
            arg_pos=0,
            name="clause"
        )

        # Add literals
        for lit_pos, lit in enumerate(literals):
            lit_idx = self._build_literal(lit, arg_pos=lit_pos)
            self.edges.append((clause_idx, lit_idx))

        return clause_idx

    def _build_literal(self, lit: Dict[str, Any], arg_pos: int) -> int:
        """Build graph for literal."""
        polarity = lit.get("polarity", True)
        atom = lit["atom"]
        pred = atom["predicate"]
        args = atom.get("args", [])

        # Create literal node
        lit_idx = self._add_node(
            node_type=TYPE_LITERAL,
            arity=1,  # Literal has one child (predicate)
            arg_pos=arg_pos,
            name=f"{'~' if not polarity else ''}{pred}"
        )

        # Create predicate node
        pred_idx = self._add_node(
            node_type=TYPE_PREDICATE,
            arity=len(args),
            arg_pos=0,
            name=pred
        )
        self.edges.append((lit_idx, pred_idx))

        # Add arguments
        for child_pos, arg in enumerate(args):
            arg_idx = self._build_term(arg, arg_pos=child_pos)
            self.edges.append((pred_idx, arg_idx))

        return lit_idx

    def _build_term(self, term: Dict[str, Any], arg_pos: int) -> int:
        """Build graph for term."""
        term_type = term["type"]

        if term_type == "Variable":
            return self._add_node(
                node_type=TYPE_VARIABLE,
                arity=0,
                arg_pos=arg_pos,
                name=term["name"]
            )
        elif term_type == "Constant":
            return self._add_node(
                node_type=TYPE_CONSTANT,
                arity=0,
                arg_pos=arg_pos,
                name=term["name"]
            )
        elif term_type == "Function":
            args = term.get("args", [])
            func_idx = self._add_node(
                node_type=TYPE_FUNCTION,
                arity=len(args),
                arg_pos=arg_pos,
                name=term["name"]
            )

            for child_pos, child in enumerate(args):
                child_idx = self._build_term(child, arg_pos=child_pos)
                self.edges.append((func_idx, child_idx))

            return func_idx
        else:
            # Unknown term type, treat as constant
            return self._add_node(
                node_type=TYPE_CONSTANT,
                arity=0,
                arg_pos=arg_pos,
                name=f"?{term_type}"
            )

    def _add_node(
        self,
        node_type: int,
        arity: int = 0,
        arg_pos: int = 0,
        name: str = ""
    ) -> int:
        """Add a node with 3-dim feature vector (type, arity, arg_pos)."""
        features = np.zeros(self.NODE_FEATURE_DIM, dtype=np.float32)

        # Feature 0: Node type (raw value)
        features[FEAT_NODE_TYPE] = float(node_type)

        # Feature 1: Arity
        features[FEAT_ARITY] = float(arity)

        # Feature 2: Argument position
        features[FEAT_ARG_POSITION] = float(arg_pos)

        idx = len(self.nodes)
        self.nodes.append(features)
        self.node_types.append(node_type)
        self.node_names.append(name)

        return idx

    def to_tensors(self, clause: Dict[str, Any], device: str = "cpu") -> Dict[str, Any]:
        """Convert to PyTorch tensors.

        Args:
            clause: Original clause dict (for extracting clause-level features)
            device: PyTorch device

        Returns:
            Dictionary with node features, edge index, and clause features
        """
        import torch

        num_nodes = len(self.nodes)
        num_edges = len(self.edges)

        # Node features (3-dim)
        x = torch.tensor(
            np.stack(self.nodes) if self.nodes else np.zeros((0, self.NODE_FEATURE_DIM)),
            dtype=torch.float32,
            device=device
        )

        # Edge index
        if self.edges:
            edge_index = torch.tensor(
                [[e[0] for e in self.edges], [e[1] for e in self.edges]],
                dtype=torch.long,
                device=device
            )
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)

        # Node types
        node_types = torch.tensor(
            self.node_types,
            dtype=torch.uint8,
            device=device
        )

        # Clause-level features (raw values, will be sinusoidal encoded by scorer)
        age = clause.get("age", 0)
        role = clause.get("role", "derived")
        size = len(clause.get("literals", []))

        # Normalize age by max_age for consistency
        normalized_age = float(age) / float(max(self.max_age, 1))
        role_idx = float(ROLE_MAP.get(role, 4))

        clause_features = torch.tensor(
            [normalized_age, role_idx, float(size)],
            dtype=torch.float32,
            device=device
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
    device: str = "cpu"
) -> List[Dict[str, Any]]:
    """Convert list of structured clauses to graph tensors.

    Args:
        clauses: List of structured clause dicts
        max_age: Maximum age for normalization (default: len(clauses))
        device: PyTorch device

    Returns:
        List of graph tensor dicts
    """
    if max_age is None:
        max_age = len(clauses)

    return [clause_to_graph(c, max_age, device) for c in clauses]


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
        graphs: List of graph dicts from clause_to_graph
        labels: Optional list of labels (one per graph)
        device: PyTorch device

    Returns:
        Batched graph dict with batch indices and clause features
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
        all_edges.append(edges + node_offset)
        all_types.append(types)
        batch_indices.extend([i] * num_nodes)

        # Collect clause features if present
        if 'clause_features' in g:
            all_clause_features.append(g['clause_features'])

        node_offset += num_nodes

    result = {
        'x': torch.cat(all_x, dim=0) if all_x else torch.zeros(0, _GraphBuilder.NODE_FEATURE_DIM),
        'edge_index': torch.cat(all_edges, dim=1) if all_edges else torch.zeros(2, 0, dtype=torch.long),
        'node_types': torch.cat(all_types, dim=0) if all_types else torch.zeros(0, dtype=torch.uint8),
        'batch': torch.tensor(batch_indices, dtype=torch.long, device=device),
        'num_graphs': len(graphs),
    }

    # Stack clause features if present
    if all_clause_features:
        result['clause_features'] = torch.stack(all_clause_features, dim=0)

    if labels is not None:
        result['y'] = torch.tensor(labels, dtype=torch.float, device=device)

    return result
