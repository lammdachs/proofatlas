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

# Feature indices (matching Rust GraphBuilder - 8 features)
FEAT_NODE_TYPE = 0
FEAT_ARITY = 1
FEAT_ARG_POSITION = 2
FEAT_DEPTH = 3
FEAT_AGE = 4
FEAT_ROLE = 5
FEAT_POLARITY = 6
FEAT_IS_EQUALITY = 7


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

    Args:
        clause: Structured clause dict
        max_age: Maximum age for normalization
        device: PyTorch device

    Returns:
        Dictionary with edge_index, x, node_types, num_nodes, etc.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for graph conversion")

    builder = _GraphBuilder(max_age)
    builder.build_clause(clause)

    return builder.to_tensors(device)


class _GraphBuilder:
    """Build graph representation from structured clause."""

    # Feature dimension (matching Rust GraphBuilder)
    FEATURE_DIM = 8

    def __init__(self, max_age: int = 1000):
        self.max_age = max_age
        self.nodes: List[np.ndarray] = []  # Node features
        self.node_types: List[int] = []
        self.edges: List[Tuple[int, int]] = []
        self.node_names: List[str] = []

    def build_clause(self, clause: Dict[str, Any]) -> int:
        """Build graph for clause, return root node index."""
        literals = clause.get("literals", [])
        age = clause.get("age", 0)
        role = clause.get("role", "derived")

        # Create clause node
        clause_idx = self._add_node(
            node_type=TYPE_CLAUSE,
            arity=len(literals),
            depth=0,
            age=age,
            role=role,
            name="clause"
        )

        # Add literals
        for lit_pos, lit in enumerate(literals):
            lit_idx = self._build_literal(lit, depth=1, arg_pos=lit_pos)
            self.edges.append((clause_idx, lit_idx))

        return clause_idx

    def _build_literal(self, lit: Dict[str, Any], depth: int, arg_pos: int) -> int:
        """Build graph for literal."""
        polarity = lit.get("polarity", True)
        atom = lit["atom"]
        pred = atom["predicate"]
        args = atom.get("args", [])

        # Create literal node
        lit_idx = self._add_node(
            node_type=TYPE_LITERAL,
            arity=1,  # Literal has one child (predicate)
            depth=depth,
            arg_pos=arg_pos,
            polarity=polarity,
            name=f"{'~' if not polarity else ''}{pred}"
        )

        # Create predicate node
        pred_idx = self._add_node(
            node_type=TYPE_PREDICATE,
            arity=len(args),
            depth=depth + 1,
            arg_pos=0,
            is_equality=(pred == "="),
            name=pred
        )
        self.edges.append((lit_idx, pred_idx))

        # Add arguments
        for arg_pos, arg in enumerate(args):
            arg_idx = self._build_term(arg, depth=depth + 2, arg_pos=arg_pos)
            self.edges.append((pred_idx, arg_idx))

        return lit_idx

    def _build_term(self, term: Dict[str, Any], depth: int, arg_pos: int) -> int:
        """Build graph for term."""
        term_type = term["type"]

        if term_type == "Variable":
            return self._add_node(
                node_type=TYPE_VARIABLE,
                arity=0,
                depth=depth,
                arg_pos=arg_pos,
                name=term["name"]
            )
        elif term_type == "Constant":
            return self._add_node(
                node_type=TYPE_CONSTANT,
                arity=0,
                depth=depth,
                arg_pos=arg_pos,
                name=term["name"]
            )
        elif term_type == "Function":
            args = term.get("args", [])
            func_idx = self._add_node(
                node_type=TYPE_FUNCTION,
                arity=len(args),
                depth=depth,
                arg_pos=arg_pos,
                name=term["name"]
            )

            for child_pos, child in enumerate(args):
                child_idx = self._build_term(child, depth=depth + 1, arg_pos=child_pos)
                self.edges.append((func_idx, child_idx))

            return func_idx
        else:
            # Unknown term type, treat as constant
            return self._add_node(
                node_type=TYPE_CONSTANT,
                arity=0,
                depth=depth,
                arg_pos=arg_pos,
                name=f"?{term_type}"
            )

    def _add_node(
        self,
        node_type: int,
        arity: int = 0,
        depth: int = 0,
        arg_pos: int = 0,
        is_equality: bool = False,
        polarity: bool = True,
        age: int = 0,
        role: str = "derived",
        name: str = ""
    ) -> int:
        """Add a node with feature vector (8 features, matching Rust)."""
        features = np.zeros(self.FEATURE_DIM, dtype=np.float32)

        # Feature 0: Node type (raw value)
        features[FEAT_NODE_TYPE] = float(node_type)

        # Feature 1: Arity
        features[FEAT_ARITY] = float(arity)

        # Feature 2: Argument position
        features[FEAT_ARG_POSITION] = float(arg_pos)

        # Feature 3: Depth
        features[FEAT_DEPTH] = float(depth)

        # Feature 4: Age (normalized)
        features[FEAT_AGE] = float(age) / float(max(self.max_age, 1))

        # Feature 5: Role (encoded as float)
        role_map = {
            "axiom": 0.0,
            "hypothesis": 1.0,
            "definition": 2.0,
            "negated_conjecture": 3.0,
            "derived": 4.0
        }
        features[FEAT_ROLE] = role_map.get(role, 4.0)

        # Feature 6: Polarity (1.0 = positive, 0.0 = negative)
        features[FEAT_POLARITY] = 1.0 if polarity else 0.0

        # Feature 7: Is equality predicate
        features[FEAT_IS_EQUALITY] = 1.0 if is_equality else 0.0

        idx = len(self.nodes)
        self.nodes.append(features)
        self.node_types.append(node_type)
        self.node_names.append(name)

        return idx

    def to_tensors(self, device: str = "cpu") -> Dict[str, Any]:
        """Convert to PyTorch tensors."""
        import torch

        num_nodes = len(self.nodes)
        num_edges = len(self.edges)

        # Node features
        x = torch.tensor(
            np.stack(self.nodes) if self.nodes else np.zeros((0, self.FEATURE_DIM)),
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

        return {
            'edge_index': edge_index,
            'x': x,
            'node_types': node_types,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
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
        Batched graph dict with batch indices
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for batching")

    import torch

    if len(graphs) == 0:
        raise ValueError("Cannot batch empty list of graphs")

    all_x = []
    all_edges = []
    all_types = []
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

        node_offset += num_nodes

    result = {
        'x': torch.cat(all_x, dim=0) if all_x else torch.zeros(0, _GraphBuilder.FEATURE_DIM),
        'edge_index': torch.cat(all_edges, dim=1) if all_edges else torch.zeros(2, 0, dtype=torch.long),
        'node_types': torch.cat(all_types, dim=0) if all_types else torch.zeros(0, dtype=torch.uint8),
        'batch': torch.tensor(batch_indices, dtype=torch.long, device=device),
        'num_graphs': len(graphs),
    }

    if labels is not None:
        result['y'] = torch.tensor(labels, dtype=torch.float, device=device)

    return result
