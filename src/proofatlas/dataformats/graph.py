"""Graph-based data format for proof states."""

from typing import List, Any, Dict, Tuple
import torch
import networkx as nx
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_networkx

from proofatlas.core.fol.logic import Clause, Literal, Term, _PREDEFINED
from .base import DataFormat, ProofState


class GraphFormat(DataFormat):
    """Convert proof states to graph representations."""
    
    def __init__(self, max_arity: int = 8):
        self.max_arity = max_arity
        self._init_type_mapping()
    
    def _init_type_mapping(self):
        """Initialize type mapping for nodes."""
        self._type_mapping = {
            'symbol': 0,
            'term': 1,
            'variable': 2,
            'literal': 3,
            'negated_literal': 4,
            'clause': 5,
            'placeholder': 6,
        }
        for i, pred in enumerate(_PREDEFINED):
            self._type_mapping[pred] = i + 7
    
    def encode_state(self, proof_state: ProofState) -> Batch:
        """Encode a proof state into a batched graph."""
        graphs = []
        
        # Encode processed clauses
        for clause in proof_state.processed:
            graph = self._clause_to_graph(clause, is_processed=True)
            graphs.append(graph)
        
        # Encode unprocessed clauses
        for clause in proof_state.unprocessed:
            graph = self._clause_to_graph(clause, is_processed=False)
            graphs.append(graph)
        
        return Batch.from_data_list(graphs)
    
    def encode_clauses(self, clauses: List[Clause]) -> Batch:
        """Encode a list of clauses into a batched graph."""
        graphs = [self._clause_to_graph(clause) for clause in clauses]
        return Batch.from_data_list(graphs)
    
    def encode_clause(self, clause: Clause) -> Data:
        """Encode a single clause into a graph."""
        return self._clause_to_graph(clause)
    
    def _clause_to_graph(self, clause: Clause, is_processed: bool = False) -> Data:
        """Convert a clause to a PyTorch Geometric Data object."""
        G = nx.DiGraph()
        node_features = []
        
        # Add clause node
        clause_id = 0
        G.add_node(clause_id)
        node_features.append({
            'type': self._type_mapping['clause'],
            'is_processed': int(is_processed),
            'arity': len(clause.literals),
            'position': 0
        })
        
        node_counter = 1
        
        # Add literal nodes
        for lit_idx, literal in enumerate(clause.literals):
            lit_node_id = node_counter
            node_counter += 1
            
            # Add literal node
            G.add_node(lit_node_id)
            G.add_edge(clause_id, lit_node_id)
            
            lit_type = 'negated_literal' if literal.negated else 'literal'
            node_features.append({
                'type': self._type_mapping[lit_type],
                'is_processed': int(is_processed),
                'arity': literal.atom.predicate.arity,
                'position': lit_idx
            })
            
            # Add predicate and term nodes
            pred_node_id = node_counter
            node_counter += 1
            G.add_node(pred_node_id)
            G.add_edge(lit_node_id, pred_node_id)
            
            pred = literal.atom.predicate
            pred_type = pred if pred in _PREDEFINED else 'symbol'
            node_features.append({
                'type': self._type_mapping.get(pred_type, self._type_mapping['symbol']),
                'is_processed': int(is_processed),
                'arity': pred.arity,
                'position': 0
            })
            
            # Add term nodes
            for term_idx, term in enumerate(literal.atom.args):
                term_nodes = self._add_term_nodes(G, term, lit_node_id, node_counter, 
                                                 term_idx, is_processed, node_features)
                node_counter += len(term_nodes)
        
        # Convert to PyTorch Geometric format
        data = from_networkx(G)
        
        # Add node features
        data.x = torch.tensor([
            [f['type'], f['is_processed'], f['arity'], f['position']] 
            for f in node_features
        ], dtype=torch.float)
        
        return data
    
    def _add_term_nodes(self, G: nx.DiGraph, term: Term, parent_id: int, 
                       start_id: int, position: int, is_processed: bool,
                       node_features: List[Dict]) -> List[int]:
        """Add term nodes to the graph recursively."""
        added_nodes = []
        
        if hasattr(term, 'symbol'):
            # Function term
            func_node_id = start_id
            G.add_node(func_node_id)
            G.add_edge(parent_id, func_node_id)
            added_nodes.append(func_node_id)
            
            node_features.append({
                'type': self._type_mapping['term'],
                'is_processed': int(is_processed),
                'arity': term.symbol.arity,
                'position': position
            })
            
            # Add argument nodes
            current_id = start_id + 1
            for arg_idx, arg in enumerate(term.args):
                arg_nodes = self._add_term_nodes(G, arg, func_node_id, current_id,
                                                arg_idx, is_processed, node_features)
                added_nodes.extend(arg_nodes)
                current_id += len(arg_nodes)
        else:
            # Variable or constant
            node_id = start_id
            G.add_node(node_id)
            G.add_edge(parent_id, node_id)
            added_nodes.append(node_id)
            
            node_type = 'variable' if hasattr(term, 'is_variable') else 'symbol'
            node_features.append({
                'type': self._type_mapping[node_type],
                'is_processed': int(is_processed),
                'arity': 0,
                'position': position
            })
        
        return added_nodes
    
    @property
    def format_name(self) -> str:
        return "graph"