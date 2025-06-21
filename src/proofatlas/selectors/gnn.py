"""GNN-based clause selector."""

from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
from torch.nn import ReLU, Linear, Sequential, Tanh, Sigmoid, Embedding
from torch_geometric.nn import GINConv, GCNConv, global_add_pool, GraphNorm
from pytorch_lightning import LightningModule

from proofatlas.proofs.state import ProofState
from proofatlas.dataformats import get_data_format
from .base import Selector


class SphericalCode(torch.nn.Module):
    """Spherical code embeddings for symbol names.
    
    Based on: J. H. Conway, R. H. Hardin, and N. J. A. Sloane, 
    Packing lines, planes, etc.: packings in Grassmannian spaces, 
    Experiment. Math. 5 (1996), 139–159
    """
    def __init__(self, dim=8, requires_grad=False):
        super().__init__()
        self.dim = dim
        self.embedding = torch.nn.Embedding(33, dim, padding_idx=0)
        self.embedding.weight.requires_grad = requires_grad
        self.embedding.weight[:, :8] = torch.tensor([
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0., -1.,  1.,  1.],
            [ 0.,  0.,  0.,  0.,  0.,  1., -1.,  1.],
            [ 0.,  0.,  0.,  0.,  0.,  1.,  1., -1.],
            [ 0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.],
            [ 0.,  0.,  0., -1.,  1.,  0.,  0.,  1.],
            [ 0.,  0.,  0.,  1., -1.,  0.,  0.,  1.],
            [ 0.,  0.,  0.,  1.,  1.,  0.,  0., -1.],
            [ 0.,  0.,  0.,  1.,  1.,  0.,  0.,  1.],
            [ 0.,  0., -1.,  0.,  1.,  0.,  1.,  0.],
            [ 0.,  0.,  1.,  0., -1.,  0.,  1.,  0.],
            [ 0.,  0.,  1.,  0.,  1.,  0., -1.,  0.],
            [ 0.,  0.,  1.,  0.,  1.,  0.,  1.,  0.],
            [ 0., -1.,  0.,  0.,  1.,  1.,  0.,  0.],
            [ 0.,  1.,  0.,  0., -1.,  1.,  0.,  0.],
            [ 0.,  1.,  0.,  0.,  1., -1.,  0.,  0.],
            [ 0.,  1.,  0.,  0.,  1.,  1.,  0.,  0.],
            [ 0., -1.,  1.,  1.,  0.,  0.,  0.,  0.],
            [ 0.,  1., -1.,  1.,  0.,  0.,  0.,  0.],
            [ 0.,  1.,  1., -1.,  0.,  0.,  0.,  0.],
            [ 0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
            [-1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.],
            [ 1.,  0.,  0., -1.,  0.,  0.,  1.,  0.],
            [ 1.,  0.,  0.,  1.,  0.,  0., -1.,  0.],
            [ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.],
            [-1.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
            [ 1.,  0., -1.,  0.,  0.,  1.,  0.,  0.],
            [ 1.,  0.,  1.,  0.,  0., -1.,  0.,  0.],
            [ 1.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
            [-1.,  1.,  0.,  0.,  0.,  0.,  0.,  1.],
            [ 1., -1.,  0.,  0.,  0.,  0.,  0.,  1.],
            [ 1.,  1.,  0.,  0.,  0.,  0.,  0., -1.],
            [ 1.,  1.,  0.,  0.,  0.,  0.,  0.,  1.]
        ], dtype=torch.float)

    def forward(self, x):
        return self.embedding(x)


def gin_conv(in_channels, out_channels):
    """Create a GIN convolution layer."""
    nn = Sequential(
        Linear(in_channels, 2 * in_channels), 
        GraphNorm(2 * in_channels), 
        ReLU(), 
        Linear(2 * in_channels, out_channels)
    )
    return GINConv(nn)


class GNNModel(LightningModule):
    """Graph Neural Network model for clause selection."""
    
    def __init__(self, num_types, max_arity, layers, dim, conv="GCN", activation="ReLU"):
        super().__init__()
        self.num_types = num_types
        self.max_arity = max_arity
        self.layers = layers
        self.dim = dim
        self.conv_name = conv

        self.type_embedding = Embedding(num_types, dim)
        self.spherical_code = SphericalCode(dim, requires_grad=False)
        self.arity_embedding = Embedding(max_arity + 2, dim, padding_idx=0)
        self.position_embedding = Embedding(max_arity + 2, dim, padding_idx=0)
        
        match activation:
            case "ReLU": self.act = ReLU()
            case "Tanh": self.act = Tanh()
            case "Sigmoid": self.act = Sigmoid()
            case _: raise ValueError(f"Unknown activation {activation}")
        
        match conv:
            case "GCN": self.conv = GCNConv
            case "GIN": self.conv = gin_conv
            case _: raise ValueError(f"Unknown conv type {conv}")
        
        self.norms = torch.nn.ModuleList(
            [GraphNorm(dim) for _ in range(layers)])
        self.conv_layers = torch.nn.ModuleList(
            [self.conv(dim, dim) for _ in range(layers)])
        self.out = Sequential(Linear(dim, dim), self.act, Linear(dim, dim))

    def forward(self, data):
        typ, name, arity, pos, edge_index, batch = (
            data.type, data.name, data.arity, data.pos, data.edge_index, data.batch
        )
        x = (self.type_embedding(typ) + self.spherical_code(name) + 
             self.arity_embedding(arity) + self.position_embedding(pos))
        
        for conv, norm in zip(self.conv_layers, self.norms):
            x = x + norm(conv(x, edge_index))
            x = self.act(x)
        
        return self.out(global_add_pool(x, batch))


class GNNSelector(Selector):
    """Select clauses using a Graph Neural Network model."""
    
    def __init__(self, model: Optional[GNNModel] = None, 
                 data_format: str = 'graph', 
                 device: str = 'cpu', 
                 temperature: float = 1.0,
                 **model_kwargs):
        super().__init__()
        
        # Create default model if none provided
        if model is None:
            model = GNNModel(**model_kwargs)
        
        self.model = model
        self.data_format_name = data_format
        self.data_format = get_data_format(data_format)
        self.device = device
        self.temperature = temperature
        self.model.to(device)
        self.model.eval()
    
    def select(self, proof_state: ProofState) -> Optional[int]:
        """Select clause using GNN predictions."""
        if len(proof_state.unprocessed) == 0:
            return None
        
        scores = self.score_clauses(proof_state)
        
        # Apply temperature and softmax for probabilistic selection
        if self.temperature > 0:
            scores_tensor = torch.tensor(scores) / self.temperature
            probs = torch.softmax(scores_tensor, dim=0)
            selected_idx = torch.multinomial(probs, 1).item()
        else:
            # Greedy selection
            selected_idx = scores.index(max(scores))
        
        return selected_idx
    
    def run(self, proof_state: ProofState) -> Optional[int]:
        """Run the selector (alias for select)."""
        return self.select(proof_state)
    
    def score_clauses(self, proof_state: ProofState) -> List[float]:
        """Score clauses using GNN."""
        scores = []
        
        with torch.no_grad():
            # Encode each unprocessed clause
            for clause in proof_state.unprocessed:
                # Create temporary state with just this clause
                temp_state = ProofState(
                    processed=proof_state.processed,
                    unprocessed=[clause]
                )
                
                # Encode the state
                encoded = self.data_format.encode_state(temp_state)
                
                # Move to device
                if isinstance(encoded, torch.Tensor):
                    encoded = encoded.to(self.device)
                elif isinstance(encoded, dict):
                    encoded = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                              for k, v in encoded.items()}
                
                # Get model prediction
                if isinstance(encoded, dict):
                    score = self.model(**encoded)
                else:
                    score = self.model(encoded)
                
                # Extract scalar score
                if isinstance(score, torch.Tensor):
                    score = score.item()
                
                scores.append(score)
        
        return scores
    
    def train(self, dataset, **kwargs):
        """Train the GNN model on a dataset."""
        # This would implement the training logic
        # For now, it's a placeholder
        raise NotImplementedError("Training logic not yet implemented")
    
    def update(self, selected_idx: int, reward: float, info: Dict[str, Any]):
        """Update selector based on selection outcome."""
        super().update(selected_idx, reward, info)
        # GNN selectors typically update during separate training phase
        # Could implement online learning here if desired
    
    @property
    def name(self) -> str:
        return f"gnn_{self.data_format_name}"