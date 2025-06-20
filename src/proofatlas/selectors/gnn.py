"""Neural network-based clause selector."""

from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn

from proofatlas.dataformats.base import ProofState
from proofatlas.dataformats import get_data_format
from .base import ClauseSelector


class NeuralSelector(ClauseSelector):
    """Select clauses using a neural network model."""
    
    def __init__(self, model: nn.Module, data_format: str = 'graph', 
                 device: str = 'cpu', temperature: float = 1.0):
        super().__init__()
        self.model = model
        self.data_format_name = data_format
        self.data_format = get_data_format(data_format)
        self.device = device
        self.temperature = temperature
        self.model.to(device)
        self.model.eval()
    
    def select(self, proof_state: ProofState) -> Optional[int]:
        """Select clause using neural network predictions."""
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
    
    def score_clauses(self, proof_state: ProofState) -> List[float]:
        """Score clauses using neural network."""
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
    
    def update(self, selected_idx: int, reward: float, info: Dict[str, Any]):
        """Update selector based on selection outcome."""
        super().update(selected_idx, reward, info)
        # Neural selectors typically update during separate training phase
        # Could implement online learning here if desired
    
    @property
    def name(self) -> str:
        return f"neural_{self.data_format_name}"