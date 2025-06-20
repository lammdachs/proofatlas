"""Base class for clause selectors."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from proofatlas.dataformats.base import ProofState


class ClauseSelector(ABC):
    """Abstract base class for clause selection strategies."""
    
    def __init__(self):
        """Initialize the selector."""
        self.stats = {
            'selections': 0,
            'total_reward': 0.0
        }
    
    @abstractmethod
    def select(self, proof_state: ProofState) -> Optional[int]:
        """Select a clause from the unprocessed list.
        
        Args:
            proof_state: Current proof state
            
        Returns:
            Index of selected clause in unprocessed list, or None if no selection
        """
        pass
    
    def run(self, proof_state: ProofState) -> Optional[int]:
        """Run the selector (alias for select).
        
        Args:
            proof_state: Current proof state
            
        Returns:
            Index of selected clause in unprocessed list, or None if no selection
        """
        return self.select(proof_state)
    
    def train(self, dataset, **kwargs):
        """Train the selector on a dataset.
        
        Args:
            dataset: Training dataset
            **kwargs: Additional training parameters
        """
        # Default implementation does nothing
        # Subclasses can override for trainable selectors
        pass
    
    def update(self, selected_idx: int, reward: float, info: Dict[str, Any]):
        """Update selector based on selection outcome.
        
        Args:
            selected_idx: Index of selected clause
            reward: Reward signal
            info: Additional information about the outcome
        """
        self.stats['selections'] += 1
        self.stats['total_reward'] += reward
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return name of the selector."""
        pass