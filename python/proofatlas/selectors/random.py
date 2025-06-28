"""Random clause selector."""

import random
from typing import Optional

from proofatlas.proofs.state import ProofState
from .base import Selector


class RandomSelector(Selector):
    """Select clauses uniformly at random.
    
    This selector randomly chooses from the available unprocessed clauses
    with equal probability. It's useful as a baseline and for exploration
    during proof search.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the random selector.
        
        Args:
            seed: Random seed for reproducibility. If None, uses system time.
        """
        super().__init__()
        self.seed = seed
        # Create a per-instance Random object to avoid global state issues
        self._random = random.Random(seed)
    
    def select(self, proof_state: ProofState) -> Optional[int]:
        """Select a random clause from the unprocessed list.
        
        Args:
            proof_state: Current proof state
            
        Returns:
            Random index from unprocessed list, or None if empty
        """
        if not proof_state.unprocessed:
            return None
        
        # Select random index using instance-specific Random
        selected_idx = self._random.randint(0, len(proof_state.unprocessed) - 1)
        
        return selected_idx
    
    @property
    def name(self) -> str:
        """Return name of the selector."""
        return "random"