"""FIFO (First-In-First-Out) clause selector."""

from typing import List, Optional

from proofatlas.dataformats.base import ProofState
from .base import ClauseSelector


class FIFOSelector(ClauseSelector):
    """Select clauses in the order they were generated (FIFO)."""
    
    def select(self, proof_state: ProofState) -> Optional[int]:
        """Select the first unprocessed clause."""
        if len(proof_state.unprocessed) > 0:
            return 0
        return None
    
    def score_clauses(self, proof_state: ProofState) -> List[float]:
        """Score clauses by their position (earlier = higher score)."""
        n = len(proof_state.unprocessed)
        if n == 0:
            return []
        
        # Higher scores for earlier clauses
        return [n - i for i in range(n)]
    
    @property
    def name(self) -> str:
        return "fifo"