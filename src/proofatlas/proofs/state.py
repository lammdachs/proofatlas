"""Proof state representation."""

from dataclasses import dataclass
from typing import List

from proofatlas.core.logic import Clause


@dataclass
class ProofState:
    """Represents a proof state with processed and unprocessed clauses."""
    processed: List[Clause]
    unprocessed: List[Clause]
    
    def __post_init__(self):
        self.processed = list(self.processed)
        self.unprocessed = list(self.unprocessed)
    
    @property
    def all_clauses(self) -> List[Clause]:
        """Return all clauses (processed + unprocessed)."""
        return self.processed + self.unprocessed
    
    def add_processed(self, clause: Clause):
        """Add a clause to the processed set."""
        self.processed.append(clause)
    
    def add_unprocessed(self, clause: Clause):
        """Add a clause to the unprocessed set."""
        self.unprocessed.append(clause)
    
    def move_to_processed(self, clause: Clause):
        """Move a clause from unprocessed to processed."""
        if clause in self.unprocessed:
            self.unprocessed.remove(clause)
            self.processed.append(clause)