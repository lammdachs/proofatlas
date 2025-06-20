"""Base classes for data format handlers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Set, Any, Optional

from proofatlas.core.fol.logic import Clause


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


class DataFormat(ABC):
    """Abstract base class for data format handlers."""
    
    @abstractmethod
    def encode_state(self, proof_state: ProofState) -> Any:
        """Encode a proof state into machine-learnable format."""
        pass
    
    @abstractmethod
    def encode_clauses(self, clauses: List[Clause]) -> Any:
        """Encode a list of clauses into machine-learnable format."""
        pass
    
    @abstractmethod
    def encode_clause(self, clause: Clause) -> Any:
        """Encode a single clause into machine-learnable format."""
        pass
    
    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return the name of this data format."""
        pass