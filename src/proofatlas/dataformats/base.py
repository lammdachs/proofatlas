"""Base classes for data format handlers."""

from abc import ABC, abstractmethod
from typing import List, Any

from proofatlas.core.logic import Clause
from proofatlas.core.proof import ProofState


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
