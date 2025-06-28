"""Base interface for inference rules."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

from proofatlas.core.logic import Clause
from proofatlas.proofs.state import ProofState


@dataclass
class RuleApplication:
    """Result of applying an inference rule."""
    rule_name: str
    parents: List[int]  # Indices of parent clauses
    generated_clauses: List[Clause] = field(default_factory=list)
    deleted_clause_indices: List[int] = field(default_factory=list)  # For deletion rules like subsumption
    metadata: Dict[str, Any] = field(default_factory=dict)


class Rule(ABC):
    """Abstract base class for inference rules."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the inference rule."""
        pass
    
    @abstractmethod
    def apply(self, state: ProofState, clause_indices: List[int]) -> Optional[RuleApplication]:
        """
        Apply the rule to the given clauses.
        
        Args:
            state: Current proof state
            clause_indices: Indices of clauses to apply the rule to
                           (typically indices into processed clauses, but rule-specific)
            
        Returns:
            RuleApplication if the rule was successfully applied, None otherwise
        """
        pass
    
    def is_applicable(self, state: ProofState, clause_indices: List[int]) -> bool:
        """
        Check if the rule can be applied to the given clauses.
        
        Default implementation tries to apply and checks if result is not None.
        Subclasses can override for more efficient checking.
        """
        return self.apply(state, clause_indices) is not None