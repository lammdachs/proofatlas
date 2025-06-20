"""Base classes for proving environments."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set, Any
from enum import Enum

from proofatlas.core.fol.logic import Clause, Literal
from proofatlas.dataformats.base import ProofState


class ActionType(Enum):
    """Types of proof actions."""
    SELECT = "select"
    GENERATE = "generate"
    DELETE = "delete"
    SIMPLIFY = "simplify"


@dataclass
class ProofAction:
    """Represents an action in the proof search."""
    action_type: ActionType
    clause_idx: Optional[int] = None
    clause: Optional[Clause] = None
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ProofTransition:
    """Represents a transition between proof states."""
    old_state: ProofState
    action: ProofAction
    new_state: ProofState
    reward: float = 0.0
    done: bool = False
    info: dict = None
    
    def __post_init__(self):
        if self.info is None:
            self.info = {}


class ProvingEnvironment(ABC):
    """Abstract base class for proving environments."""
    
    def __init__(self, initial_clauses: List[Clause]):
        self.initial_clauses = initial_clauses
        self.state = self.reset()
    
    def reset(self) -> ProofState:
        """Reset environment to initial state."""
        self.state = ProofState(
            processed=[],
            unprocessed=list(self.initial_clauses)
        )
        self._step_count = 0
        self._generated_clauses = set()
        return self.state
    
    @abstractmethod
    def step(self, action: ProofAction) -> ProofTransition:
        """Execute an action and return the transition."""
        pass
    
    @abstractmethod
    def get_valid_actions(self) -> List[ProofAction]:
        """Return list of valid actions in current state."""
        pass
    
    def is_contradiction(self, clause: Clause) -> bool:
        """Check if a clause is a contradiction (empty clause)."""
        return len(clause.literals) == 0
    
    def is_tautology(self, clause: Clause) -> bool:
        """Check if a clause is a tautology."""
        # Check for complementary literals
        for i, lit1 in enumerate(clause.literals):
            for lit2 in clause.literals[i+1:]:
                if self._are_complementary(lit1, lit2):
                    return True
        return False
    
    def _are_complementary(self, lit1: Literal, lit2: Literal) -> bool:
        """Check if two literals are complementary."""
        if lit1.negated != lit2.negated:
            return lit1.atom == lit2.atom
        return False
    
    def subsumes(self, clause1: Clause, clause2: Clause) -> bool:
        """Check if clause1 subsumes clause2."""
        # Every literal in clause1 must be in clause2
        for lit1 in clause1.literals:
            found = False
            for lit2 in clause2.literals:
                if lit1 == lit2:
                    found = True
                    break
            if not found:
                return False
        return True
    
    def is_subsumed(self, clause: Clause, clause_set: List[Clause]) -> bool:
        """Check if clause is subsumed by any clause in clause_set."""
        for other in clause_set:
            if self.subsumes(other, clause):
                return True
        return False
    
    @property
    def done(self) -> bool:
        """Check if proof search is complete."""
        # Check for contradiction
        for clause in self.state.all_clauses:
            if self.is_contradiction(clause):
                return True
        
        # Check for saturation
        return len(self.state.unprocessed) == 0
    
    @property
    def found_proof(self) -> bool:
        """Check if a proof was found."""
        for clause in self.state.all_clauses:
            if self.is_contradiction(clause):
                return True
        return False