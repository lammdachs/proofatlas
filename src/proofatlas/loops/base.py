"""Base class for given clause loops."""

from abc import ABC, abstractmethod
from typing import List, Optional

from proofatlas.proofs import Proof
from proofatlas.proofs.state import ProofState
from proofatlas.core.logic import Clause, Literal


class Loop(ABC):
    """Abstract base class for given clause loops."""
    
    @abstractmethod
    def step(self, proof: Proof, given_clause: int) -> Proof:
        """
        Execute one step of the given clause loop.
        
        Args:
            proof: Current proof state
            given_clause: Index of the clause to process from unprocessed
            
        Returns:
            Updated proof with new state
        """
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
        if lit1.polarity != lit2.polarity:
            return lit1.predicate == lit2.predicate
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