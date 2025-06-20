"""Subsumption elimination rule."""

from typing import List

from proofatlas.core.proof import Rule, Proof, ProofState
from proofatlas.core.fol.logic import Clause, Literal


class SubsumptionRule(Rule):
    """Subsumption elimination - remove subsumed clauses."""
    
    @property
    def name(self) -> str:
        return "subsumption"
    
    def apply(self, proof: Proof) -> Proof:
        """
        Apply forward and backward subsumption elimination.
        
        Args:
            proof: Current proof
            
        Returns:
            Updated proof with subsumed clauses removed
        """
        state = proof.current_state
        
        # Remove subsumed clauses from both processed and unprocessed
        new_processed = self._remove_subsumed(state.processed)
        new_unprocessed = self._remove_subsumed(state.unprocessed)
        
        # Check cross-subsumption
        filtered_unprocessed = []
        for clause in new_unprocessed:
            if not self._is_subsumed_by(clause, new_processed):
                filtered_unprocessed.append(clause)
        
        # Create new state
        new_state = ProofState(
            processed=new_processed,
            unprocessed=filtered_unprocessed
        )
        
        # Add step to proof
        removed_count = (len(state.processed) - len(new_processed) + 
                        len(state.unprocessed) - len(filtered_unprocessed))
        
        proof.add_step(
            state=new_state,
            rule=self,
            removed_clauses=removed_count
        )
        
        return proof
    
    def _remove_subsumed(self, clauses: List[Clause]) -> List[Clause]:
        """Remove subsumed clauses from a list."""
        result = []
        for i, clause in enumerate(clauses):
            subsumed = False
            for j, other in enumerate(clauses):
                if i != j and self._subsumes(other, clause):
                    subsumed = True
                    break
            if not subsumed:
                result.append(clause)
        return result
    
    def _is_subsumed_by(self, clause: Clause, clause_set: List[Clause]) -> bool:
        """Check if clause is subsumed by any clause in clause_set."""
        for other in clause_set:
            if self._subsumes(other, clause):
                return True
        return False
    
    def _subsumes(self, clause1: Clause, clause2: Clause) -> bool:
        """Check if clause1 subsumes clause2."""
        # Every literal in clause1 must be in clause2
        for lit1 in clause1.literals:
            found = False
            for lit2 in clause2.literals:
                if self._literals_equal(lit1, lit2):
                    found = True
                    break
            if not found:
                return False
        return True
    
    def _literals_equal(self, lit1: Literal, lit2: Literal) -> bool:
        """Check if two literals are equal."""
        # Simple string comparison for now
        return str(lit1) == str(lit2)