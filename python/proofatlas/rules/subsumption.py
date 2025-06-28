"""Subsumption elimination rule."""

from typing import List, Optional

from .base import Rule, RuleApplication
from proofatlas.proofs.state import ProofState
from proofatlas.core.logic import Clause, Literal


class SubsumptionRule(Rule):
    """Subsumption elimination - remove subsumed clauses."""
    
    @property
    def name(self) -> str:
        return "subsumption"
    
    def apply(self, state: ProofState, clause_indices: List[int]) -> Optional[RuleApplication]:
        """
        Apply subsumption checking. 
        
        This rule is special - it doesn't use clause_indices but checks
        all clauses for subsumption relationships.
        
        Args:
            state: Current proof state
            clause_indices: Ignored for this rule
            
        Returns:
            RuleApplication with deleted indices if any subsumptions found, None otherwise
        """
        deleted_indices = []
        
        # Check for subsumption within processed clauses
        processed_to_delete = set()
        for i, clause1 in enumerate(state.processed):
            for j, clause2 in enumerate(state.processed):
                if i != j and j not in processed_to_delete:
                    if self._subsumes(clause1, clause2):
                        processed_to_delete.add(j)
        
        # Check for subsumption within unprocessed clauses
        unprocessed_to_delete = set()
        for i, clause1 in enumerate(state.unprocessed):
            for j, clause2 in enumerate(state.unprocessed):
                if i != j and j not in unprocessed_to_delete:
                    if self._subsumes(clause1, clause2):
                        unprocessed_to_delete.add(j)
        
        # Check if processed clauses subsume unprocessed ones
        for clause1 in state.processed:
            for j, clause2 in enumerate(state.unprocessed):
                if j not in unprocessed_to_delete:
                    if self._subsumes(clause1, clause2):
                        unprocessed_to_delete.add(j)
        
        if not processed_to_delete and not unprocessed_to_delete:
            return None
            
        # Convert to indices relative to full clause list
        # (negative indices for unprocessed clauses)
        deleted_indices.extend(processed_to_delete)
        deleted_indices.extend([-idx-1 for idx in unprocessed_to_delete])
        
        return RuleApplication(
            rule_name=self.name,
            parents=[],  # No parent clauses for subsumption
            deleted_clause_indices=deleted_indices,
            metadata={
                "processed_deleted": len(processed_to_delete),
                "unprocessed_deleted": len(unprocessed_to_delete)
            }
        )
    
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