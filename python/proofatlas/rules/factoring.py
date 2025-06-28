"""Factoring inference rule."""

from typing import List, Optional, Dict

from .base import Rule, RuleApplication
from proofatlas.proofs.state import ProofState
from proofatlas.core.logic import Clause, Literal
from proofatlas.core.unification import unify_terms, Substitution


class FactoringRule(Rule):
    """Factoring inference rule - unify literals within a clause."""
    
    @property
    def name(self) -> str:
        return "factoring"
    
    def apply(self, state: ProofState, clause_indices: List[int]) -> Optional[RuleApplication]:
        """
        Apply factoring to a clause.
        
        Args:
            state: Current proof state
            clause_indices: List containing one index [clause_idx] in processed set
            
        Returns:
            RuleApplication if successful, None otherwise
        """
        if len(clause_indices) != 1:
            return None
            
        clause_idx = clause_indices[0]
        
        # Check index is valid
        if clause_idx >= len(state.processed):
            return None
            
        clause = state.processed[clause_idx]
        
        # Generate factors
        factors = self._factoring(clause)
        
        if not factors:
            return None
            
        return RuleApplication(
            rule_name=self.name,
            parents=[clause_idx],
            generated_clauses=factors
        )
    
    def _factoring(self, clause: Clause) -> List[Clause]:
        """Apply factoring to a clause."""
        factors = []
        
        # Try to unify pairs of literals with same polarity
        for i in range(len(clause.literals)):
            for j in range(i + 1, len(clause.literals)):
                lit1 = clause.literals[i]
                lit2 = clause.literals[j]
                
                if lit1.polarity == lit2.polarity:
                    mgu = self._unify(lit1.predicate, lit2.predicate)
                    if mgu is not None:
                        # Create factor
                        new_literals = []
                        for k, lit in enumerate(clause.literals):
                            if k != j:  # Skip the unified literal
                                new_lit = self._apply_substitution(lit, mgu)
                                new_literals.append(new_lit)
                        
                        factor = Clause(*new_literals)
                        factors.append(factor)
        
        return factors
    
    def _unify(self, atom1, atom2) -> Optional[Substitution]:
        """Unify two atoms using proper unification algorithm."""
        return unify_terms(atom1, atom2)
    
    def _apply_substitution(self, literal: Literal, substitution: Substitution) -> Literal:
        """Apply substitution to a literal."""
        # Apply substitution to the predicate term
        new_predicate = substitution.apply(literal.predicate)
        return Literal(new_predicate, literal.polarity)