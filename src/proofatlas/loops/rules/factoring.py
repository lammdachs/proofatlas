"""Factoring inference rule."""

from typing import List, Optional, Dict

from proofatlas.core.proof import Rule, Proof
from proofatlas.core.state import ProofState
from proofatlas.core.logic import Clause, Literal


class FactoringRule(Rule):
    """Factoring inference rule - unify literals within a clause."""
    
    @property
    def name(self) -> str:
        return "factoring"
    
    def apply(self, proof: Proof, clause_idx: int) -> Proof:
        """
        Apply factoring to a clause.
        
        Args:
            proof: Current proof
            clause_idx: Index of clause in processed set
            
        Returns:
            Updated proof with factors added
        """
        state = proof.current_state
        clause = state.processed[clause_idx]
        
        # Generate factors
        factors = self._factoring(clause)
        
        # Create new state with factors added to unprocessed
        new_state = ProofState(
            processed=list(state.processed),
            unprocessed=list(state.unprocessed) + factors
        )
        
        # Add step to proof
        proof.add_step(
            state=new_state,
            rule=self,
            selected_clause=clause_idx,
            generated_clauses=factors
        )
        
        return proof
    
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
                        
                        factor = Clause(new_literals)
                        factors.append(factor)
        
        return factors
    
    def _unify(self, atom1, atom2) -> Optional[Dict]:
        """Simple unification algorithm (placeholder)."""
        # Same as in resolution - should be moved to a shared module
        if atom1.predicate != atom2.predicate:
            return None
        
        if len(atom1.args) != len(atom2.args):
            return None
        
        if str(atom1) == str(atom2):
            return {}
        
        return None
    
    def _apply_substitution(self, literal: Literal, substitution: Dict) -> Literal:
        """Apply substitution to a literal (placeholder)."""
        return literal