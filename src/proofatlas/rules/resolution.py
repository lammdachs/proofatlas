"""Binary resolution inference rule."""

from typing import List, Optional, Dict, Tuple

from .base import Rule, RuleApplication
from proofatlas.proofs.state import ProofState
from proofatlas.core.logic import Clause, Literal


class ResolutionRule(Rule):
    """Binary resolution inference rule."""
    
    @property
    def name(self) -> str:
        return "resolution"
    
    def apply(self, state: ProofState, clause_indices: List[int]) -> Optional[RuleApplication]:
        """
        Apply binary resolution between two clauses.
        
        Args:
            state: Current proof state
            clause_indices: List of two indices [clause1_idx, clause2_idx] in processed set
            
        Returns:
            RuleApplication if successful, None otherwise
        """
        if len(clause_indices) != 2:
            return None
            
        clause1_idx, clause2_idx = clause_indices
        
        # Check indices are valid
        if clause1_idx >= len(state.processed) or clause2_idx >= len(state.processed):
            return None
            
        clause1 = state.processed[clause1_idx]
        clause2 = state.processed[clause2_idx]
        
        # Generate resolvents
        resolvents = self._binary_resolution(clause1, clause2)
        
        if not resolvents:
            return None
            
        return RuleApplication(
            rule_name=self.name,
            parents=[clause1_idx, clause2_idx],
            generated_clauses=resolvents
        )
    
    def _binary_resolution(self, clause1: Clause, clause2: Clause) -> List[Clause]:
        """Apply binary resolution between two clauses."""
        resolvents = []
        
        for i, lit1 in enumerate(clause1.literals):
            for j, lit2 in enumerate(clause2.literals):
                if lit1.polarity != lit2.polarity:
                    # Try to unify the predicates
                    mgu = self._unify(lit1.predicate, lit2.predicate)
                    if mgu is not None:
                        # Create resolvent
                        new_literals = []
                        
                        # Add literals from clause1 (except resolved)
                        for k, lit in enumerate(clause1.literals):
                            if k != i:
                                new_lit = self._apply_substitution(lit, mgu)
                                new_literals.append(new_lit)
                        
                        # Add literals from clause2 (except resolved)
                        for k, lit in enumerate(clause2.literals):
                            if k != j:
                                new_lit = self._apply_substitution(lit, mgu)
                                new_literals.append(new_lit)
                        
                        # Remove duplicates
                        unique_literals = []
                        seen = set()
                        for lit in new_literals:
                            lit_str = str(lit)
                            if lit_str not in seen:
                                seen.add(lit_str)
                                unique_literals.append(lit)
                        
                        if unique_literals or len(new_literals) == 0:  # Allow empty clause
                            resolvent = Clause(*unique_literals)
                            resolvents.append(resolvent)
        
        return resolvents
    
    def _unify(self, atom1, atom2) -> Optional[Dict]:
        """Simple unification algorithm (placeholder - needs proper implementation)."""
        # This is a simplified version - a real implementation would need
        # occurs check and proper term unification
        if atom1.symbol != atom2.symbol:
            return None
        
        if len(atom1.args) != len(atom2.args):
            return None
        
        # For now, only unify if atoms are identical
        # A proper implementation would compute MGU
        if str(atom1) == str(atom2):
            return {}
        
        return None
    
    def _apply_substitution(self, literal: Literal, substitution: Dict) -> Literal:
        """Apply substitution to a literal (placeholder)."""
        # This would need proper implementation
        return literal