"""Binary resolution inference rule."""

from typing import List, Optional, Dict, Tuple

from proofatlas.core.proof import Rule, Proof, ProofState
from proofatlas.core.fol.logic import Clause, Literal


class ResolutionRule(Rule):
    """Binary resolution inference rule."""
    
    @property
    def name(self) -> str:
        return "resolution"
    
    def apply(self, proof: Proof, clause1_idx: int, clause2_idx: int) -> Proof:
        """
        Apply binary resolution between two clauses.
        
        Args:
            proof: Current proof
            clause1_idx: Index of first clause in processed set
            clause2_idx: Index of second clause in processed set
            
        Returns:
            Updated proof with resolvents added
        """
        state = proof.current_state
        clause1 = state.processed[clause1_idx]
        clause2 = state.processed[clause2_idx]
        
        # Generate resolvents
        resolvents = self._binary_resolution(clause1, clause2)
        
        # Create new state with resolvents added to unprocessed
        new_state = ProofState(
            processed=list(state.processed),
            unprocessed=list(state.unprocessed) + resolvents
        )
        
        # Add step to proof
        proof.add_step(
            state=new_state,
            rule=self,
            generated_clauses=resolvents,
            clause1_idx=clause1_idx,
            clause2_idx=clause2_idx
        )
        
        return proof
    
    def _binary_resolution(self, clause1: Clause, clause2: Clause) -> List[Clause]:
        """Apply binary resolution between two clauses."""
        resolvents = []
        
        for i, lit1 in enumerate(clause1.literals):
            for j, lit2 in enumerate(clause2.literals):
                if lit1.negated != lit2.negated:
                    # Try to unify the atoms
                    mgu = self._unify(lit1.atom, lit2.atom)
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
                            resolvent = Clause(unique_literals)
                            resolvents.append(resolvent)
        
        return resolvents
    
    def _unify(self, atom1, atom2) -> Optional[Dict]:
        """Simple unification algorithm (placeholder - needs proper implementation)."""
        # This is a simplified version - a real implementation would need
        # occurs check and proper term unification
        if atom1.predicate != atom2.predicate:
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