"""Given clause algorithm implementation."""

from typing import List, Set, Optional, Tuple
from itertools import combinations

from proofatlas.core.fol.logic import Clause, Literal, Term, Predicate, Variable
from proofatlas.dataformats.base import ProofState
from .base import ProvingEnvironment, ProofAction, ProofTransition, ActionType


class GivenClauseEnvironment(ProvingEnvironment):
    """Environment implementing the given clause algorithm."""
    
    def __init__(self, initial_clauses: List[Clause], 
                 max_clause_size: int = 100,
                 forward_simplify: bool = True,
                 backward_simplify: bool = True):
        self.max_clause_size = max_clause_size
        self.forward_simplify = forward_simplify
        self.backward_simplify = backward_simplify
        super().__init__(initial_clauses)
    
    def step(self, action: ProofAction) -> ProofTransition:
        """Execute an action in the given clause algorithm."""
        old_state = ProofState(
            processed=list(self.state.processed),
            unprocessed=list(self.state.unprocessed)
        )
        
        if action.action_type == ActionType.SELECT:
            # Select clause from unprocessed
            if action.clause_idx is None or action.clause_idx >= len(self.state.unprocessed):
                raise ValueError("Invalid clause index for selection")
            
            given_clause = self.state.unprocessed[action.clause_idx]
            self.state.move_to_processed(given_clause)
            
            # Generate new clauses
            new_clauses = self._generate_clauses(given_clause)
            
            # Forward simplification
            if self.forward_simplify:
                new_clauses = self._forward_simplify(new_clauses)
            
            # Backward simplification
            if self.backward_simplify:
                self._backward_simplify(new_clauses)
            
            # Add new clauses to unprocessed
            for clause in new_clauses:
                if not self._is_redundant(clause):
                    self.state.add_unprocessed(clause)
            
            # Calculate reward
            reward = self._calculate_reward(old_state, self.state, given_clause, new_clauses)
            
            # Create transition
            transition = ProofTransition(
                old_state=old_state,
                action=action,
                new_state=self.state,
                reward=reward,
                done=self.done,
                info={
                    'given_clause': given_clause,
                    'new_clauses': new_clauses,
                    'found_proof': self.found_proof
                }
            )
            
            self._step_count += 1
            return transition
        
        else:
            raise ValueError(f"Unsupported action type: {action.action_type}")
    
    def get_valid_actions(self) -> List[ProofAction]:
        """Get all valid selection actions."""
        actions = []
        for i in range(len(self.state.unprocessed)):
            actions.append(ProofAction(
                action_type=ActionType.SELECT,
                clause_idx=i,
                clause=self.state.unprocessed[i]
            ))
        return actions
    
    def _generate_clauses(self, given_clause: Clause) -> List[Clause]:
        """Generate new clauses using inference rules."""
        new_clauses = []
        
        # Binary resolution with processed clauses
        for processed_clause in self.state.processed:
            resolvents = self._binary_resolution(given_clause, processed_clause)
            new_clauses.extend(resolvents)
        
        # Factoring
        factors = self._factoring(given_clause)
        new_clauses.extend(factors)
        
        # Filter by size
        new_clauses = [c for c in new_clauses if len(c.literals) <= self.max_clause_size]
        
        return new_clauses
    
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
    
    def _factoring(self, clause: Clause) -> List[Clause]:
        """Apply factoring to a clause."""
        factors = []
        
        # Try to unify pairs of literals with same polarity
        for i in range(len(clause.literals)):
            for j in range(i + 1, len(clause.literals)):
                lit1 = clause.literals[i]
                lit2 = clause.literals[j]
                
                if lit1.negated == lit2.negated:
                    mgu = self._unify(lit1.atom, lit2.atom)
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
    
    def _unify(self, atom1, atom2) -> Optional[dict]:
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
    
    def _apply_substitution(self, literal: Literal, substitution: dict) -> Literal:
        """Apply substitution to a literal (placeholder)."""
        # This would need proper implementation
        return literal
    
    def _forward_simplify(self, clauses: List[Clause]) -> List[Clause]:
        """Apply forward simplification to new clauses."""
        simplified = []
        
        for clause in clauses:
            # Remove tautologies
            if self.is_tautology(clause):
                continue
            
            # Check subsumption by processed clauses
            if self.is_subsumed(clause, self.state.processed):
                continue
            
            simplified.append(clause)
        
        return simplified
    
    def _backward_simplify(self, new_clauses: List[Clause]):
        """Apply backward simplification with new clauses."""
        # Remove processed clauses subsumed by new clauses
        to_remove = []
        for i, proc_clause in enumerate(self.state.processed):
            for new_clause in new_clauses:
                if self.subsumes(new_clause, proc_clause):
                    to_remove.append(i)
                    break
        
        # Remove in reverse order to maintain indices
        for i in reversed(to_remove):
            self.state.processed.pop(i)
    
    def _is_redundant(self, clause: Clause) -> bool:
        """Check if a clause is redundant."""
        # Check if already in processed or unprocessed
        for existing in self.state.all_clauses:
            if str(clause) == str(existing):
                return True
        
        return False
    
    def _calculate_reward(self, old_state: ProofState, new_state: ProofState,
                         given_clause: Clause, new_clauses: List[Clause]) -> float:
        """Calculate reward for the transition."""
        # Found proof
        if self.found_proof:
            return 100.0
        
        # Penalty for no progress
        if len(new_clauses) == 0:
            return -1.0
        
        # Reward for generating short clauses
        avg_length = sum(len(c.literals) for c in new_clauses) / max(len(new_clauses), 1)
        length_reward = 1.0 / (1.0 + avg_length)
        
        # Penalty for large unprocessed set
        size_penalty = -0.01 * len(new_state.unprocessed)
        
        return length_reward + size_penalty