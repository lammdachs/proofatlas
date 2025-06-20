"""Proof representation with history of states and applied rules."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Tuple

from .state import ProofState
from proofatlas.core.fol.logic import Clause


class Rule(ABC):
    """Abstract base class for inference rules."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the rule."""
        pass
    
    @abstractmethod
    def apply(self, proof: 'Proof', *args) -> 'Proof':
        """Apply the rule to create a new proof state."""
        pass


@dataclass
class ProofStep:
    """A single step in a proof."""
    state: ProofState
    rule: Optional[Rule] = None
    selected_clause: Optional[int] = None
    generated_clauses: List[Clause] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Proof:
    """A proof consisting of a list of ProofStates and applied rules."""
    
    def __init__(self, initial_state: Optional[ProofState] = None):
        if initial_state is None:
            initial_state = ProofState([], [])
        self.steps: List[ProofStep] = [ProofStep(initial_state)]
    
    @property
    def current_state(self) -> ProofState:
        """Get the current (latest) proof state."""
        return self.steps[-1].state
    
    @property
    def initial_state(self) -> ProofState:
        """Get the initial proof state."""
        return self.steps[0].state
    
    def apply_rule(self, rule: Rule, *args) -> 'Proof':
        """Apply a rule to create a new proof with an additional step."""
        # Rules implement their own logic and call add_step
        return rule.apply(self, *args)
    
    def add_step(self, state: ProofState, rule: Optional[Rule] = None, 
                 selected_clause: Optional[int] = None, 
                 generated_clauses: Optional[List[Clause]] = None,
                 **metadata) -> 'Proof':
        """Add a new step to the proof (used by rules)."""
        if generated_clauses is None:
            generated_clauses = []
        
        step = ProofStep(
            state=state,
            rule=rule,
            selected_clause=selected_clause,
            generated_clauses=generated_clauses,
            metadata=metadata
        )
        self.steps.append(step)
        return self
    
    @property
    def length(self) -> int:
        """Number of steps in the proof (excluding initial state)."""
        return len(self.steps) - 1
    
    @property
    def is_complete(self) -> bool:
        """Check if proof is complete (found contradiction)."""
        current = self.current_state
        # Check for empty clause (contradiction)
        for clause in current.all_clauses:
            if len(clause.literals) == 0:
                return True
        return False
    
    @property
    def is_saturated(self) -> bool:
        """Check if proof search is saturated (no unprocessed clauses)."""
        return len(self.current_state.unprocessed) == 0
    
    def get_selected_clauses(self) -> List[int]:
        """Get list of selected clause indices throughout the proof."""
        return [step.selected_clause for step in self.steps[1:] 
                if step.selected_clause is not None]
    
    def get_applied_rules(self) -> List[Rule]:
        """Get list of rules applied throughout the proof."""
        return [step.rule for step in self.steps[1:] 
                if step.rule is not None]