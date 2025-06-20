"""Proof representation with history of states and selected clauses."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .state import ProofState


@dataclass
class ProofStep:
    """A single step in a proof.
    
    Records the state at this step and which clause was selected for processing.
    Any additional information (rule applied, generated clauses, etc.) goes in metadata.
    """
    state: ProofState
    selected_clause: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Proof:
    """A proof consisting of a sequence of proof steps.
    
    The proof tracks the progression of states and clause selections throughout
    the proof search. The last step always has selected_clause = None, indicating
    the final state with no further selection.
    """
    
    def __init__(self, initial_state: Optional[ProofState] = None):
        """Initialize a proof with an optional initial state."""
        if initial_state is None:
            initial_state = ProofState([], [])
        
        # List of proof steps - last step always has selected_clause = None
        self.steps: List[ProofStep] = [ProofStep(initial_state)]
    
    @property
    def initial_state(self) -> ProofState:
        """Get the initial proof state."""
        return self.steps[0].state if self.steps else None
    
    @property
    def final_state(self) -> ProofState:
        """Get the final proof state (from the last step)."""
        return self.steps[-1].state if self.steps else None
    
    @property
    def length(self) -> int:
        """Number of inference steps in the proof.
        
        This is len(steps) - 1 if the last step has no selection,
        otherwise it's len(steps).
        """
        if not self.steps:
            return 0
        # If last step has no selection, don't count it as an inference step
        if self.steps[-1].selected_clause is None:
            return len(self.steps) - 1
        return len(self.steps)
    
    def add_step(self, state: ProofState, selected_clause: Optional[int] = None, **metadata) -> 'Proof':
        """Add a new step to the proof.
        
        If the last step has selected_clause = None, it will be replaced.
        Otherwise, a new step is appended.
        
        Args:
            state: The proof state at this step
            selected_clause: Index of selected clause in previous state's unprocessed list
            **metadata: Additional information to store with this step
        
        Returns:
            self for method chaining
        """
        new_step = ProofStep(state, selected_clause, metadata)
        
        # If last step has no selection, replace it
        if self.steps and self.steps[-1].selected_clause is None:
            self.steps[-1] = new_step
        else:
            self.steps.append(new_step)
        
        # Ensure last step always has no selection
        if selected_clause is not None and self.steps[-1].selected_clause is not None:
            # Add a final step with no selection
            self.steps.append(ProofStep(state))
        
        return self
    
    def get_selected_clauses(self) -> List[int]:
        """Get list of selected clause indices throughout the proof."""
        return [step.selected_clause for step in self.steps
                if step.selected_clause is not None]
    
    @property
    def is_complete(self) -> bool:
        """Check if proof is complete (found empty clause)."""
        # Check final state for empty clause
        final = self.final_state
        if final:
            for clause in final.all_clauses:
                if len(clause.literals) == 0:
                    return True
        return False
    
    @property
    def is_saturated(self) -> bool:
        """Check if proof search is saturated (no unprocessed clauses)."""
        final = self.final_state
        return final and len(final.unprocessed) == 0
    
    def get_step(self, index: int) -> Optional[ProofStep]:
        """Get a specific proof step by index."""
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None
    
    def get_metadata_history(self, key: str) -> List[Any]:
        """Get the history of a specific metadata key across all steps."""
        return [step.metadata.get(key) for step in self.steps 
                if key in step.metadata]
    
    def finalize(self, final_state: Optional[ProofState] = None) -> 'Proof':
        """Finalize the proof by ensuring the last step has no selection.
        
        Args:
            final_state: Optional final state to use. If None, uses the current last state.
            
        Returns:
            self for method chaining
        """
        if final_state is not None:
            # Add or replace final step with new state
            if self.steps and self.steps[-1].selected_clause is None:
                self.steps[-1] = ProofStep(final_state)
            else:
                self.steps.append(ProofStep(final_state))
        elif self.steps and self.steps[-1].selected_clause is not None:
            # Duplicate last state but with no selection
            self.steps.append(ProofStep(self.steps[-1].state))
        
        return self
    
    def __repr__(self) -> str:
        return f"Proof(steps={len(self.steps)}, complete={self.is_complete})"