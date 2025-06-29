"""Proof and ProofStep implementations using Rust backend."""

from typing import List, Optional, Dict, Any
import proofatlas_rust

from proofatlas.core.logic import Clause
from proofatlas.proofs.state_rust import ProofState
from proofatlas.rules.base import RuleApplication


class ProofStep:
    """
    Python wrapper for Rust ProofStep implementation.
    """
    
    def __init__(self, 
                 state: ProofState,
                 selected_clause: Optional[int] = None,
                 applied_rules: Optional[List[RuleApplication]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """Initialize a ProofStep."""
        self.state = state
        self.selected_clause = selected_clause
        self.applied_rules = applied_rules or []
        self.metadata = metadata or {}
        
        # Create Rust ProofStep
        rust_rules = []
        for rule in self.applied_rules:
            rust_rule = proofatlas_rust.proofs.RuleApplication(
                rule.rule_name,
                rule.parents,
                deleted_clause_indices=rule.deleted_clause_indices
            )
            rust_rules.append(rust_rule)
        
        self._rust_step = proofatlas_rust.proofs.ProofStep(
            state._rust_state,
            selected_clause=selected_clause,
            applied_rules=rust_rules if rust_rules else None,
            metadata=metadata
        )
    
    def __repr__(self) -> str:
        """Return string representation."""
        return f"ProofStep(selected={self.selected_clause}, rules={len(self.applied_rules)})"


class Proof:
    """
    Python wrapper for Rust Proof implementation.
    
    This class provides a Python-friendly interface to the Rust Proof
    while maintaining compatibility with the existing Python API.
    """
    
    def __init__(self, initial_state: ProofState):
        """Initialize a Proof with an initial state."""
        self.steps: List[ProofStep] = []
        
        # Create Rust Proof
        self._rust_proof = proofatlas_rust.proofs.Proof(initial_state._rust_state)
        
        # Add initial step
        initial_step = ProofStep(initial_state)
        self.steps.append(initial_step)
    
    @property
    def initial_state(self) -> ProofState:
        """Return the initial proof state."""
        return self.steps[0].state if self.steps else None
    
    @property
    def final_state(self) -> ProofState:
        """Return the final proof state."""
        return self.steps[-1].state if self.steps else None
    
    @property
    def is_complete(self) -> bool:
        """Check if the proof found a contradiction."""
        return self.final_state.contains_empty_clause if self.final_state else False
    
    def add_step(self, 
                 state: ProofState,
                 selected_clause: Optional[int] = None,
                 rule: Optional[str] = None,
                 parent_clauses: Optional[List[int]] = None,
                 **kwargs):
        """Add a new step to the proof."""
        # Handle metadata
        metadata = {}
        if rule:
            metadata['rule'] = rule
        if parent_clauses:
            metadata['parent_clauses'] = parent_clauses
        metadata.update(kwargs)
        
        # Create step
        step = ProofStep(state, selected_clause=selected_clause, metadata=metadata)
        
        # Don't replace initial step - it has the initial state with no selection
        # Only replace if there's a non-initial step with no selection
        if len(self.steps) > 1 and self.steps[-1].selected_clause is None:
            self.steps.pop()
        
        self.steps.append(step)
        
        # Update Rust proof
        self._rust_proof.add_step(step._rust_step)
    
    def finalize(self, final_state: ProofState):
        """Finalize the proof with a final state."""
        # Remove any existing final step with no selection
        if self.steps and self.steps[-1].selected_clause is None:
            self.steps.pop()
        
        # Add final step
        final_step = ProofStep(final_state)
        self.steps.append(final_step)
        
        # Update Rust proof
        self._rust_proof.finalize(final_state._rust_state)
    
    def get_metadata_history(self, key: str) -> List[Any]:
        """Get the history of a metadata key across all steps."""
        history = []
        for step in self.steps:
            if key in step.metadata:
                history.append(step.metadata[key])
        return history
    
    def __repr__(self) -> str:
        """Return string representation."""
        return f"Proof(steps={len(self.steps)}, complete={self.is_complete})"