"""
Python wrappers for Rust-based Proof and ProofStep.

This module provides thin wrappers around the Rust implementations
to maintain backward compatibility with existing Python code.
"""

try:
    from proofatlas_rust.proofs import (
        Proof as _RustProof,
        ProofStep as _RustProofStep,
        RuleApplication as _RustRuleApplication
    )
except ImportError:
    # Fall back to pure Python implementation if Rust module not available
    from .proof import Proof as _RustProof, ProofStep as _RustProofStep
    from ..rules.base import RuleApplication as _RustRuleApplication


class RuleApplication(_RustRuleApplication):
    """
    Python-friendly wrapper around Rust RuleApplication.
    """
    pass


class ProofStep(_RustProofStep):
    """
    Python-friendly wrapper around Rust ProofStep.
    """
    pass


class Proof(_RustProof):
    """
    Python-friendly wrapper around Rust Proof.
    
    Maintains compatibility with existing Python API while using
    Rust implementation for performance.
    """
    
    def add_step(self, state, selected_clause=None, applied_rules=None, **metadata):
        """
        Add a new step to the proof.
        
        This method maintains the Python API while delegating to Rust.
        """
        # Create ProofStep and add it
        step = ProofStep(state, selected_clause, applied_rules, metadata)
        super().add_step(step)
        return self  # For method chaining
    
    def get_metadata_history(self, key):
        """Get metadata history for a specific key."""
        # Would need to implement this in Rust
        return []


# For backward compatibility
__all__ = ['Proof', 'ProofStep', 'RuleApplication']