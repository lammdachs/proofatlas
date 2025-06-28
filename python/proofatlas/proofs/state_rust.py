"""
Python wrappers for Rust-based ProofState.

This module provides thin wrappers around the Rust implementations
to maintain backward compatibility with existing Python code.
"""

try:
    from proofatlas_rust.proofs import ProofState as _RustProofState
except ImportError:
    # Fall back to pure Python implementation if Rust module not available
    from .state import ProofState as _RustProofState


class ProofState(_RustProofState):
    """
    Python-friendly wrapper around Rust ProofState.
    
    Maintains compatibility with existing Python API while using
    Rust implementation for performance.
    """
    
    def __init__(self, processed, unprocessed):
        """Initialize a ProofState with processed and unprocessed clauses."""
        # Convert Python clauses to string representation for Rust
        # In full implementation, would properly convert clause objects
        processed_strs = [str(c) for c in processed]
        unprocessed_strs = [str(c) for c in unprocessed]
        super().__init__(processed_strs, unprocessed_strs)
    
    def add_processed(self, clause):
        """Add a clause to the processed set."""
        # Would need to implement this in Rust bindings
        pass
    
    def add_unprocessed(self, clause):
        """Add a clause to the unprocessed set."""
        # Would need to implement this in Rust bindings
        pass


# For backward compatibility
__all__ = ['ProofState']