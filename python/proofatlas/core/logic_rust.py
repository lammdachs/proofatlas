"""
Python wrappers for Rust-based logic types.

This module provides thin wrappers around the Rust implementations
to maintain backward compatibility with existing Python code.
"""

try:
    from proofatlas_rust.core import Problem as _RustProblem
except ImportError:
    # Fall back to pure Python implementation if Rust module not available
    from .logic import Problem as _RustProblem


class Problem(_RustProblem):
    """
    Python-friendly wrapper around Rust Problem.
    
    Maintains compatibility with existing Python API while using
    Rust implementation for performance.
    """
    
    def __init__(self, *clauses, conjecture_indices=None):
        """Initialize a Problem with clauses."""
        # For now, delegate to Rust implementation
        # In full implementation, would convert Python clauses to Rust format
        super().__init__(*clauses, conjecture_indices=conjecture_indices)
    
    # Add any Python-specific convenience methods here
    def depth(self):
        """Get maximum depth of terms in the problem."""
        # Would be implemented in Rust eventually
        return 0
    
    def predicate_symbols(self):
        """Get all predicate symbols used in the problem."""
        # Would be implemented in Rust eventually
        return set()
    
    def function_symbols(self):
        """Get all function symbols used in the problem."""
        # Would be implemented in Rust eventually
        return set()
    
    def variables(self):
        """Get all variables used in the problem."""
        # Would be implemented in Rust eventually
        return set()
    
    def terms(self):
        """Get all terms used in the problem."""
        # Would be implemented in Rust eventually
        return set()


# For backward compatibility, expose the wrapped class as the main one
__all__ = ['Problem']