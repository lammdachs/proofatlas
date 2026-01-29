"""
ProofAtlas: A high-performance theorem prover for first-order logic

This module provides Python bindings to the Rust-based ProofAtlas theorem prover.
"""

from typing import Dict, Any
import os
import sys

# Set up library paths for torch before importing the native extension
def _setup_torch_libs():
    """Add PyTorch library paths to enable tch-rs CUDA support."""
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.isdir(torch_lib):
            # Add to LD_LIBRARY_PATH for current and child processes
            current = os.environ.get('LD_LIBRARY_PATH', '')
            if torch_lib not in current:
                os.environ['LD_LIBRARY_PATH'] = f"{torch_lib}:{current}" if current else torch_lib

            # Preload libtorch to ensure it's available
            import ctypes
            for lib_name in ['libc10.so', 'libtorch_cpu.so', 'libtorch_cuda.so']:
                lib_path = os.path.join(torch_lib, lib_name)
                if os.path.exists(lib_path):
                    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
    except ImportError:
        pass  # torch not installed, skip

_setup_torch_libs()

# Import from the compiled Rust extension
try:
    from .proofatlas import ProofState, ClauseInfo, InferenceResult, ProofStep
except ImportError:
    # Try without the dot for direct module import
    from proofatlas import ProofState, ClauseInfo, InferenceResult, ProofStep

# Version information
__version__ = "0.2.0"
__author__ = "ProofAtlas Contributors"

# Add helper function
def saturate_step(state: ProofState, clause_selection: str = "age") -> Dict[str, Any]:
    """
    Perform one step of the saturation algorithm.
    
    Args:
        state: The proof state
        clause_selection: Strategy for selecting given clause ('age' or 'smallest')
                         Note: 'fifo' is accepted as an alias for 'age', 'size' for 'smallest'
        
    Returns:
        Dictionary with step information:
        - 'given_id': ID of the given clause (or None if saturated)
        - 'new_clauses': List of new clause IDs generated
        - 'saturated': True if no more clauses to process
        - 'proof_found': True if empty clause derived
    """
    given_id = state.select_given_clause(strategy=clause_selection)
    
    if given_id is None:
        return {
            'given_id': None,
            'new_clauses': [],
            'saturated': True,
            'proof_found': state.contains_empty_clause()
        }
    
    # Generate inferences
    inferences = state.generate_inferences(given_id)
    
    # Add non-redundant clauses
    new_clauses = []
    for inf in inferences:
        new_id = state.add_inference(inf)
        if new_id is not None:
            new_clauses.append(new_id)
    
    # Process the given clause
    state.process_clause(given_id)
    
    return {
        'given_id': given_id,
        'new_clauses': new_clauses,
        'saturated': False,
        'proof_found': state.contains_empty_clause()
    }

# Re-export classes and functions
__all__ = ['ProofState', 'ClauseInfo', 'InferenceResult', 'ProofStep', 'saturate_step']