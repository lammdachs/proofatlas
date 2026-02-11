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
    from .proofatlas import ProofAtlas, ProofStep
except ImportError:
    # Try without the dot for direct module import
    from proofatlas import ProofAtlas, ProofStep

# Version information
__version__ = "0.3.0"
__author__ = "ProofAtlas Contributors"

# Re-export classes
__all__ = ['ProofAtlas', 'ProofStep']
