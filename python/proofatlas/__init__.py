"""
ProofAtlas: A high-performance theorem prover for first-order logic

This module provides Python bindings to the Rust-based ProofAtlas theorem prover.

The Rust extension (and libtorch) is lazy-loaded on first access to ProofAtlas
or ProofStep, allowing `import proofatlas.ml` to work without loading libtorch.
"""

from typing import Dict, Any
import os
import sys

# Version information
__version__ = "0.3.0"
__author__ = "ProofAtlas Contributors"

__all__ = ['ProofAtlas', 'ProofStep']


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
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            cuda_disabled = cuda_visible is not None and cuda_visible.strip() == ''
            for lib_name in ['libc10.so', 'libtorch_cpu.so', 'libtorch_cuda.so']:
                if cuda_disabled and 'cuda' in lib_name:
                    continue  # Skip CUDA libs when explicitly disabled
                lib_path = os.path.join(torch_lib, lib_name)
                if os.path.exists(lib_path):
                    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
    except ImportError:
        pass  # torch not installed, skip


# Lazy imports — only load Rust extension when needed
_ProofAtlas = None
_ProofStep = None


def __getattr__(name):
    global _ProofAtlas, _ProofStep
    if name in ("ProofAtlas", "ProofStep"):
        if _ProofAtlas is None:
            _setup_torch_libs()
            try:
                from .proofatlas import ProofAtlas as _PA, ProofStep as _PS
            except ImportError:
                from proofatlas import ProofAtlas as _PA, ProofStep as _PS
            _ProofAtlas, _ProofStep = _PA, _PS
        return _ProofAtlas if name == "ProofAtlas" else _ProofStep
    raise AttributeError(f"module 'proofatlas' has no attribute {name!r}")
