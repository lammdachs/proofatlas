"""
Proof representation and management.
"""

import os

# Check if we should use Rust implementation
USE_RUST = os.environ.get('PROOFATLAS_USE_RUST', 'false').lower() == 'true'

if USE_RUST:
    try:
        # Try to import Rust implementations
        from .proof_rust import Proof, ProofStep
        from .state_rust import ProofState
        print("Using Rust implementation for proofs")
    except ImportError:
        # Fall back to Python implementation
        from .proof import Proof, ProofStep
        from .state import ProofState
        print("Rust module not available, using Python implementation")
else:
    # Use Python implementation
    from .proof import Proof, ProofStep
    from .state import ProofState

from .serialization import (
    ProofJSONEncoder, ProofJSONDecoder,
    proof_to_json, proof_from_json,
    save_proof, load_proof
)

__all__ = [
    'Proof', 'ProofStep', 'ProofState',
    'ProofJSONEncoder', 'ProofJSONDecoder',
    'proof_to_json', 'proof_from_json',
    'save_proof', 'load_proof'
]