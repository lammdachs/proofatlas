"""ProofState implementation using Rust backend."""

from typing import List, Optional
import proofatlas_rust

from proofatlas.core.logic import Clause


class ProofState:
    """
    Python wrapper for Rust ProofState implementation.
    
    This class provides a Python-friendly interface to the Rust ProofState
    while maintaining compatibility with the existing Python API.
    """
    
    def __init__(self, processed: List[Clause], unprocessed: List[Clause]):
        """Initialize a ProofState with processed and unprocessed clauses."""
        # Store Python clauses
        self._processed = list(processed)
        self._unprocessed = list(unprocessed)
        
        # Create string representations for Rust
        processed_strs = [str(clause) for clause in processed]
        unprocessed_strs = [str(clause) for clause in unprocessed]
        
        # Create Rust ProofState
        self._rust_state = proofatlas_rust.proofs.ProofState(processed_strs, unprocessed_strs)
    
    @property
    def processed(self) -> List[Clause]:
        """Return processed clauses."""
        return self._processed
    
    @property
    def unprocessed(self) -> List[Clause]:
        """Return unprocessed clauses."""
        return self._unprocessed
    
    @property
    def all_clauses(self) -> List[Clause]:
        """Return all clauses (processed + unprocessed)."""
        return self._processed + self._unprocessed
    
    @property
    def contains_empty_clause(self) -> bool:
        """Check if the state contains the empty clause."""
        return any(len(clause.literals) == 0 for clause in self.all_clauses)
    
    def add_processed(self, clause: Clause):
        """Add a clause to the processed set."""
        self._processed.append(clause)
        # Update Rust state
        self._sync_to_rust()
    
    def add_unprocessed(self, clause: Clause):
        """Add a clause to the unprocessed set."""
        self._unprocessed.append(clause)
        # Update Rust state
        self._sync_to_rust()
    
    def move_to_processed(self, clause: Clause):
        """Move a clause from unprocessed to processed."""
        if clause in self._unprocessed:
            self._unprocessed.remove(clause)
            self._processed.append(clause)
            # Update Rust state
            self._sync_to_rust()
    
    def get_next_unprocessed_idx(self) -> Optional[int]:
        """Get the index of the next unprocessed clause, or None if empty."""
        return 0 if self._unprocessed else None
    
    def _sync_to_rust(self):
        """Synchronize Python state to Rust state."""
        processed_strs = [str(clause) for clause in self._processed]
        unprocessed_strs = [str(clause) for clause in self._unprocessed]
        self._rust_state = proofatlas_rust.proofs.ProofState(processed_strs, unprocessed_strs)
    
    def __repr__(self) -> str:
        """Return string representation."""
        return f"ProofState(processed={len(self._processed)}, unprocessed={len(self._unprocessed)})"