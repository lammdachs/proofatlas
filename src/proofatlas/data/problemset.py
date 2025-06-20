"""Dataset class for theorem proving problems."""

from pathlib import Path
from typing import List, Optional, Any, Dict, Tuple
import glob
from torch.utils.data import Dataset

from proofatlas.fileformats import get_format_handler
from proofatlas.dataformats import get_data_format, ProofState
from proofatlas.core.logic import Problem, Clause
from .config import DatasetConfig, DatasetSplit


class Proofset(Dataset):
    """Dataset for theorem proving problems."""
    
    def __init__(self, config: DatasetConfig, split_name: str = 'train'):
        self.config = config
        self.split_name = split_name
        self.split = config.get_split(split_name)
        if not self.split:
            raise ValueError(f"Split '{split_name}' not found in dataset config")
        
        # Initialize format handlers
        self.file_handler = get_format_handler(config.file_format)
        self.data_format = get_data_format(config.data_format)
        
        # Collect problem files
        self.problem_files = self._collect_problem_files()
        
        # Cache for loaded problems
        self._problem_cache: Dict[int, Problem] = {}
        self._state_cache: Dict[int, ProofState] = {}
    
    def _collect_problem_files(self) -> List[Path]:
        """Collect all problem files for this split."""
        files = []
        
        # Add explicitly listed files
        for file_path in self.split.files:
            full_path = self.config.base_path / file_path
            if full_path.exists():
                files.append(full_path)
        
        # Add files matching patterns
        for pattern in self.split.patterns:
            pattern_path = str(self.config.base_path / pattern)
            matching_files = glob.glob(pattern_path, recursive=True)
            files.extend(Path(f) for f in matching_files)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_files = []
        for f in files:
            if f not in seen:
                seen.add(f)
                unique_files.append(f)
        
        return unique_files
    
    def __len__(self) -> int:
        return len(self.problem_files)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, Any]]:
        """Get a problem and its encoded representation."""
        problem = self.get_problem(idx)
        state = self.get_proof_state(idx)
        encoded = self.data_format.encode_state(state)
        
        metadata = {
            'problem_file': str(self.problem_files[idx]),
            'problem_name': self.problem_files[idx].stem,
            'num_clauses': len(problem.clauses),
            'num_processed': len(state.processed),
            'num_unprocessed': len(state.unprocessed)
        }
        
        return encoded, metadata
    
    def get_problem(self, idx: int) -> Problem:
        """Get a problem by index."""
        if idx not in self._problem_cache:
            problem_file = self.problem_files[idx]
            problem = self.file_handler.parse_file(problem_file)
            self._problem_cache[idx] = problem
        return self._problem_cache[idx]
    
    def get_proof_state(self, idx: int) -> ProofState:
        """Get initial proof state for a problem."""
        if idx not in self._state_cache:
            problem = self.get_problem(idx)
            clauses = self.file_handler.to_cnf(problem)
            # Initial state: all clauses are unprocessed
            state = ProofState(processed=[], unprocessed=clauses)
            self._state_cache[idx] = state
        return self._state_cache[idx]
    
    def get_clauses(self, idx: int) -> List[Clause]:
        """Get CNF clauses for a problem."""
        problem = self.get_problem(idx)
        return self.file_handler.to_cnf(problem)
    
    def clear_cache(self):
        """Clear the problem and state caches."""
        self._problem_cache.clear()
        self._state_cache.clear()