"""Base class for file format handlers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from proofatlas.core.fol.logic import Clause, Problem


class FileFormat(ABC):
    """Abstract base class for file format handlers."""
    
    @abstractmethod
    def parse_file(self, file_path: Path, max_size: Optional[int] = None) -> Problem:
        """Parse a file and return a Problem object."""
        pass
    
    @abstractmethod
    def parse_string(self, content: str) -> Problem:
        """Parse a string and return a Problem object."""
        pass
    
    @abstractmethod
    def to_cnf(self, problem: Problem) -> List[Clause]:
        """Convert a Problem to CNF clauses."""
        pass
    
    @abstractmethod
    def write_cnf(self, clauses: List[Clause], output_path: Path) -> None:
        """Write CNF clauses to a file in TPTP format."""
        pass
    
    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        pass