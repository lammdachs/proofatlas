"""Base class for file format handlers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from proofatlas.core.logic import Problem


class FileFormat(ABC):
    """Abstract base class for file format handlers.
    
    File format handlers are responsible for:
    1. Parsing files in specific formats (TPTP, SMT-LIB, etc.)
    2. Converting parsed content to Problem objects
    3. Writing Problem objects back to files
    """
    
    @abstractmethod
    def parse_file(self, file_path: Path, **kwargs) -> Problem:
        """Parse a file and return a Problem object.
        
        Args:
            file_path: Path to the file to parse
            **kwargs: Additional format-specific options
            
        Returns:
            Problem object containing the parsed clauses
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file content is invalid
        """
        pass
    
    @abstractmethod
    def parse_string(self, content: str, **kwargs) -> Problem:
        """Parse a string and return a Problem object.
        
        Args:
            content: String content to parse
            **kwargs: Additional format-specific options
            
        Returns:
            Problem object containing the parsed clauses
            
        Raises:
            ValueError: If content is invalid
        """
        pass
    
    @abstractmethod
    def write_file(self, problem: Problem, file_path: Path, **kwargs) -> None:
        """Write a Problem to a file.
        
        Args:
            problem: Problem object to write
            file_path: Path where to write the file
            **kwargs: Additional format-specific options
        """
        pass
    
    @abstractmethod
    def format_problem(self, problem: Problem, **kwargs) -> str:
        """Format a Problem as a string.
        
        Args:
            problem: Problem object to format
            **kwargs: Additional format-specific options
            
        Returns:
            String representation of the problem
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this file format."""
        pass
    
    @property
    @abstractmethod
    def extensions(self) -> list[str]:
        """Return list of file extensions this format handles."""
        pass