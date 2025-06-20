"""Tests for fileformats base module."""

import pytest
from pathlib import Path
from abc import ABC

from proofatlas.fileformats.base import FileFormat
from proofatlas.core.logic import Problem, Clause, Literal, Predicate, Constant


class TestFileFormat:
    """Test FileFormat abstract base class."""
    
    def test_abstract_base_class(self):
        """Test that FileFormat is an abstract base class."""
        with pytest.raises(TypeError):
            FileFormat()
    
    def test_required_methods(self):
        """Test that all required abstract methods are defined."""
        # Create a minimal concrete implementation
        class MinimalFormat(FileFormat):
            def parse_file(self, file_path: Path, **kwargs) -> Problem:
                return Problem()
            
            def parse_string(self, content: str, **kwargs) -> Problem:
                return Problem()
            
            def write_file(self, problem: Problem, file_path: Path, **kwargs) -> None:
                pass
            
            def format_problem(self, problem: Problem, **kwargs) -> str:
                return ""
            
            @property
            def name(self) -> str:
                return "minimal"
            
            @property
            def extensions(self) -> list[str]:
                return [".min"]
        
        # Should be able to instantiate
        fmt = MinimalFormat()
        assert fmt.name == "minimal"
        assert fmt.extensions == [".min"]
    
    def test_missing_methods(self):
        """Test that missing methods cause TypeError."""
        # Missing parse_file
        class MissingParseFile(FileFormat):
            def parse_string(self, content: str, **kwargs) -> Problem:
                return Problem()
            
            def write_file(self, problem: Problem, file_path: Path, **kwargs) -> None:
                pass
            
            def format_problem(self, problem: Problem, **kwargs) -> str:
                return ""
            
            @property
            def name(self) -> str:
                return "test"
            
            @property
            def extensions(self) -> list[str]:
                return [".test"]
        
        with pytest.raises(TypeError):
            MissingParseFile()