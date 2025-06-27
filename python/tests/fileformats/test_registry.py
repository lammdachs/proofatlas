"""Tests for fileformats registry module."""

import pytest
from pathlib import Path

from proofatlas.fileformats.registry import FileFormatRegistry, get_format_handler
from proofatlas.fileformats.base import FileFormat
from proofatlas.fileformats.tptp import TPTPFormat
from proofatlas.core.logic import Problem


class DummyFormat(FileFormat):
    """Dummy format for testing."""
    
    def parse_file(self, file_path: Path, **kwargs) -> Problem:
        return Problem()
    
    def parse_string(self, content: str, **kwargs) -> Problem:
        return Problem()
    
    def write_file(self, problem: Problem, file_path: Path, **kwargs) -> None:
        pass
    
    def format_problem(self, problem: Problem, **kwargs) -> str:
        return "dummy"
    
    @property
    def name(self) -> str:
        return "dummy"
    
    @property
    def extensions(self) -> list[str]:
        return [".dum", ".dummy"]


class TestFileFormatRegistry:
    """Test FileFormatRegistry functionality."""
    
    def test_create_registry(self):
        """Test creating a new registry."""
        registry = FileFormatRegistry()
        assert isinstance(registry, FileFormatRegistry)
    
    def test_default_formats(self):
        """Test that default formats are registered."""
        registry = FileFormatRegistry()
        
        # TPTP should be registered by default
        handler = registry.get_handler('tptp')
        assert isinstance(handler, TPTPFormat)
    
    def test_register_format(self):
        """Test registering a new format."""
        registry = FileFormatRegistry()
        registry.register('dummy', DummyFormat)
        
        handler = registry.get_handler('dummy')
        assert isinstance(handler, DummyFormat)
    
    def test_case_insensitive(self):
        """Test that format names are case insensitive."""
        registry = FileFormatRegistry()
        registry.register('TEST', DummyFormat)
        
        # All of these should work
        assert isinstance(registry.get_handler('test'), DummyFormat)
        assert isinstance(registry.get_handler('TEST'), DummyFormat)
        assert isinstance(registry.get_handler('Test'), DummyFormat)
    
    def test_unknown_format(self):
        """Test getting unknown format raises error."""
        registry = FileFormatRegistry()
        
        with pytest.raises(ValueError, match="Unknown file format: unknown"):
            registry.get_handler('unknown')
    
    def test_get_handler_for_file(self):
        """Test getting handler based on file extension."""
        registry = FileFormatRegistry()
        registry.register('dummy', DummyFormat)
        
        # Test TPTP extensions
        assert isinstance(registry.get_handler_for_file(Path('test.p')), TPTPFormat)
        assert isinstance(registry.get_handler_for_file(Path('test.tptp')), TPTPFormat)
        assert isinstance(registry.get_handler_for_file(Path('test.ax')), TPTPFormat)
        
        # Test dummy extensions
        assert isinstance(registry.get_handler_for_file(Path('test.dum')), DummyFormat)
        assert isinstance(registry.get_handler_for_file(Path('test.dummy')), DummyFormat)
    
    def test_no_handler_for_extension(self):
        """Test that unknown extension raises error."""
        registry = FileFormatRegistry()
        
        with pytest.raises(ValueError, match="No handler found for file extension: .xyz"):
            registry.get_handler_for_file(Path('test.xyz'))


class TestGetFormatHandler:
    """Test the get_format_handler convenience function."""
    
    def test_get_by_name(self):
        """Test getting handler by format name."""
        handler = get_format_handler(format_name='tptp')
        assert isinstance(handler, TPTPFormat)
    
    def test_get_by_file_path(self):
        """Test getting handler by file path."""
        handler = get_format_handler(file_path=Path('test.p'))
        assert isinstance(handler, TPTPFormat)
    
    def test_no_arguments(self):
        """Test that no arguments raises error."""
        with pytest.raises(ValueError, match="Either format_name or file_path must be provided"):
            get_format_handler()
    
    def test_both_arguments(self):
        """Test that format_name takes precedence."""
        handler = get_format_handler(format_name='tptp', file_path=Path('test.xyz'))
        assert isinstance(handler, TPTPFormat)