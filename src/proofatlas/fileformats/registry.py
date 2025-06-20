"""Registry for file format handlers."""

from pathlib import Path
from typing import Dict, Optional, Type

from .base import FileFormat
from .tptp import TPTPFormat


class FileFormatRegistry:
    """Registry for managing file format handlers."""
    
    def __init__(self):
        self._formats: Dict[str, Type[FileFormat]] = {}
        self._register_default_formats()
    
    def _register_default_formats(self):
        """Register default file format handlers."""
        self.register('tptp', TPTPFormat)
    
    def register(self, name: str, format_class: Type[FileFormat]):
        """Register a new file format handler."""
        self._formats[name.lower()] = format_class
    
    def get_handler(self, format_name: str) -> FileFormat:
        """Get a file format handler by name."""
        name = format_name.lower()
        if name not in self._formats:
            raise ValueError(f"Unknown file format: {format_name}")
        return self._formats[name]()
    
    def get_handler_for_file(self, file_path: Path) -> FileFormat:
        """Get appropriate handler based on file extension."""
        for format_name, format_class in self._formats.items():
            handler = format_class()
            if file_path.suffix in handler.extensions:
                return handler
        
        raise ValueError(f"No handler found for file extension: {file_path.suffix}")


_registry = FileFormatRegistry()


def get_format_handler(format_name: Optional[str] = None, file_path: Optional[Path] = None) -> FileFormat:
    """Get a file format handler."""
    if format_name:
        return _registry.get_handler(format_name)
    elif file_path:
        return _registry.get_handler_for_file(file_path)
    else:
        raise ValueError("Either format_name or file_path must be provided")