"""Registry for data format handlers."""

from typing import Dict, Optional, Type, Any

from .base import DataFormat
from .graph import GraphFormat


class DataFormatRegistry:
    """Registry for managing data format handlers."""
    
    def __init__(self):
        self._formats: Dict[str, Type[DataFormat]] = {}
        self._register_default_formats()
    
    def _register_default_formats(self):
        """Register default data format handlers."""
        self.register('graph', GraphFormat)
    
    def register(self, name: str, format_class: Type[DataFormat]):
        """Register a new data format handler."""
        self._formats[name.lower()] = format_class
    
    def get_format(self, name: str, **kwargs: Any) -> DataFormat:
        """Get a data format handler by name."""
        name = name.lower()
        if name not in self._formats:
            raise ValueError(f"Unknown data format: {name}")
        return self._formats[name](**kwargs)


_registry = DataFormatRegistry()


def get_data_format(name: str, **kwargs: Any) -> DataFormat:
    """Get a data format handler."""
    return _registry.get_format(name, **kwargs)