"""Registry for given clause loops."""

from typing import Dict, Type, List, Any

from .base import Loop
from .basic import BasicLoop


class LoopRegistry:
    """Registry for managing given clause loops."""
    
    def __init__(self):
        self._loops: Dict[str, Type[Loop]] = {}
        self._register_default_loops()
    
    def _register_default_loops(self):
        """Register default loops."""
        self.register('basic', BasicLoop)
    
    def register(self, name: str, loop_class: Type[Loop]):
        """Register a new loop type."""
        self._loops[name.lower()] = loop_class
    
    def create_loop(self, name: str, **kwargs: Any) -> Loop:
        """Create a loop instance."""
        name = name.lower()
        if name not in self._loops:
            raise ValueError(f"Unknown loop: {name}")
        
        return self._loops[name](**kwargs)
    
    def list_loops(self) -> List[str]:
        """List available loop names."""
        return list(self._loops.keys())


_registry = LoopRegistry()


def get_loop(name: str, **kwargs: Any) -> Loop:
    """Get a loop instance."""
    return _registry.create_loop(name, **kwargs)