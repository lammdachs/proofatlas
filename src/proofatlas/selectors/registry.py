"""Registry for clause selectors."""

from typing import Dict, Type, Any, Optional

from .base import ClauseSelector
from .random import FIFOSelector
from .gnn import GNNSelector


class SelectorRegistry:
    """Registry for managing clause selectors."""
    
    def __init__(self):
        self._selectors: Dict[str, Type[ClauseSelector]] = {}
        self._register_default_selectors()
    
    def _register_default_selectors(self):
        """Register default selectors."""
        self.register('fifo', FIFOSelector)  # Legacy name
        self.register('random', FIFOSelector)  # Preferred name
        self.register('gnn', GNNSelector)
    
    def register(self, name: str, selector_class: Type[ClauseSelector]):
        """Register a new selector type."""
        self._selectors[name.lower()] = selector_class
    
    def create_selector(self, name: str, **kwargs: Any) -> ClauseSelector:
        """Create a selector instance."""
        name = name.lower()
        
        if name not in self._selectors:
            raise ValueError(f"Unknown selector: {name}")
        
        return self._selectors[name](**kwargs)
    
    def list_selectors(self) -> list:
        """List available selector names."""
        return list(self._selectors.keys())


_registry = SelectorRegistry()


def get_selector(name: str, **kwargs: Any) -> ClauseSelector:
    """Get a clause selector instance."""
    return _registry.create_selector(name, **kwargs)