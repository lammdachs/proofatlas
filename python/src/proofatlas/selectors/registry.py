"""Registry for clause selectors."""

from typing import Dict, Type, Any, Optional

from .base import Selector
from .random import RandomSelector
from .gnn import GNNSelector


class SelectorRegistry:
    """Registry for managing clause selectors."""
    
    def __init__(self):
        self._selectors: Dict[str, Type[Selector]] = {}
        self._register_default_selectors()
    
    def _register_default_selectors(self):
        """Register default selectors."""
        self.register('fifo', RandomSelector)  # Legacy name
        self.register('random', RandomSelector)  # Preferred name
        self.register('gnn', GNNSelector)
    
    def register(self, name: str, selector_class: Type[Selector]):
        """Register a new selector type."""
        self._selectors[name.lower()] = selector_class
    
    def create_selector(self, name: str, **kwargs: Any) -> Selector:
        """Create a selector instance."""
        name = name.lower()
        
        if name not in self._selectors:
            raise ValueError(f"Unknown selector: {name}")
        
        return self._selectors[name](**kwargs)
    
    def list_selectors(self) -> list:
        """List available selector names."""
        return list(self._selectors.keys())


_registry = SelectorRegistry()


def get_selector(name: str, **kwargs: Any) -> Selector:
    """Get a clause selector instance."""
    return _registry.create_selector(name, **kwargs)