"""Registry for proving environments."""

from typing import Dict, Type, List, Any

from proofatlas.core.fol.logic import Clause
from .base import ProvingEnvironment
from .given_clause import GivenClauseEnvironment


class EnvironmentRegistry:
    """Registry for managing proving environments."""
    
    def __init__(self):
        self._environments: Dict[str, Type[ProvingEnvironment]] = {}
        self._register_default_environments()
    
    def _register_default_environments(self):
        """Register default environments."""
        self.register('given_clause', GivenClauseEnvironment)
    
    def register(self, name: str, env_class: Type[ProvingEnvironment]):
        """Register a new environment type."""
        self._environments[name.lower()] = env_class
    
    def create_environment(self, name: str, initial_clauses: List[Clause], 
                         **kwargs: Any) -> ProvingEnvironment:
        """Create an environment instance."""
        name = name.lower()
        if name not in self._environments:
            raise ValueError(f"Unknown environment: {name}")
        
        return self._environments[name](initial_clauses, **kwargs)
    
    def list_environments(self) -> List[str]:
        """List available environment names."""
        return list(self._environments.keys())


_registry = EnvironmentRegistry()


def get_environment(name: str, initial_clauses: List[Clause], **kwargs: Any) -> ProvingEnvironment:
    """Get a proving environment instance."""
    return _registry.create_environment(name, initial_clauses, **kwargs)