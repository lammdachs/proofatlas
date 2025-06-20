"""
Proving environments implementing transition relations between proof states.
"""

from .base import ProvingEnvironment, ProofAction, ProofTransition
from .given_clause import GivenClauseEnvironment
from .registry import EnvironmentRegistry, get_environment

__all__ = [
    'ProvingEnvironment', 'ProofAction', 'ProofTransition',
    'GivenClauseEnvironment', 'EnvironmentRegistry', 'get_environment'
]