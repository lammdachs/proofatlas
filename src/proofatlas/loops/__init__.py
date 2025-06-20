"""
Given clause loops for theorem proving.
"""

from .base import Loop
from .basic import BasicLoop
from .registry import LoopRegistry, get_loop

__all__ = ['Loop', 'BasicLoop', 'LoopRegistry', 'get_loop']