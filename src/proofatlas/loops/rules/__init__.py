"""Inference rules for theorem proving."""

from .resolution import ResolutionRule
from .factoring import FactoringRule
from .subsumption import SubsumptionRule

__all__ = ['ResolutionRule', 'FactoringRule', 'SubsumptionRule']