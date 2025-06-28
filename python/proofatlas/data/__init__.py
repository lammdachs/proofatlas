"""
Dataset management for theorem proving problems.
"""

from .config import DatasetConfig, DatasetSplit
from .problemset import Problemset
from .proofset import Proofset
from .manager import DatasetManager
from .splits import (
    random_split,
    stratified_split,
    create_standard_splits,
    split_by_pattern,
    k_fold_split
)

__all__ = [
    'DatasetConfig', 
    'DatasetSplit',
    'Problemset', 
    'Proofset', 
    'DatasetManager',
    'random_split',
    'stratified_split',
    'create_standard_splits',
    'split_by_pattern',
    'k_fold_split'
]