"""
Dataset splitting utilities for creating train/validation/test splits.
"""

import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict

from .config import DatasetSplit


def random_split(
    files: List[Union[str, Path]], 
    ratios: Dict[str, float],
    seed: Optional[int] = None
) -> Dict[str, DatasetSplit]:
    """
    Randomly split files into train/val/test sets.
    
    Args:
        files: List of file paths to split
        ratios: Dictionary mapping split names to ratios (e.g., {"train": 0.7, "val": 0.15, "test": 0.15})
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping split names to DatasetSplit objects
        
    Raises:
        ValueError: If ratios don't sum to 1.0 or invalid split names
    """
    # Validate ratios
    ratio_sum = sum(ratios.values())
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum}")
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Shuffle files
    files = list(files)  # Copy to avoid modifying original
    random.shuffle(files)
    
    # Calculate split sizes
    n_files = len(files)
    splits = {}
    start_idx = 0
    
    split_names = list(ratios.keys())
    for i, split_name in enumerate(split_names):
        ratio = ratios[split_name]
        
        # For the last split, take all remaining files to avoid rounding errors
        if i == len(split_names) - 1:
            end_idx = n_files
        else:
            end_idx = start_idx + int(n_files * ratio)
        
        split_files = [str(f) for f in files[start_idx:end_idx]]
        splits[split_name] = DatasetSplit(
            name=split_name,
            files=split_files,
            patterns=[],
            ratio=ratio
        )
        
        start_idx = end_idx
    
    return splits


def stratified_split(
    files: List[Union[str, Path]], 
    ratios: Dict[str, float],
    stratify_fn,
    seed: Optional[int] = None
) -> Dict[str, DatasetSplit]:
    """
    Create stratified splits based on a stratification function.
    
    Args:
        files: List of file paths to split
        ratios: Dictionary mapping split names to ratios
        stratify_fn: Function that takes a file path and returns a stratification key
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping split names to DatasetSplit objects
    """
    # Group files by stratification key
    strata = defaultdict(list)
    for file in files:
        key = stratify_fn(file)
        strata[key].append(file)
    
    # Split each stratum independently
    split_files = defaultdict(list)
    
    for key, stratum_files in strata.items():
        stratum_splits = random_split(stratum_files, ratios, seed)
        for split_name, split in stratum_splits.items():
            split_files[split_name].extend(split.files)
    
    # Create DatasetSplit objects
    splits = {}
    for split_name, files in split_files.items():
        splits[split_name] = DatasetSplit(
            name=split_name,
            files=files,
            patterns=[],
            ratio=ratios.get(split_name, 0.0)
        )
    
    return splits


def create_standard_splits(
    files: List[Union[str, Path]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: Optional[int] = None
) -> Dict[str, DatasetSplit]:
    """
    Create standard train/val/test splits with default ratios.
    
    Args:
        files: List of file paths to split
        train_ratio: Ratio for training set (default: 0.7)
        val_ratio: Ratio for validation set (default: 0.15)
        test_ratio: Ratio for test set (default: 0.15)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with "train", "val", and "test" DatasetSplit objects
    """
    ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio
    }
    return random_split(files, ratios, seed)


def split_by_pattern(
    directory: Union[str, Path],
    patterns: Dict[str, List[str]]
) -> Dict[str, DatasetSplit]:
    """
    Create splits based on file patterns.
    
    Args:
        directory: Base directory for file patterns
        patterns: Dictionary mapping split names to lists of glob patterns
        
    Returns:
        Dictionary mapping split names to DatasetSplit objects
        
    Example:
        patterns = {
            "train": ["train/*.tptp", "training/**/*.p"],
            "val": ["val/*.tptp", "validation/**/*.p"],
            "test": ["test/*.tptp", "testing/**/*.p"]
        }
    """
    splits = {}
    
    for split_name, split_patterns in patterns.items():
        splits[split_name] = DatasetSplit(
            name=split_name,
            files=[],
            patterns=split_patterns,
            ratio=None
        )
    
    return splits


def k_fold_split(
    files: List[Union[str, Path]],
    n_folds: int = 5,
    seed: Optional[int] = None
) -> List[Tuple[DatasetSplit, DatasetSplit]]:
    """
    Create k-fold cross-validation splits.
    
    Args:
        files: List of file paths to split
        n_folds: Number of folds (default: 5)
        seed: Random seed for reproducibility
        
    Returns:
        List of (train, val) split tuples, one for each fold
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Shuffle files
    files = list(files)
    random.shuffle(files)
    
    # Calculate fold size
    fold_size = len(files) // n_folds
    
    folds = []
    for i in range(n_folds):
        # Validation fold
        val_start = i * fold_size
        if i == n_folds - 1:
            # Last fold gets remaining files
            val_end = len(files)
        else:
            val_end = (i + 1) * fold_size
        
        val_files = [str(f) for f in files[val_start:val_end]]
        train_files = [str(f) for f in files[:val_start] + files[val_end:]]
        
        train_split = DatasetSplit(
            name=f"train_fold_{i}",
            files=train_files,
            patterns=[],
            ratio=None
        )
        
        val_split = DatasetSplit(
            name=f"val_fold_{i}",
            files=val_files,
            patterns=[],
            ratio=None
        )
        
        folds.append((train_split, val_split))
    
    return folds