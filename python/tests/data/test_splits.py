"""Tests for dataset splitting utilities."""

import pytest
from pathlib import Path
from typing import List

from proofatlas.data.splits import (
    random_split,
    stratified_split,
    create_standard_splits,
    split_by_pattern,
    k_fold_split
)
from proofatlas.data.config import DatasetSplit


class TestRandomSplit:
    """Test random_split function."""
    
    def test_basic_split(self):
        """Test basic random splitting."""
        files = [f"file_{i}.tptp" for i in range(100)]
        ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
        
        splits = random_split(files, ratios, seed=42)
        
        # Check split names
        assert set(splits.keys()) == {"train", "val", "test"}
        
        # Check split sizes (allowing for rounding)
        assert len(splits["train"].files) == 70
        assert len(splits["val"].files) == 15
        assert len(splits["test"].files) == 15
        
        # Check no overlap
        all_files = set()
        for split in splits.values():
            files_set = set(split.files)
            assert len(files_set & all_files) == 0
            all_files.update(files_set)
        
        # Check all files are included
        assert all_files == set(files)
    
    def test_invalid_ratios(self):
        """Test that invalid ratios raise errors."""
        files = ["file1.tptp", "file2.tptp"]
        
        # Ratios don't sum to 1
        with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
            random_split(files, {"train": 0.7, "val": 0.2})
    
    def test_reproducibility(self):
        """Test that same seed gives same results."""
        files = [f"file_{i}.tptp" for i in range(50)]
        ratios = {"train": 0.8, "test": 0.2}
        
        splits1 = random_split(files, ratios, seed=123)
        splits2 = random_split(files, ratios, seed=123)
        
        assert splits1["train"].files == splits2["train"].files
        assert splits1["test"].files == splits2["test"].files
    
    def test_small_dataset(self):
        """Test splitting a small dataset."""
        files = ["a.tptp", "b.tptp", "c.tptp"]
        ratios = {"train": 0.67, "test": 0.33}  # Should give 2 train, 1 test
        
        splits = random_split(files, ratios, seed=42)
        
        assert len(splits["train"].files) == 2
        assert len(splits["test"].files) == 1


class TestStratifiedSplit:
    """Test stratified_split function."""
    
    def test_stratified_by_prefix(self):
        """Test stratification by file prefix."""
        files = []
        # Create files with different prefixes
        for prefix in ["easy", "medium", "hard"]:
            for i in range(10):
                files.append(f"{prefix}_{i}.tptp")
        
        def stratify_fn(file):
            return file.split("_")[0]
        
        ratios = {"train": 0.7, "val": 0.3}
        splits = stratified_split(files, ratios, stratify_fn, seed=42)
        
        # Check that each stratum is split proportionally
        for prefix in ["easy", "medium", "hard"]:
            train_count = sum(1 for f in splits["train"].files if f.startswith(prefix))
            val_count = sum(1 for f in splits["val"].files if f.startswith(prefix))
            
            assert train_count == 7
            assert val_count == 3
    
    def test_uneven_strata(self):
        """Test stratification with uneven strata sizes."""
        files = ["a_1.tptp", "a_2.tptp", "b_1.tptp", "b_2.tptp", "b_3.tptp", "b_4.tptp"]
        
        def stratify_fn(file):
            return file[0]  # First character
        
        ratios = {"train": 0.5, "test": 0.5}
        splits = stratified_split(files, ratios, stratify_fn, seed=42)
        
        # Each stratum should be split evenly
        assert len(splits["train"].files) == 3
        assert len(splits["test"].files) == 3


class TestCreateStandardSplits:
    """Test create_standard_splits function."""
    
    def test_default_ratios(self):
        """Test with default ratios."""
        files = [f"file_{i}.tptp" for i in range(100)]
        
        splits = create_standard_splits(files, seed=42)
        
        assert len(splits["train"].files) == 70
        assert len(splits["val"].files) == 15
        assert len(splits["test"].files) == 15
    
    def test_custom_ratios(self):
        """Test with custom ratios."""
        files = [f"file_{i}.tptp" for i in range(100)]
        
        splits = create_standard_splits(
            files, 
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42
        )
        
        assert len(splits["train"].files) == 80
        assert len(splits["val"].files) == 10
        assert len(splits["test"].files) == 10


class TestSplitByPattern:
    """Test split_by_pattern function."""
    
    def test_pattern_splits(self):
        """Test creating splits from patterns."""
        patterns = {
            "train": ["train/*.tptp", "training/**/*.p"],
            "val": ["val/*.tptp"],
            "test": ["test/*.tptp", "testing/**/*.p"]
        }
        
        splits = split_by_pattern("/data", patterns)
        
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        
        assert splits["train"].patterns == ["train/*.tptp", "training/**/*.p"]
        assert splits["val"].patterns == ["val/*.tptp"]
        assert splits["test"].patterns == ["test/*.tptp", "testing/**/*.p"]
        
        # Files should be empty (patterns will be resolved when loading)
        assert all(split.files == [] for split in splits.values())


class TestKFoldSplit:
    """Test k_fold_split function."""
    
    def test_5_fold(self):
        """Test 5-fold cross-validation."""
        files = [f"file_{i}.tptp" for i in range(50)]
        
        folds = k_fold_split(files, n_folds=5, seed=42)
        
        assert len(folds) == 5
        
        # Check each fold
        for i, (train, val) in enumerate(folds):
            assert train.name == f"train_fold_{i}"
            assert val.name == f"val_fold_{i}"
            assert len(train.files) == 40  # 80% of 50
            assert len(val.files) == 10    # 20% of 50
            
            # No overlap between train and val
            assert set(train.files) & set(val.files) == set()
    
    def test_3_fold_uneven(self):
        """Test 3-fold with uneven split."""
        files = [f"file_{i}.tptp" for i in range(10)]
        
        folds = k_fold_split(files, n_folds=3, seed=42)
        
        assert len(folds) == 3
        
        # First two folds get 3 files each, last fold gets 4
        assert len(folds[0][1].files) == 3
        assert len(folds[1][1].files) == 3
        assert len(folds[2][1].files) == 4
        
        # All validation sets together should cover all files
        all_val = set()
        for _, val in folds:
            all_val.update(val.files)
        assert all_val == set(files)
    
    def test_invalid_folds(self):
        """Test that invalid n_folds raises error."""
        files = ["a.tptp", "b.tptp"]
        
        with pytest.raises(ValueError, match="n_folds must be at least 2"):
            k_fold_split(files, n_folds=1)