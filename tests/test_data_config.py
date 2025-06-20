"""Tests for data config module."""

import pytest
from pathlib import Path
import tempfile
import yaml

from proofatlas.data.config import DatasetSplit, DatasetConfig


class TestDatasetSplit:
    """Test DatasetSplit class."""
    
    def test_create_split(self):
        """Test creating a dataset split."""
        split = DatasetSplit(
            name="train",
            files=["file1.p", "file2.p"],
            patterns=["problems/*.p"],
            ratio=0.8
        )
        
        assert split.name == "train"
        assert split.files == ["file1.p", "file2.p"]
        assert split.patterns == ["problems/*.p"]
        assert split.ratio == 0.8
    
    def test_create_minimal_split(self):
        """Test creating split with minimal info."""
        split = DatasetSplit(name="test")
        
        assert split.name == "test"
        assert split.files == []
        assert split.patterns == []
        assert split.ratio is None


class TestDatasetConfig:
    """Test DatasetConfig class."""
    
    def test_create_config(self):
        """Test creating a dataset configuration."""
        splits = [
            DatasetSplit(name="train", files=["train1.p", "train2.p"]),
            DatasetSplit(name="val", files=["val1.p"]),
            DatasetSplit(name="test", files=["test1.p"])
        ]
        
        config = DatasetConfig(
            name="test_dataset",
            file_format="tptp",
            data_format="graph",
            base_path=Path("/data/problems"),
            splits=splits,
            metadata={"version": "1.0"}
        )
        
        assert config.name == "test_dataset"
        assert config.file_format == "tptp"
        assert config.data_format == "graph"
        assert config.base_path == Path("/data/problems")
        assert len(config.splits) == 3
        assert config.metadata["version"] == "1.0"
    
    def test_get_split(self):
        """Test getting a split by name."""
        splits = [
            DatasetSplit(name="train", files=["train1.p"]),
            DatasetSplit(name="val", files=["val1.p"])
        ]
        
        config = DatasetConfig(
            name="test",
            file_format="tptp",
            data_format="graph",
            base_path=Path("."),
            splits=splits
        )
        
        train_split = config.get_split("train")
        assert train_split is not None
        assert train_split.name == "train"
        assert train_split.files == ["train1.p"]
        
        # Non-existent split
        assert config.get_split("nonexistent") is None
    
    def test_from_yaml(self):
        """Test loading config from YAML."""
        yaml_content = """
name: test_dataset
file_format: tptp
data_format: graph
base_path: /data/problems
splits:
  - name: train
    files:
      - train1.p
      - train2.p
    patterns:
      - "train/*.p"
    ratio: 0.8
  - name: val
    files:
      - val1.p
    ratio: 0.1
  - name: test
    files:
      - test1.p
    ratio: 0.1
metadata:
  version: "1.0"
  description: "Test dataset"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            config = DatasetConfig.from_yaml(Path(f.name))
            
            assert config.name == "test_dataset"
            assert config.file_format == "tptp"
            assert config.data_format == "graph"
            assert config.base_path == Path("/data/problems")
            assert len(config.splits) == 3
            
            # Check splits
            train = config.get_split("train")
            assert train.files == ["train1.p", "train2.p"]
            assert train.patterns == ["train/*.p"]
            assert train.ratio == 0.8
            
            # Check metadata
            assert config.metadata["version"] == "1.0"
            assert config.metadata["description"] == "Test dataset"
            
            # Cleanup
            Path(f.name).unlink()
    
    def test_to_yaml(self):
        """Test saving config to YAML."""
        config = DatasetConfig(
            name="test_dataset",
            file_format="tptp",
            data_format="graph",
            base_path=Path("/data/problems"),
            splits=[
                DatasetSplit(name="train", files=["train1.p"], ratio=0.8),
                DatasetSplit(name="val", files=["val1.p"], ratio=0.2)
            ],
            metadata={"version": "1.0"}
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.to_yaml(Path(f.name))
            
            # Read back and verify
            with open(f.name, 'r') as rf:
                data = yaml.safe_load(rf)
                
                assert data['name'] == "test_dataset"
                assert data['file_format'] == "tptp"
                assert data['data_format'] == "graph"
                assert data['base_path'] == "/data/problems"
                assert len(data['splits']) == 2
                assert data['metadata']['version'] == "1.0"
                
                # Check splits
                assert data['splits'][0]['name'] == "train"
                assert data['splits'][0]['files'] == ["train1.p"]
                assert data['splits'][0]['ratio'] == 0.8
            
            # Cleanup
            Path(f.name).unlink()
    
    def test_yaml_round_trip(self):
        """Test that saving and loading preserves data."""
        original = DatasetConfig(
            name="test",
            file_format="tptp",
            data_format="graph",
            base_path=Path("/data"),
            splits=[
                DatasetSplit(name="train", files=["a.p", "b.p"], patterns=["*.p"]),
                DatasetSplit(name="test", files=["c.p"])
            ],
            metadata={"key": "value"}
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            original.to_yaml(Path(f.name))
            loaded = DatasetConfig.from_yaml(Path(f.name))
            
            assert loaded.name == original.name
            assert loaded.file_format == original.file_format
            assert loaded.data_format == original.data_format
            assert loaded.base_path == original.base_path
            assert len(loaded.splits) == len(original.splits)
            assert loaded.metadata == original.metadata
            
            # Cleanup
            Path(f.name).unlink()