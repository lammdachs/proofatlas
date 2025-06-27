"""Dataset manager for loading and managing multiple datasets."""

from pathlib import Path
from typing import Dict, List, Optional
import yaml

from .config import DatasetConfig
from .problemset import Problemset


class DatasetManager:
    """Manages multiple datasets and their configurations."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path("configs/datasets")
        self.datasets: Dict[str, DatasetConfig] = {}
        self._loaded_datasets: Dict[str, Problemset] = {}
        
        if self.config_dir.exists():
            self._load_configs()
    
    def _load_configs(self):
        """Load all dataset configurations from config directory."""
        for config_file in self.config_dir.glob("*.yaml"):
            try:
                config = DatasetConfig.from_yaml(config_file)
                self.datasets[config.name] = config
            except Exception as e:
                print(f"Error loading config {config_file}: {e}")
    
    def add_dataset(self, config: DatasetConfig):
        """Add a dataset configuration."""
        self.datasets[config.name] = config
    
    def get_dataset(self, name: str, split: str = 'train') -> Problemset:
        """Get a dataset by name and split."""
        cache_key = f"{name}_{split}"
        
        if cache_key not in self._loaded_datasets:
            if name not in self.datasets:
                raise ValueError(f"Dataset '{name}' not found")
            
            config = self.datasets[name]
            dataset = Problemset(config, split)
            self._loaded_datasets[cache_key] = dataset
        
        return self._loaded_datasets[cache_key]
    
    def list_datasets(self) -> List[str]:
        """List all available dataset names."""
        return list(self.datasets.keys())
    
    def get_config(self, name: str) -> DatasetConfig:
        """Get dataset configuration by name."""
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found")
        return self.datasets[name]
    
    def save_config(self, config: DatasetConfig, filename: Optional[str] = None):
        """Save a dataset configuration to file."""
        if filename is None:
            filename = f"{config.name}.yaml"
        
        config_path = self.config_dir / filename
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config.to_yaml(config_path)
        
        # Add to loaded configs
        self.datasets[config.name] = config
    
    def create_dataset_from_files(self, name: str, file_format: str, data_format: str,
                                base_path: Path, train_files: List[str],
                                val_files: Optional[List[str]] = None,
                                test_files: Optional[List[str]] = None) -> DatasetConfig:
        """Create a dataset configuration from file lists."""
        from .config import DatasetSplit
        
        splits = [
            DatasetSplit(name='train', files=train_files)
        ]
        
        if val_files:
            splits.append(DatasetSplit(name='val', files=val_files))
        
        if test_files:
            splits.append(DatasetSplit(name='test', files=test_files))
        
        config = DatasetConfig(
            name=name,
            file_format=file_format,
            data_format=data_format,
            base_path=base_path,
            splits=splits
        )
        
        self.add_dataset(config)
        return config