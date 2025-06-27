"""Dataset configuration classes."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any
import yaml


@dataclass
class DatasetSplit:
    """Configuration for a dataset split."""
    name: str
    files: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    ratio: Optional[float] = None


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    file_format: str
    data_format: str
    base_path: Path
    splits: List[DatasetSplit]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'DatasetConfig':
        """Load dataset configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Parse splits
        splits = []
        for split_data in data.get('splits', []):
            split = DatasetSplit(
                name=split_data['name'],
                files=split_data.get('files', []),
                patterns=split_data.get('patterns', []),
                ratio=split_data.get('ratio')
            )
            splits.append(split)
        
        return cls(
            name=data['name'],
            file_format=data['file_format'],
            data_format=data['data_format'],
            base_path=Path(data['base_path']),
            splits=splits,
            metadata=data.get('metadata', {})
        )
    
    def to_yaml(self, yaml_path: Path):
        """Save dataset configuration to YAML file."""
        data = {
            'name': self.name,
            'file_format': self.file_format,
            'data_format': self.data_format,
            'base_path': str(self.base_path),
            'splits': [],
            'metadata': self.metadata
        }
        
        for split in self.splits:
            split_data = {
                'name': split.name,
                'files': split.files,
                'patterns': split.patterns
            }
            if split.ratio is not None:
                split_data['ratio'] = split.ratio
            data['splits'].append(split_data)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def get_split(self, split_name: str) -> Optional[DatasetSplit]:
        """Get a specific split by name."""
        for split in self.splits:
            if split.name == split_name:
                return split
        return None