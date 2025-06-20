import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self.config = self._load_config()
        self._resolve_environment_variables()
    
    def _find_config_file(self) -> str:
        """Find the default config file."""
        possible_paths = [
            Path.cwd() / "configs" / "default.yaml",
            Path(__file__).parent.parent.parent.parent / "configs" / "default.yaml",
            Path.home() / ".foreduce" / "config.yaml",
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        raise FileNotFoundError("No configuration file found")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _resolve_environment_variables(self):
        """Resolve environment variables in config values."""
        def resolve_value(value):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                var_default = value[2:-1].split(":", 1)
                var_name = var_default[0]
                default_value = var_default[1] if len(var_default) > 1 else ""
                return os.environ.get(var_name, default_value)
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(v) for v in value]
            return value
        
        self.config = resolve_value(self.config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = deep_update(self.config, updates)


# Global config instance
_config = None

def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config