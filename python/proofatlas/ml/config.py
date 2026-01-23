"""
Configuration loading and validation for ML training and data pipelines.

Configs are stored in:
- configs/data/ - Data collection and filtering (references Rust selectors)
- configs/training/ - ML model definitions and training params (PyTorch)

Selectors are implemented in:
- rust/src/selection/ - Rust/Burn implementations (used at runtime)
- python/proofatlas/selectors/ - PyTorch implementations (used for training)

ML selectors (gcn, gat, mlp, transformer) are mirrored between Rust and Python.
Heuristic selectors (age_weight) only have Rust implementations.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def get_config_dir() -> Path:
    """Get the configs directory."""
    # Try relative to this file first
    config_dir = Path(__file__).parent.parent.parent.parent / "configs"
    if config_dir.exists():
        return config_dir
    # Fall back to current working directory
    config_dir = Path.cwd() / "configs"
    if config_dir.exists():
        return config_dir
    raise FileNotFoundError("configs directory not found")


# =============================================================================
# Data Configuration
# =============================================================================


@dataclass
class ProblemFilters:
    """Filters for selecting problems from metadata."""
    status: Optional[List[str]] = None  # ["unsatisfiable", "satisfiable", "unknown"]
    format: Optional[List[str]] = None  # ["cnf", "fof"]
    has_equality: Optional[bool] = None
    is_unit_only: Optional[bool] = None
    min_rating: Optional[float] = None
    max_rating: Optional[float] = None
    min_clauses: Optional[int] = None
    max_clauses: Optional[int] = None
    domains: Optional[List[str]] = None  # Include only these domains
    exclude_domains: Optional[List[str]] = None  # Exclude these domains


@dataclass
class SplitConfig:
    """Train/val/test split configuration."""
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    stratify_by: Optional[str] = "domain"


@dataclass
class SelectorSpec:
    """Specification for which selector to use."""
    name: str = "age_weight"  # Selector name (maps to Rust implementation)
    weights: Optional[str] = None  # Weights file in .weights/ (for ML selectors)


@dataclass
class SolverConfig:
    """Solver configuration for data collection."""
    literal_selection: str = "0"  # 0=all, 20=maximal, 21=unique


@dataclass
class TraceCollectionConfig:
    """Configuration for proof trace collection."""
    prover_timeout: float = 60.0
    max_clauses: int = 5000
    max_steps: int = 10000
    save_all_steps: bool = False
    save_interval: int = 10


@dataclass
class OutputConfig:
    """Configuration for data output."""
    trace_dir: str = ".data/traces"
    cache_dir: str = ".data/cache"
    format: str = "torch"  # torch, numpy, json


@dataclass
class DataConfig:
    """Complete data configuration."""
    name: str = "default"
    description: str = ""
    selector: SelectorSpec = field(default_factory=SelectorSpec)
    solver: SolverConfig = field(default_factory=SolverConfig)
    problem_filters: ProblemFilters = field(default_factory=ProblemFilters)
    split: SplitConfig = field(default_factory=SplitConfig)
    trace_collection: TraceCollectionConfig = field(default_factory=TraceCollectionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataConfig":
        return cls(
            name=d.get("name", "default"),
            description=d.get("description", ""),
            selector=SelectorSpec(**d.get("selector", {})),
            solver=SolverConfig(**d.get("solver", {})),
            problem_filters=ProblemFilters(**d.get("problem_filters", {})),
            split=SplitConfig(**d.get("split", {})),
            trace_collection=TraceCollectionConfig(**d.get("trace_collection", {})),
            output=OutputConfig(**d.get("output", {})),
        )

    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DataConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def load_preset(cls, name: str) -> "DataConfig":
        """Load a preset data config by name."""
        config_dir = get_config_dir() / "data"
        path = config_dir / f"{name}.json"
        if not path.exists():
            available = [p.stem for p in config_dir.glob("*.json")]
            raise FileNotFoundError(
                f"Data config '{name}' not found. Available: {available}"
            )
        return cls.load(path)


# =============================================================================
# Training Configuration (for ML selectors)
# =============================================================================


@dataclass
class ModelConfig:
    """Model architecture configuration (PyTorch implementation)."""
    type: str = "gcn"  # gcn, gat, graphsage, transformer, gnn_transformer, mlp, sentence
    hidden_dim: int = 64
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    input_dim: int = 8  # raw feature dimension from graph.rs (compact format)
    # Scorer configuration
    scorer_type: str = "mlp"  # mlp, attention, transformer, cross_attention
    scorer_num_heads: int = 4
    scorer_num_layers: int = 2
    # Sentence model configuration
    sentence_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # HuggingFace model name
    freeze_encoder: bool = False  # freeze pretrained encoder weights


@dataclass
class TrainingParams:
    """Training hyperparameters."""
    data_config: Optional[str] = None  # Reference to data config name
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    max_epochs: int = 100
    patience: int = 10
    gradient_clip: float = 1.0
    # Loss configuration
    loss_type: str = "info_nce"  # info_nce, margin, bce, cross_entropy
    temperature: float = 1.0  # Temperature for InfoNCE loss


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    type: str = "adamw"
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    type: str = "cosine"  # cosine, step, plateau
    min_lr_ratio: float = 0.01
    warmup_epochs: int = 5
    step_size: int = 30  # For step scheduler
    gamma: float = 0.1  # For step scheduler


@dataclass
class DistributedConfig:
    """Distributed training configuration."""
    num_gpus: int = -1  # -1 = all available
    strategy: str = "auto"  # auto, ddp, ddp_spawn
    precision: str = "32-true"  # 32-true, 16-mixed, bf16-mixed


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_dir: str = ".logs"
    log_every_n_steps: int = 10
    enable_progress_bar: bool = True


@dataclass
class CheckpointingConfig:
    """Checkpointing configuration."""
    monitor: str = "val_loss"
    mode: str = "min"
    save_top_k: int = 3


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    eval_every_n_epochs: int = 10
    num_eval_problems: int = 0  # 0 = disable
    eval_timeout: float = 10.0


@dataclass
class TrainingConfig:
    """Complete training configuration for ML selectors."""
    name: str = "default"
    description: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingParams = field(default_factory=TrainingParams)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        return cls(
            name=d.get("name", "default"),
            description=d.get("description", ""),
            model=ModelConfig(**d.get("model", {})),
            training=TrainingParams(**d.get("training", {})),
            optimizer=OptimizerConfig(**d.get("optimizer", {})),
            scheduler=SchedulerConfig(**d.get("scheduler", {})),
            distributed=DistributedConfig(**d.get("distributed", {})),
            logging=LoggingConfig(**d.get("logging", {})),
            checkpointing=CheckpointingConfig(**d.get("checkpointing", {})),
            evaluation=EvaluationConfig(**d.get("evaluation", {})),
        )

    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TrainingConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def load_preset(cls, name: str) -> "TrainingConfig":
        """Load a preset training config by name."""
        config_dir = get_config_dir() / "training"
        path = config_dir / f"{name}.json"
        if not path.exists():
            available = [p.stem for p in config_dir.glob("*.json")]
            raise FileNotFoundError(
                f"Training config '{name}' not found. Available: {available}"
            )
        return cls.load(path)

    def get_data_config(self) -> Optional[DataConfig]:
        """Load the referenced data config, if any."""
        if self.training and self.training.data_config:
            return DataConfig.load_preset(self.training.data_config)
        return None


# Keep SelectorConfig as alias for backwards compatibility
SelectorConfig = TrainingConfig


# =============================================================================
# Utility Functions
# =============================================================================


def list_configs(config_type: str = "all") -> Dict[str, List[str]]:
    """List available configuration presets.

    Args:
        config_type: "data", "training", or "all"

    Returns:
        Dictionary mapping config type to list of available names
    """
    config_dir = get_config_dir()
    result = {}

    if config_type in ("all", "data"):
        data_dir = config_dir / "data"
        if data_dir.exists():
            result["data"] = [p.stem for p in data_dir.glob("*.json")]

    if config_type in ("all", "training"):
        training_dir = config_dir / "training"
        if training_dir.exists():
            result["training"] = [p.stem for p in training_dir.glob("*.json")]

    return result


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two config dictionaries, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result
