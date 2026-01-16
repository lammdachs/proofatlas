#!/usr/bin/env python3
"""
Train clause selection models.

USAGE:
    proofatlas-train --traces steps_sel22 --preset gcn_mlp   # Use preset from proofatlas.json
    proofatlas-train --traces steps_sel22 --embedding gcn --scorer mlp  # Specify directly
    proofatlas-train --traces steps_sel22                    # Uses default preset

CONFIG FILES:
    configs/proofatlas.json  - Presets with embedding/scorer combinations
    configs/embeddings.json  - Embedding architecture configs (gcn, etc.)
    configs/scorers.json     - Scorer architecture configs (mlp, attention, etc.)

OUTPUT:
    .logs/<run_name>/              - Training logs and checkpoints
    .logs/<run_name>/metrics.json  - Metrics for web visualization
    .weights/<run_name>.safetensors - Exported model weights
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

# Find project root
def find_project_root() -> Path:
    """Find the proofatlas project root."""
    candidates = [Path.cwd(), Path(__file__).parent.parent]
    for candidate in candidates:
        if (candidate / "configs" / "proofatlas.json").exists():
            return candidate.resolve()

    path = Path.cwd()
    while path != path.parent:
        if (path / "configs" / "proofatlas.json").exists():
            return path.resolve()
        path = path.parent

    return Path.cwd()


ROOT = find_project_root()


# =============================================================================
# Config Loading
# =============================================================================


def load_configs(root: Path) -> tuple[dict, dict, dict]:
    """Load proofatlas, embeddings, and scorers configs."""
    configs_dir = root / "configs"

    with open(configs_dir / "proofatlas.json") as f:
        proofatlas_config = json.load(f)

    with open(configs_dir / "embeddings.json") as f:
        embeddings_config = json.load(f)

    with open(configs_dir / "scorers.json") as f:
        scorers_config = json.load(f)

    return proofatlas_config, embeddings_config, scorers_config


def get_model_config(
    proofatlas_config: dict,
    embeddings_config: dict,
    scorers_config: dict,
    preset: Optional[str] = None,
    embedding: Optional[str] = None,
    scorer: Optional[str] = None,
) -> Dict[str, Any]:
    """Get model configuration from preset or explicit embedding/scorer.

    Returns dict with keys: embedding_type, embedding_config, scorer_type, scorer_config
    """
    # If preset specified, get embedding/scorer from it
    if preset:
        if preset not in proofatlas_config.get("presets", {}):
            available = list(proofatlas_config.get("presets", {}).keys())
            raise ValueError(f"Unknown preset '{preset}'. Available: {available}")
        preset_config = proofatlas_config["presets"][preset]
        embedding = preset_config.get("embedding", embedding)
        scorer = preset_config.get("scorer", scorer)

    # Default to gcn/mlp if not specified
    if not embedding:
        embedding = "gcn"
    if not scorer:
        scorer = "mlp"

    # Look up embedding config
    if embedding not in embeddings_config.get("architectures", {}):
        available = list(embeddings_config.get("architectures", {}).keys())
        raise ValueError(f"Unknown embedding '{embedding}'. Available: {available}")
    embedding_config = embeddings_config["architectures"][embedding]

    # Look up scorer config
    if scorer not in scorers_config.get("architectures", {}):
        available = list(scorers_config.get("architectures", {}).keys())
        raise ValueError(f"Unknown scorer '{scorer}'. Available: {available}")
    scorer_config = scorers_config["architectures"][scorer]

    return {
        "embedding_type": embedding,
        "embedding_config": embedding_config,
        "scorer_type": scorer,
        "scorer_config": scorer_config,
    }


# =============================================================================
# Trace Loading
# =============================================================================


class ProofDatasetFromFiles:
    """Dataset for JSON proof traces, takes file list directly."""

    def __init__(self, files: List[Path], sample_prefix: bool = True, min_prefix: int = 10, max_clauses: int = 200):
        self.files = files
        self.sample_prefix = sample_prefix
        self.min_prefix = min_prefix
        self.max_clauses = max_clauses
        self._clause_to_graph = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        import json as json_module

        with open(self.files[idx]) as f:
            trace = json_module.load(f)

        clauses = trace.get("clauses", [])
        n = len(clauses)

        # Sample a random prefix
        if self.sample_prefix and n > self.min_prefix:
            prefix_len = random.randint(self.min_prefix, min(n, self.max_clauses))
            clauses = clauses[:prefix_len]
        elif n > self.max_clauses:
            # Keep all positives, sample negatives
            pos_indices = [i for i, c in enumerate(clauses) if c.get("label", 0) == 1]
            neg_indices = [i for i, c in enumerate(clauses) if c.get("label", 0) == 0]
            max_neg = self.max_clauses - len(pos_indices)
            if max_neg > 0 and len(neg_indices) > max_neg:
                neg_indices = random.sample(neg_indices, max_neg)
            indices = sorted(pos_indices + neg_indices)
            clauses = [clauses[i] for i in indices]

        # Lazy load converter
        if self._clause_to_graph is None:
            from proofatlas.ml.structured import clause_to_graph
            self._clause_to_graph = clause_to_graph

        max_age = len(clauses)
        graphs = [self._clause_to_graph(c, max_age) for c in clauses]
        labels = [c.get("label", 0) for c in clauses]

        return {
            "graphs": graphs,
            "labels": labels,
            "problem": self.files[idx].stem,
        }


# =============================================================================
# Training
# =============================================================================


def train_model(
    trace_dir: Path,
    model_config: Dict[str, Any],
    batch_size: int = 32,
    max_epochs: int = 100,
    learning_rate: float = 0.001,
    patience: int = 10,
    run_name: Optional[str] = None,
    log_dir: Path = None,
) -> Dict[str, Any]:
    """Train a clause selection model.

    Args:
        trace_dir: Directory with JSON trace files
        model_config: Dict from get_model_config with embedding/scorer configs
        batch_size: Training batch size
        max_epochs: Maximum training epochs
        learning_rate: Initial learning rate
        patience: Early stopping patience
        run_name: Name for this training run
        log_dir: Directory for logs

    Returns:
        Training metrics dictionary
    """
    # Import ML modules
    try:
        from proofatlas.ml.config import SelectorConfig, ModelConfig, TrainingParams, LoggingConfig
        from proofatlas.ml.training import train
    except ImportError as e:
        print(f"Error: ML dependencies not available. Install with: pip install -e '.[ml]'")
        print(f"  {e}")
        sys.exit(1)

    if log_dir is None:
        log_dir = ROOT / ".logs"

    embedding_type = model_config["embedding_type"]
    embedding_cfg = model_config["embedding_config"]
    scorer_type = model_config["scorer_type"]
    scorer_cfg = model_config["scorer_config"]

    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{embedding_type}_{scorer_type}_{trace_dir.name}_{timestamp}"

    print(f"\n{'='*60}")
    print(f"Training: {run_name}")
    print(f"{'='*60}")
    print(f"  Trace dir: {trace_dir}")
    print(f"  Embedding: {embedding_type}")
    for k, v in embedding_cfg.items():
        if k != "type":
            print(f"    {k}: {v}")
    print(f"  Scorer: {scorer_type}")
    for k, v in scorer_cfg.items():
        if k != "type":
            print(f"    {k}: {v}")
    print(f"  Batch size: {batch_size}, LR: {learning_rate}")
    print(f"  Max epochs: {max_epochs}, Patience: {patience}")
    print()

    # Load JSON traces
    json_files = sorted(trace_dir.glob("*.json"))
    if not json_files:
        print(f"Error: No JSON trace files found in {trace_dir}")
        sys.exit(1)

    print(f"  Found {len(json_files)} JSON traces")

    # Split traces into train/val by file
    random.seed(42)
    random.shuffle(json_files)

    n = len(json_files)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_files = json_files[:train_end]
    val_files = json_files[train_end:val_end]

    # Create datasets with file lists directly
    train_dataset = ProofDatasetFromFiles(train_files, sample_prefix=True)
    val_dataset = ProofDatasetFromFiles(val_files, sample_prefix=False)

    print(f"  Train: {len(train_dataset)} proofs, Val: {len(val_dataset)} proofs")

    # Build config - combine embedding type with scorer type for factory
    # The factory will interpret this and create the right architecture
    config = SelectorConfig(
        name=run_name,
        model=ModelConfig(
            type=embedding_type,  # Main type is the embedding
            hidden_dim=embedding_cfg.get("hidden_dim", 64),
            num_layers=embedding_cfg.get("num_layers", 3),
            dropout=embedding_cfg.get("dropout", 0.1),
            scorer_type=scorer_type,
            scorer_num_layers=scorer_cfg.get("num_layers", 2),
            scorer_num_heads=scorer_cfg.get("num_heads", 4),
        ),
        training=TrainingParams(
            batch_size=batch_size,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            patience=patience,
            loss_type="info_nce",
        ),
        logging=LoggingConfig(
            log_dir=str(log_dir),
            enable_progress_bar=True,
        ),
    )

    # Train
    print("\nStarting training...")
    model, metrics = train(train_dataset, val_dataset, config)

    print(f"\nTraining complete!")
    print(f"  Best val_loss: {metrics.get('best_val_loss', 'N/A'):.4f}")
    print(f"  Logs saved to: {log_dir / run_name}")

    return metrics


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train clause selection models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Main options
    parser.add_argument(
        "--traces",
        type=str,
        help="Trace directory name (e.g., steps_sel22) or full path",
    )

    # Model selection (from configs)
    parser.add_argument(
        "--preset",
        type=str,
        help="Preset name from proofatlas.json (e.g., gcn_mlp)",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        help="Embedding type from embeddings.json (e.g., gcn, none)",
    )
    parser.add_argument(
        "--scorer",
        type=str,
        help="Scorer type from scorers.json (e.g., mlp, attention)",
    )

    # Training params
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs (default: 100)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")

    # Run options
    parser.add_argument("--name", type=str, help="Run name (default: auto-generated)")
    parser.add_argument("--log-dir", type=Path, help="Log directory (default: .logs)")

    args = parser.parse_args()

    # Set up log directory
    log_dir = args.log_dir if args.log_dir else ROOT / ".logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Training mode requires --traces
    if not args.traces:
        parser.error("--traces is required")

    # Resolve trace directory
    if "/" in args.traces or "\\" in args.traces:
        trace_dir = Path(args.traces)
    else:
        trace_dir = ROOT / ".data" / "traces" / args.traces

    if not trace_dir.exists():
        print(f"Error: Trace directory not found: {trace_dir}")
        sys.exit(1)

    # Load configs and get model configuration
    proofatlas_config, embeddings_config, scorers_config = load_configs(ROOT)

    try:
        model_config = get_model_config(
            proofatlas_config,
            embeddings_config,
            scorers_config,
            preset=args.preset,
            embedding=args.embedding,
            scorer=args.scorer,
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Train
    train_model(
        trace_dir=trace_dir,
        model_config=model_config,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        run_name=args.name,
        log_dir=log_dir,
    )


if __name__ == "__main__":
    main()
