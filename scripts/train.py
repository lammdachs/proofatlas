#!/usr/bin/env python3
"""
Train clause selection models.

USAGE:
    proofatlas-train --traces steps_sel22              # Train on traces
    proofatlas-train --traces steps_sel22 --model gcn  # Specify model type
    proofatlas-train --traces steps_sel22 --overparameterized  # Large model
    proofatlas-train --viewer                          # Start viewer only

MODEL SIZES:
    --small        : hidden_dim=32,  num_layers=2  (fast iteration)
    --default      : hidden_dim=64,  num_layers=3  (standard)
    --large        : hidden_dim=128, num_layers=4  (more capacity)
    --overparameterized : hidden_dim=256, num_layers=6  (for overfitting experiments)

OUTPUT:
    .logs/<run_name>/              - Training logs and checkpoints
    .logs/<run_name>/metrics.json  - Metrics for web visualization
    .weights/<run_name>.safetensors - Exported model weights (when --export)
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
# Model Size Presets
# =============================================================================


MODEL_SIZES = {
    "small": {"hidden_dim": 32, "num_layers": 2},
    "default": {"hidden_dim": 64, "num_layers": 3},
    "large": {"hidden_dim": 128, "num_layers": 4},
    "overparameterized": {"hidden_dim": 256, "num_layers": 6},
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
    model_type: str = "gcn",
    hidden_dim: int = 64,
    num_layers: int = 3,
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
        model_type: Model architecture (gcn, mlp, gat, etc.)
        hidden_dim: Hidden dimension size
        num_layers: Number of layers
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

    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_type}_{trace_dir.name}_{timestamp}"

    print(f"\n{'='*60}")
    print(f"Training: {run_name}")
    print(f"{'='*60}")
    print(f"  Trace dir: {trace_dir}")
    print(f"  Model: {model_type}, hidden_dim={hidden_dim}, num_layers={num_layers}")
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

    # Build config
    config = SelectorConfig(
        name=run_name,
        model=ModelConfig(
            type=model_type,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
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
# Viewer
# =============================================================================


def start_viewer(log_dir: Path, port: int = 5000, host: str = "127.0.0.1"):
    """Start the training viewer web server."""
    try:
        from proofatlas.ml.viewer import create_app
    except ImportError as e:
        print(f"Error: Viewer dependencies not available. Install with: pip install flask")
        print(f"  {e}")
        sys.exit(1)

    app = create_app(log_dir)
    print(f"\nStarting training viewer at http://{host}:{port}")
    print(f"  Log directory: {log_dir}")
    print(f"  Press Ctrl+C to stop\n")
    app.run(host=host, port=port, debug=False)


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
    parser.add_argument(
        "--model",
        type=str,
        default="gcn",
        choices=["gcn", "mlp", "gat", "graphsage", "transformer"],
        help="Model architecture (default: gcn)",
    )

    # Model size presets
    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument("--small", action="store_true", help="Small model (32 hidden, 2 layers)")
    size_group.add_argument("--large", action="store_true", help="Large model (128 hidden, 4 layers)")
    size_group.add_argument("--overparameterized", action="store_true", help="Overparameterized (256 hidden, 6 layers)")

    # Fine-grained control
    parser.add_argument("--hidden-dim", type=int, help="Hidden dimension (overrides size preset)")
    parser.add_argument("--num-layers", type=int, help="Number of layers (overrides size preset)")

    # Training params
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs (default: 100)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")

    # Run options
    parser.add_argument("--name", type=str, help="Run name (default: auto-generated)")

    # Viewer
    parser.add_argument("--viewer", action="store_true", help="Start the training viewer only")
    parser.add_argument("--port", type=int, default=5000, help="Viewer port (default: 5000)")

    # Paths
    parser.add_argument("--log-dir", type=Path, help="Log directory (default: .logs)")

    args = parser.parse_args()

    # Set up log directory
    log_dir = args.log_dir if args.log_dir else ROOT / ".logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Viewer only mode
    if args.viewer:
        start_viewer(log_dir, port=args.port)
        return

    # Training mode requires --traces
    if not args.traces:
        parser.error("--traces is required for training (or use --viewer)")

    # Resolve trace directory
    if "/" in args.traces or "\\" in args.traces:
        trace_dir = Path(args.traces)
    else:
        trace_dir = ROOT / ".data" / "traces" / args.traces

    if not trace_dir.exists():
        print(f"Error: Trace directory not found: {trace_dir}")
        sys.exit(1)

    # Determine model size
    if args.small:
        size = MODEL_SIZES["small"]
    elif args.large:
        size = MODEL_SIZES["large"]
    elif args.overparameterized:
        size = MODEL_SIZES["overparameterized"]
    else:
        size = MODEL_SIZES["default"]

    hidden_dim = args.hidden_dim if args.hidden_dim else size["hidden_dim"]
    num_layers = args.num_layers if args.num_layers else size["num_layers"]

    # Train
    train_model(
        trace_dir=trace_dir,
        model_type=args.model,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        run_name=args.name,
        log_dir=log_dir,
    )


if __name__ == "__main__":
    main()
