"""Web visualization logging for training runs.

Writes metrics to JSON files that the web interface reads to display
training progress in real time.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any


class JSONLogger:
    """Logger that writes metrics to JSON for web visualization.

    Writes to web/data/training/{run_name}.json and updates index.json.
    The web interface reads these files to display training progress.
    """

    def __init__(self, web_data_dir: Path, run_name: str):
        """Initialize logger.

        Args:
            web_data_dir: Path to web/data directory
            run_name: Name for this training run (used as filename)
        """
        self.web_data_dir = Path(web_data_dir)
        self.training_dir = self.web_data_dir / "training"
        self.training_dir.mkdir(parents=True, exist_ok=True)

        self.run_name = run_name
        self.log_file = self.training_dir / f"{run_name}.json"
        self.index_file = self.training_dir / "index.json"

        self.metrics = {
            "generated": datetime.now().isoformat(),
            "name": run_name,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_time_seconds": None,
            "termination_reason": None,
            "best_epoch": None,
            "best_val_loss": None,
            "model": {},
            "training": {},
            "epochs": [],
            "evaluations": [],
        }
        self._save()
        self._update_index()

    def log_config(self, model_config: Dict[str, Any], training_config: Dict[str, Any]):
        """Log model and training configuration."""
        self.metrics["model"] = model_config
        self.metrics["training"] = training_config
        self._save()

    def log_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float] = None,
                  val_acc: Optional[float] = None, lr: Optional[float] = None):
        """Log metrics for one epoch."""
        self.metrics["epochs"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_mrr": None,
            "learning_rate": lr,
        })
        self._save()

    def log_evaluation(self, epoch: int, results: Dict[str, Any]):
        """Log evaluation results."""
        self.metrics["evaluations"].append({
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            **results,
        })
        self._save()

    def log_final(self, best_epoch: Optional[int], best_val_loss: Optional[float],
                  total_time: float, termination_reason: str = "completed"):
        """Log final training results."""
        self.metrics["end_time"] = datetime.now().isoformat()
        self.metrics["best_epoch"] = best_epoch
        self.metrics["best_val_loss"] = best_val_loss
        self.metrics["total_time_seconds"] = total_time
        self.metrics["termination_reason"] = termination_reason
        self._save()

    def _save(self):
        """Save metrics to JSON file."""
        with open(self.log_file, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def _update_index(self):
        """Update index.json with this run."""
        # Load existing index
        if self.index_file.exists():
            with open(self.index_file) as f:
                index = json.load(f)
        else:
            index = {"runs": []}

        # Add this run if not already present
        if self.run_name not in index["runs"]:
            index["runs"].append(self.run_name)
            index["runs"].sort()

        # Save index
        with open(self.index_file, "w") as f:
            json.dump(index, f, indent=2)
