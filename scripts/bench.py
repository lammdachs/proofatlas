#!/usr/bin/env python3
"""
Benchmark and train theorem provers.

USAGE:
    proofatlas-bench                                  # Evaluate with default preset
    proofatlas-bench --preset time_sel0               # Use specific preset
    proofatlas-bench --preset gcn --force-train       # Force retrain even if weights exist

    proofatlas-bench --track                          # Start and monitor progress
    proofatlas-bench --status                         # Check progress
    proofatlas-bench --kill                           # Stop job

LEARNED SELECTORS:
    When preset uses gcn/mlp/gat selector:
    1. If weights exist in .weights/, uses them directly
    2. If not, automatically:
       - Collects traces with age_weight selector
       - Trains the model
       - Saves weights to .weights/

OUTPUT:
    .weights/<selector>.safetensors  - Trained model weights
    .data/traces/<preset>/           - Proof traces for training
    .data/runs/<prover>/<preset>/    - Per-problem results (JSON)
"""

import argparse
import json
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional



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

    raise FileNotFoundError("Could not find proofatlas project root.")


# Job management
JOB_FILE = ".data/bench_job.json"
LOG_FILE = ".data/bench.log"
PID_FILE = ".data/bench_pids.txt"


def get_job_file(base_dir: Path) -> Path:
    return base_dir / JOB_FILE


def get_log_file(base_dir: Path) -> Path:
    return base_dir / LOG_FILE


def get_pid_file(base_dir: Path) -> Path:
    return base_dir / PID_FILE


def register_pid(base_dir: Path, pid: int):
    """Register a spawned process PID for cleanup (Unix only)."""
    if sys.platform == "win32":
        return
    pid_file = get_pid_file(base_dir)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    with open(pid_file, "a") as f:
        f.write(f"{pid}\n")


def clear_pids(base_dir: Path):
    """Clear the PID tracking file."""
    pid_file = get_pid_file(base_dir)
    if pid_file.exists():
        pid_file.unlink()


def kill_tracked_pids(base_dir: Path) -> int:
    """Kill all tracked PIDs and clear the file (Unix only)."""
    if sys.platform == "win32":
        return 0
    pid_file = get_pid_file(base_dir)
    if not pid_file.exists():
        return 0

    killed = 0
    try:
        pids = pid_file.read_text().splitlines()
        for line in pids:
            if not line:
                continue
            try:
                os.kill(int(line), signal.SIGKILL)
                killed += 1
            except (ValueError, OSError, ProcessLookupError):
                pass
        pid_file.unlink(missing_ok=True)
    except (IOError, OSError):
        pass

    return killed


def is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def get_job_status(base_dir: Path) -> Optional[dict]:
    job_file = get_job_file(base_dir)
    if not job_file.exists():
        return None

    try:
        with open(job_file) as f:
            job = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    if not is_process_running(job["pid"]):
        return None

    log_file = Path(job.get("log_file", ""))
    if log_file.exists():
        try:
            with open(log_file) as f:
                lines = f.readlines()
            for line in reversed(lines):
                if line.startswith("PROGRESS:"):
                    parts = line.strip().split(":")
                    if len(parts) >= 5:
                        job["current"] = int(parts[1])
                        job["total"] = int(parts[2])
                        job["proofs"] = int(parts[3])
                        job["timeout"] = int(parts[4])
                    break
                elif line.startswith("TRAIN:"):
                    parts = line.strip().split(":")
                    if len(parts) >= 4:
                        job["train_epoch"] = int(parts[1])
                        job["train_total"] = int(parts[2])
                        job["train_loss"] = float(parts[3])
                    break
        except IOError:
            pass

    return job


def save_job_status(base_dir: Path, pid: int, args: list, num_configs: int = 1):
    job_file = get_job_file(base_dir)
    job_file.parent.mkdir(parents=True, exist_ok=True)

    job = {
        "pid": pid,
        "args": args,
        "log_file": str(get_log_file(base_dir)),
        "start_time": datetime.now().isoformat(),
        "num_configs": num_configs,
    }

    with open(job_file, "w") as f:
        json.dump(job, f, indent=2)


def clear_job_status(base_dir: Path):
    job_file = get_job_file(base_dir)
    if job_file.exists():
        job_file.unlink()


def kill_job(base_dir: Path) -> bool:
    import subprocess

    job = get_job_status(base_dir)

    # Step 1: Clear job status to stop spawning new processes
    clear_job_status(base_dir)

    # Step 2: Kill the main daemon process
    if job:
        try:
            os.kill(job['pid'], signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass

    # Step 3: Kill tracked PIDs and worker processes (Unix only)
    if sys.platform != "win32":
        # Kill any proofatlas-bench worker processes
        subprocess.run(["pkill", "-9", "-f", "proofatlas-bench.*--preset"], capture_output=True)

        # Kill tracked prover PIDs
        max_iterations = 10
        for _ in range(max_iterations):
            killed = kill_tracked_pids(base_dir)
            if killed == 0:
                break
            time.sleep(0.2)

        # Kill any remaining prover processes from this project
        subprocess.run(["pkill", "-9", "-f", str(base_dir / ".vampire")], capture_output=True)
        subprocess.run(["pkill", "-9", "-f", str(base_dir / ".spass")], capture_output=True)

    return job is not None


def format_job_status(job: dict) -> str:
    if not job:
        return "No job running"

    start = datetime.fromisoformat(job["start_time"])
    elapsed = datetime.now() - start
    hours = elapsed.seconds // 3600
    minutes = (elapsed.seconds % 3600) // 60

    parts = [f"[{hours}h{minutes:02d}m]"]

    if "train_epoch" in job:
        parts.append(f"train {job['train_epoch']}/{job['train_total']} loss={job['train_loss']:.4f}")
    elif "current" in job:
        parts.append(f"eval {job['current']}/{job['total']} +{job['proofs']} T{job['timeout']}")

    return " | ".join(parts)


def print_job_status(base_dir: Path):
    job = get_job_status(base_dir)
    if not job:
        print("No job currently running.")
        return

    start = datetime.fromisoformat(job["start_time"])
    elapsed = datetime.now() - start
    hours = elapsed.seconds // 3600
    minutes = (elapsed.seconds % 3600) // 60

    print(f"Job running (PID: {job['pid']})")
    print(f"  Started: {start.strftime('%Y-%m-%d %H:%M:%S')} ({hours}h {minutes}m ago)")

    # Parse log file for progress
    log_file = Path(job.get("log_file", ""))
    if log_file.exists():
        last_progress = None
        last_train = None
        last_config = None  # CONFIG:name:index:total
        phase = None

        with open(log_file) as f:
            for line in f:
                if line.startswith("PROGRESS:"):
                    last_progress = line.strip()
                    phase = "eval"
                elif line.startswith("TRAIN:"):
                    last_train = line.strip()
                    phase = "train"
                elif line.startswith("CONFIG:"):
                    last_config = line.strip()
                elif "Collecting traces" in line:
                    phase = "collect"
                elif "Training" in line and "problems" in line:
                    phase = "train"

        # Check if we're in trace collection
        is_collecting = False
        with open(log_file) as f:
            content = f.read()
            if "Collecting traces" in content:
                eval_count = content.count("Evaluating")
                is_collecting = eval_count == 1

        # Show current config with progress
        if last_config:
            parts = last_config.split(":")
            if len(parts) >= 4:
                config_name, config_idx, config_total = parts[1], parts[2], parts[3]
                # config_name is "prover/preset" format
                if "/" in config_name:
                    prover, preset = config_name.split("/", 1)
                    print(f"  Prover:  {prover}")
                    print(f"  Config:  {preset} ({config_idx}/{config_total})")
                else:
                    print(f"  Config:  {config_name} ({config_idx}/{config_total})")

        if phase == "train" and last_train:
            parts = last_train.split(":")
            if len(parts) >= 4:
                epoch, max_epochs, loss = parts[1], parts[2], parts[3]
                print(f"  Training: epoch {epoch}/{max_epochs}, loss={float(loss):.4f}")
        elif is_collecting and last_progress:
            parts = last_progress.split(":")
            if len(parts) >= 5:
                current, total, proofs, timeout = parts[1:5]
                print(f"  Collecting traces: {current}/{total}, +{proofs} proofs")
        elif phase == "eval" and last_progress:
            parts = last_progress.split(":")
            if len(parts) >= 5:
                current, total, proofs, timeout = parts[1:5]
                print(f"  Evaluating: {current}/{total}, +{proofs} proofs, T{timeout} timeout")
        else:
            print("  Starting...")
    else:
        print("  Starting...")

    print(f"\nTo stop: proofatlas-bench --kill")


def track_job(base_dir: Path, poll_interval: float = 1.0):
    last_status = ""
    while True:
        job = get_job_status(base_dir)
        if not job:
            sys.stdout.write("\r" + " " * len(last_status) + "\r")
            sys.stdout.flush()
            print("Job completed.")
            break

        status = format_job_status(job)
        sys.stdout.write("\r" + " " * len(last_status) + "\r")
        sys.stdout.write(status)
        sys.stdout.flush()
        last_status = status

        time.sleep(poll_interval)


@dataclass
class BenchResult:
    problem: str
    status: str  # "proof", "saturated", "timeout", "error"
    time_s: float


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return json.load(f)


def get_problems(base_dir: Path, tptp_config: dict, problem_set_name: str) -> list[Path]:
    """Get list of problem files matching the problem set filters."""
    problem_sets = tptp_config.get("problem_sets", {})
    if problem_set_name not in problem_sets:
        available = list(problem_sets.keys())
        raise ValueError(f"Unknown problem set: {problem_set_name}. Available: {available}")

    filters = problem_sets[problem_set_name]
    problems_dir = base_dir / tptp_config["paths"]["problems"]

    if not problems_dir.exists():
        raise FileNotFoundError(f"TPTP problems not found: {problems_dir}")

    # Load metadata
    metadata_path = base_dir / ".data" / "problem_metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            data = json.load(f)
            problems_list = data.get("problems", data) if isinstance(data, dict) else data
            metadata = {p["path"]: p for p in problems_list}

    # Check for explicit problem list (inline or from file)
    explicit_problems = filters.get("problems")
    if "problems_file" in filters:
        problems_file = base_dir / filters["problems_file"]
        if problems_file.exists():
            with open(problems_file) as f:
                explicit_problems = [line.strip() for line in f if line.strip()]
    if explicit_problems:
        explicit_set = set(explicit_problems)

    problems = []
    for domain_dir in sorted(problems_dir.iterdir()):
        if not domain_dir.is_dir():
            continue

        domain = domain_dir.name
        if "domains" in filters and filters["domains"] and domain not in filters["domains"]:
            continue
        if "exclude_domains" in filters and domain in filters.get("exclude_domains", []):
            continue

        for problem_file in sorted(domain_dir.glob("*.p")):
            rel_path = str(problem_file.relative_to(problems_dir))
            meta = metadata.get(rel_path, {})

            # Filter by explicit problem names (without .p extension)
            if explicit_problems:
                problem_name = problem_file.stem
                if problem_name not in explicit_set:
                    continue

            if "status" in filters and filters["status"]:
                if meta.get("status") not in filters["status"]:
                    continue
            if "format" in filters and filters["format"]:
                if meta.get("format") not in filters["format"]:
                    continue
            if "max_rating" in filters and filters["max_rating"] is not None:
                if meta.get("rating", 1.0) > filters["max_rating"]:
                    continue
            if "max_clauses" in filters and filters["max_clauses"] is not None:
                if meta.get("num_clauses", 0) > filters["max_clauses"]:
                    continue
            if "max_term_depth" in filters and filters["max_term_depth"] is not None:
                if meta.get("max_term_depth", 0) > filters["max_term_depth"]:
                    continue
            if "max_clause_size" in filters and filters["max_clause_size"] is not None:
                if meta.get("max_clause_size", 0) > filters["max_clause_size"]:
                    continue
            if "has_equality" in filters and filters["has_equality"] is not None:
                if meta.get("has_equality") != filters["has_equality"]:
                    continue
            if "is_unit_only" in filters and filters["is_unit_only"] is not None:
                if meta.get("is_unit_only") != filters["is_unit_only"]:
                    continue

            problems.append(problem_file)

    return problems


# Weights management

def find_weights(base_dir: Path, selector: str) -> Optional[Path]:
    """Find weights file for a learned selector."""
    weights_dir = base_dir / ".weights"
    if not weights_dir.exists():
        return None

    # Check exact name first
    exact = weights_dir / f"{selector}.safetensors"
    if exact.exists():
        return exact

    # Check for iteration variants (e.g., gcn_iter_5.safetensors)
    prefix = f"{selector}_iter_"
    latest_iter = None
    latest_path = None

    for f in weights_dir.glob(f"{prefix}*.safetensors"):
        try:
            iter_num = int(f.stem[len(prefix):])
            if latest_iter is None or iter_num > latest_iter:
                latest_iter = iter_num
                latest_path = f
        except ValueError:
            continue

    return latest_path


def is_learned_selector(selector_config: dict) -> bool:
    """Check if selector requires trained weights (has embedding or model field)."""
    return "embedding" in selector_config or "model" in selector_config


# Trace collection and training

def save_trace(base_dir: Path, preset: str, problem: str, trace_json: str):
    """Save proof trace for training in structured JSON format.

    Args:
        base_dir: Project root directory
        preset: Preset name for trace subdirectory
        problem: Problem file name
        trace_json: Structured JSON string from extract_structured_trace()
    """
    try:
        traces_dir = base_dir / ".data" / "traces" / preset
        traces_dir.mkdir(parents=True, exist_ok=True)
        problem_name = Path(problem).stem
        json_path = traces_dir / f"{problem_name}.json"
        with open(json_path, "w") as f:
            f.write(trace_json)
    except Exception:
        pass


def load_traces(base_dir: Path, preset: str, problem_names: set[str] = None):
    """Load traces for training, optionally filtered by problem names.

    Args:
        base_dir: Project root directory
        preset: Trace preset name (subdirectory in .data/traces/)
        problem_names: Optional set of problem names to include. If None, loads all.

    Returns:
        Dict with 'problems' list and 'num_problems' count.
    """
    sys.path.insert(0, str(base_dir / "python"))
    from proofatlas.ml.structured import clause_to_graph

    traces_dir = base_dir / ".data" / "traces" / preset
    if not traces_dir.exists():
        return {"problems": [], "num_problems": 0}

    problems = []
    for trace_file in sorted(traces_dir.glob("*.json")):
        # Filter by problem set if specified
        if problem_names is not None and trace_file.stem not in problem_names:
            continue

        try:
            with open(trace_file) as f:
                trace = json.load(f)
        except Exception:
            continue

        if not trace.get("proof_found") or not trace.get("clauses"):
            continue

        # Convert structured clauses to graph tensors
        clauses = trace["clauses"]
        max_age = len(clauses)
        graphs = [clause_to_graph(c, max_age) for c in clauses]
        labels = [c.get("label", 0) for c in clauses]

        problems.append({
            "name": trace_file.stem,
            "graphs": graphs,
            "labels": labels,
        })

    return {"problems": problems, "num_problems": len(problems)}


def run_training(base_dir: Path, preset_name: str, preset: dict, data: dict, log_file,
                 init_weights: Path = None) -> Path:
    """Train a model and return the weights path.

    Args:
        init_weights: Optional path to weights file to initialize from.
                     If provided, continues training from these weights.
    """
    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(base_dir / "python"))
    from proofatlas.ml.training import ClauseDataset, collate_clause_batch
    from proofatlas.selectors import create_model

    # Load configs
    with open(base_dir / "configs" / "embeddings.json") as f:
        embeddings_config = json.load(f)
    with open(base_dir / "configs" / "scorers.json") as f:
        scorers_config = json.load(f)
    with open(base_dir / "configs" / "training.json") as f:
        training_config = json.load(f)

    # Get model and training config from preset
    embedding_name = preset.get("embedding", preset.get("model"))  # fallback to "model" for compat
    scorer_name = preset.get("scorer", "mlp")
    training_name = preset.get("training", "standard")

    # Get embedding architecture
    embedding_arch = embeddings_config["architectures"].get(embedding_name)
    if not embedding_arch:
        raise ValueError(f"Unknown embedding: {embedding_name}")

    # Get scorer architecture
    scorer_arch = scorers_config["architectures"].get(scorer_name)
    if not scorer_arch:
        raise ValueError(f"Unknown scorer: {scorer_name}")

    # Get training config
    training_defaults = training_config.get("defaults", {})
    training_overrides = training_config.get("configs", {}).get(training_name, {})

    # Merge configs
    config = {**training_defaults, **training_overrides}
    config["embedding"] = embedding_arch
    config["scorer"] = scorer_arch
    config["input_dim"] = embeddings_config.get("input_dim", 8)

    problems = data["problems"]
    if not problems:
        raise ValueError("No training data")

    # Problem-level split
    val_ratio = config.get("val_ratio", 0.0)
    random.seed(42)
    problem_indices = list(range(len(problems)))
    random.shuffle(problem_indices)

    if val_ratio > 0:
        val_count = max(1, int(len(problems) * val_ratio))
        train_indices = problem_indices[val_count:]
        val_indices = problem_indices[:val_count]
    else:
        train_indices = problem_indices
        val_indices = []

    def make_dataset(prob_indices):
        all_graphs, all_labels = [], []
        for idx in prob_indices:
            p = problems[idx]
            all_graphs.extend(p["graphs"])
            all_labels.extend(p["labels"])
        if not all_graphs:
            return None
        return ClauseDataset(
            node_features=[g["x"] for g in all_graphs],
            edge_indices=[g["edge_index"] for g in all_graphs],
            labels=all_labels,
        )

    train_ds = make_dataset(train_indices)
    val_ds = make_dataset(val_indices) if val_indices else None

    if not train_ds:
        raise ValueError("No training examples")

    print(f"Training {preset_name}: {len(train_indices)} problems, {len(train_ds)} examples")

    # Create model
    model_type = config.get("type", "gcn")
    model = create_model(
        model_type=model_type,
        node_feature_dim=config.get("input_dim", 13),
        hidden_dim=config.get("hidden_dim", 64),
        num_layers=config.get("num_layers", 3),
        num_heads=config.get("num_heads", 4),
        dropout=config.get("dropout", 0.1),
    )

    needs_adj = model_type in ["gcn", "gat", "graphsage", "gnn_transformer"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Initialize from existing weights if provided
    if init_weights and init_weights.exists():
        print(f"Initializing from {init_weights}")
        from safetensors.torch import load_file
        state_dict = load_file(init_weights)
        model.load_state_dict(state_dict)
        model = model.to(device)

    batch_size = config.get("batch_size", 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_clause_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_clause_batch) if val_ds else None

    # Optimizer
    optimizer_type = config.get("optimizer", "adamw").lower()
    lr = config.get("learning_rate", 0.001)
    weight_decay = config.get("weight_decay", 1e-5)

    if optimizer_type == "adamw":
        betas = tuple(config.get("betas", [0.9, 0.999]))
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    elif optimizer_type == "adam":
        betas = tuple(config.get("betas", [0.9, 0.999]))
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    elif optimizer_type == "sgd":
        momentum = config.get("momentum", 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    max_epochs = config.get("max_epochs", 100)

    # Margin ranking loss for pairwise learning
    margin = config.get("margin", 0.1)
    ranking_loss = torch.nn.MarginRankingLoss(margin=margin)

    def compute_pairwise_loss(scores, labels):
        """Compute pairwise margin ranking loss between positive and negative examples."""
        pos_mask = labels == 1
        neg_mask = labels == 0

        if not pos_mask.any() or not neg_mask.any():
            # Fallback to BCE if no pairs available
            return F.binary_cross_entropy_with_logits(scores, labels.float())

        pos_scores = scores[pos_mask]
        neg_scores = scores[neg_mask]

        # Sample pairs: for each positive, sample a random negative
        n_pos = pos_scores.size(0)
        n_neg = neg_scores.size(0)

        # Random pairing: sample n_pos negatives (with replacement if needed)
        neg_indices = torch.randint(0, n_neg, (n_pos,), device=scores.device)
        neg_sampled = neg_scores[neg_indices]

        # Target: +1 means first input should be ranked higher than second
        target = torch.ones(n_pos, device=scores.device)
        return ranking_loss(pos_scores, neg_sampled, target)

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            x = batch["node_features"].to(device)
            adj = batch["adj"].to(device)
            pool = batch["pool_matrix"].to(device)
            labels = batch["labels"].to(device)

            scores = model(x, adj, pool) if needs_adj else model(x, pool)
            loss = compute_pairwise_loss(scores, labels)
            loss.backward()

            if config.get("gradient_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        val_loss = 0
        if val_loader:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["node_features"].to(device)
                    adj = batch["adj"].to(device)
                    pool = batch["pool_matrix"].to(device)
                    labels = batch["labels"].to(device)
                    scores = model(x, adj, pool) if needs_adj else model(x, pool)
                    val_loss += compute_pairwise_loss(scores, labels).item()
            val_loss /= len(val_loader)

        if val_loader:
            print(f"Epoch {epoch}/{max_epochs} | train={train_loss:.4f} | val={val_loss:.4f}")
        else:
            print(f"Epoch {epoch}/{max_epochs} | train={train_loss:.4f}")

        log_file.write(f"TRAIN:{epoch}:{max_epochs}:{train_loss:.6f}\n")
        log_file.flush()

    # Save weights
    from safetensors.torch import save_file
    weights_dir = base_dir / ".weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / f"{preset_name}.safetensors"

    metadata = {
        "model_type": model_type,
        "hidden_dim": str(config.get("hidden_dim", 64)),
        "num_layers": str(config.get("num_layers", 3)),
        "num_heads": str(config.get("num_heads", 4)),
        "input_dim": str(config.get("input_dim", 13)),
    }
    save_file(model.state_dict(), weights_path, metadata=metadata)
    print(f"Weights saved: {weights_path}")

    return weights_path


# Prover execution

def _run_proofatlas_inner(problem: Path, base_dir: Path, preset: dict, tptp_root: Path,
                          weights_path: str = None, collect_trace: bool = False,
                          trace_preset: str = None) -> BenchResult:
    """Inner function that actually runs ProofAtlas (called in subprocess)."""
    from proofatlas import ProofState

    timeout = preset.get("timeout", 10)

    try:
        with open(problem) as f:
            content = f.read()
    except Exception:
        return BenchResult(problem=problem.name, status="error", time_s=0)

    # Start timer before parsing (CNF conversion counts against timeout)
    start = time.time()

    state = ProofState()
    try:
        # Pass timeout to parsing to prevent CNF conversion hangs
        state.add_clauses_from_tptp(content, str(tptp_root), timeout)
    except Exception as e:
        elapsed = time.time() - start
        # Check if this was a timeout during CNF conversion
        if "timed out" in str(e).lower():
            return BenchResult(problem=problem.name, status="timeout", time_s=elapsed)
        return BenchResult(problem=problem.name, status="error", time_s=elapsed)

    literal_selection = str(preset.get("literal_selection", 21))
    state.set_literal_selection(literal_selection)

    max_iterations = preset.get("max_iterations", 0)  # 0 means no limit
    max_clause_memory_mb = preset.get("max_clause_memory_mb")  # None means no limit
    is_learned = "embedding" in preset or "model" in preset or "selector" in preset
    age_weight_ratio = preset.get("age_weight_ratio", 0.167)
    # Use selector key directly if present, otherwise use embedding/model for learned selectors
    selector = preset.get("selector", preset.get("embedding", preset.get("model", "age_weight")) if is_learned else "age_weight")

    # Remaining time after parsing
    elapsed_parsing = time.time() - start
    remaining_timeout = max(0.1, timeout - elapsed_parsing)

    try:
        proof_found, status = state.run_saturation(
            max_iterations,
            float(remaining_timeout),
            float(age_weight_ratio) if not is_learned else None,
            selector,
            weights_path,
            max_clause_memory_mb,
        )
    except Exception as e:
        return BenchResult(problem=problem.name, status="error", time_s=time.time() - start)

    elapsed = time.time() - start

    # Map status to benchmark format (resource_limit -> timeout for compatibility)
    if status == "resource_limit":
        status = "timeout"

    # Collect trace for training
    if collect_trace and proof_found and trace_preset:
        try:
            trace_json = state.extract_structured_trace(elapsed)
            save_trace(base_dir, trace_preset, problem.name, trace_json)
        except Exception:
            pass

    return BenchResult(problem=problem.name, status=status, time_s=elapsed)


def _worker_process(problem_str, base_dir_str, preset, tptp_root_str, weights_path,
                    collect_trace, trace_preset, result_queue):
    """Worker function that runs in subprocess and sends result via queue."""
    # Reset signal handlers inherited from parent daemon process.
    # Without this, if the worker is killed (e.g., timeout), it would run
    # the parent's signal handler which deletes the job file!
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGQUIT, signal.SIG_DFL)

    try:
        result = _run_proofatlas_inner(
            Path(problem_str), Path(base_dir_str), preset, Path(tptp_root_str),
            weights_path, collect_trace, trace_preset
        )
        result_queue.put((result.status, result.time_s))
    except Exception as e:
        result_queue.put(("error", 0))


def run_proofatlas(problem: Path, base_dir: Path, preset: dict, tptp_root: Path,
                   weights_path: str = None, collect_trace: bool = False,
                   trace_preset: str = None) -> BenchResult:
    """Run ProofAtlas on a problem in a subprocess.

    Uses multiprocessing to isolate crashes (e.g., stack overflow on deeply
    nested terms) so they don't take down the entire benchmark process.
    """
    import multiprocessing

    timeout = preset.get("timeout", 10)
    process_timeout = timeout + 10  # Extra time for overhead

    result_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_worker_process,
        args=(str(problem), str(base_dir), preset, str(tptp_root),
              weights_path, collect_trace, trace_preset, result_queue)
    )

    start = time.time()
    proc.start()
    proc.join(timeout=process_timeout)
    elapsed = time.time() - start

    if proc.is_alive():
        # Process hung - kill it
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return BenchResult(problem=problem.name, status="timeout", time_s=elapsed)

    if proc.exitcode != 0:
        # Process crashed (e.g., stack overflow gives exit code 134)
        return BenchResult(problem=problem.name, status="error", time_s=elapsed)

    try:
        status, elapsed_inner = result_queue.get_nowait()
        return BenchResult(problem=problem.name, status=status, time_s=elapsed_inner)
    except Exception:
        return BenchResult(problem=problem.name, status="error", time_s=elapsed)


def run_vampire(problem: Path, base_dir: Path, preset: dict, binary: Path, tptp_root: Path) -> BenchResult:
    """Run Vampire on a problem."""
    import subprocess

    timeout = preset.get("time_limit", 10)
    selection = preset.get("selection", 21)
    avatar = preset.get("avatar", "off")
    memory_limit = preset.get("memory_limit")
    activation_limit = preset.get("activation_limit")

    cmd = [
        str(binary),
        "--include", str(tptp_root),
        "--time_limit", str(timeout),
        "--selection", str(selection),
        "--avatar", avatar,
    ]

    if memory_limit is not None:
        cmd.extend(["--memory_limit", str(memory_limit)])

    if activation_limit is not None:
        cmd.extend(["--activation_limit", str(activation_limit)])

    cmd.append(str(problem))

    start = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        register_pid(base_dir, proc.pid)
        try:
            # timeout=0 means no time limit, use None for communicate
            proc_timeout = None if timeout == 0 else timeout + 5
            stdout, stderr = proc.communicate(timeout=proc_timeout)
            output = stdout + stderr
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return BenchResult(problem=problem.name, status="timeout", time_s=timeout)
    except Exception:
        return BenchResult(problem=problem.name, status="error", time_s=time.time() - start)

    elapsed = time.time() - start

    # Parse Vampire output
    if "Refutation found" in output or "Termination reason: Refutation" in output:
        status = "proof"
    elif "Termination reason: Satisfiable" in output:
        status = "saturated"
    elif "Termination reason: Time limit" in output or elapsed >= timeout:
        status = "timeout"
    elif "Termination reason: Memory limit" in output:
        status = "timeout"  # Memory limit treated as resource limit
    elif "Termination reason: Activation limit" in output:
        status = "timeout"  # Activation limit treated as resource limit
    else:
        status = "error"

    return BenchResult(problem=problem.name, status=status, time_s=elapsed)


def run_spass(problem: Path, base_dir: Path, preset: dict, binary: Path, tptp_root: Path) -> BenchResult:
    """Run SPASS on a problem."""
    import subprocess

    timeout = preset.get("TimeLimit", 10)
    selection = preset.get("Select", 1)
    memory = preset.get("Memory")
    loops = preset.get("Loops")

    # SPASS requires TPTP format with -TPTP flag
    cmd = [
        str(binary),
        "-TPTP",
        f"-TimeLimit={timeout}",
        f"-Select={selection}",
    ]

    if memory is not None:
        cmd.append(f"-Memory={memory}")

    if loops is not None:
        cmd.append(f"-Loops={loops}")

    cmd.append(str(problem))

    start = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "TPTP": str(tptp_root)},
        )
        register_pid(base_dir, proc.pid)
        try:
            # timeout=0 means no time limit, use None for communicate
            proc_timeout = None if timeout == 0 else timeout + 5
            stdout, stderr = proc.communicate(timeout=proc_timeout)
            output = stdout + stderr
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return BenchResult(problem=problem.name, status="timeout", time_s=timeout)
    except Exception:
        return BenchResult(problem=problem.name, status="error", time_s=time.time() - start)

    elapsed = time.time() - start

    # Parse SPASS output
    # SPASS says "SPASS beiseite: Proof found." for proofs
    if "Proof found" in output:
        status = "proof"
    elif "Completion found" in output:
        status = "saturated"
    elif "Maximal number of loops exceeded" in output:
        status = "timeout"  # Loop limit treated as resource limit
    elif elapsed >= timeout or "SPASS broke down" in output:
        status = "timeout"
    else:
        status = "error"

    return BenchResult(problem=problem.name, status=status, time_s=elapsed)


def get_run_result_path(base_dir: Path, prover: str, preset_name: str, problem: Path) -> Path:
    """Get path to result file for a problem."""
    return base_dir / ".data" / "runs" / prover / preset_name / f"{problem.stem}.json"


def load_run_result(base_dir: Path, prover: str, preset_name: str, problem: Path) -> Optional[BenchResult]:
    """Load existing result if available."""
    result_file = get_run_result_path(base_dir, prover, preset_name, problem)
    if not result_file.exists():
        return None
    try:
        with open(result_file) as f:
            data = json.load(f)
        return BenchResult(
            problem=data["problem"],
            status=data["status"],
            time_s=data["time_s"],
        )
    except (json.JSONDecodeError, KeyError, IOError):
        return None


def save_run_result(base_dir: Path, prover: str, preset_name: str, result: BenchResult):
    """Save individual result to .data/runs/<prover>/<preset>/<problem>.json"""
    runs_dir = base_dir / ".data" / "runs" / prover / preset_name
    runs_dir.mkdir(parents=True, exist_ok=True)

    problem_name = Path(result.problem).stem
    result_file = runs_dir / f"{problem_name}.json"

    data = {
        "problem": result.problem,
        "status": result.status,
        "time_s": result.time_s,
        "prover": prover,
        "preset": preset_name,
        "timestamp": datetime.now().isoformat(),
    }

    with open(result_file, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def export_benchmark_progress(base_dir: Path, prover: str, preset_name: str,
                               stats: dict, completed: int, total: int):
    """Export current benchmark progress to web/data/benchmarks/<prover>_<preset>.json"""
    output_dir = base_dir / "web" / "data" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{prover}_{preset_name}.json"

    data = {
        "generated": datetime.now().isoformat(),
        "prover": prover,
        "preset": preset_name,
        "completed": completed,
        "total": total,
        "progress_pct": 100 * completed / total if total else 0,
        "stats": {
            "proof": stats.get("proof", 0),
            "saturated": stats.get("saturated", 0),
            "timeout": stats.get("timeout", 0),
            "error": stats.get("error", 0),
        },
        "proof_rate": 100 * stats.get("proof", 0) / completed if completed else 0,
    }

    try:
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        # Update index file
        _update_benchmark_index(output_dir)
    except Exception:
        pass  # Don't fail benchmark if export fails


def _update_benchmark_index(output_dir: Path):
    """Update index.json with list of all benchmark runs."""
    index_file = output_dir / "index.json"
    runs = []

    for f in sorted(output_dir.glob("*.json")):
        if f.name == "index.json":
            continue
        runs.append(f.stem)

    index = {
        "generated": datetime.now().isoformat(),
        "runs": runs,
    }

    with open(index_file, "w") as f:
        json.dump(index, f, indent=2)


def _run_single_problem(args):
    """Worker function for parallel execution."""
    problem, base_dir, prover, preset, tptp_root, weights_path, collect_trace, trace_preset, binary, preset_name, rerun = args

    try:
        # Check if already evaluated (skip unless --rerun)
        existing = load_run_result(base_dir, prover, preset_name, problem)
        if existing and not rerun:
            return ("skip", existing)

        if prover == "proofatlas":
            result = run_proofatlas(
                problem, base_dir, preset, tptp_root,
                weights_path=weights_path, collect_trace=collect_trace,
                trace_preset=trace_preset,
            )
        elif prover == "vampire":
            result = run_vampire(problem, base_dir, preset, binary, tptp_root)
        elif prover == "spass":
            result = run_spass(problem, base_dir, preset, binary, tptp_root)
        else:
            result = BenchResult(problem=problem.name, status="error", time_s=0)

        # Save individual result to .data/runs/
        save_run_result(base_dir, prover, preset_name, result)
        return ("run", result)
    except Exception as e:
        return ("error", BenchResult(problem=problem.name, status="error", time_s=0))


def run_evaluation(base_dir: Path, problems: list[Path], tptp_root: Path,
                   prover: str, preset: dict, log_file,
                   preset_name: str = None, weights_path: str = None,
                   binary: Path = None, trace_preset: str = None,
                   rerun: bool = False, n_jobs: int = 1):
    """Run evaluation on problems with the specified prover."""
    stats = {"proof": 0, "saturated": 0, "timeout": 0, "error": 0, "skip": 0}

    if prover == "proofatlas":
        selector_type = preset.get("selector", preset.get("embedding", preset.get("model", "age_weight")))
        print(f"\nEvaluating {len(problems)} problems with {selector_type}" + (f" ({n_jobs} jobs)" if n_jobs > 1 else ""))
        if weights_path:
            print(f"Weights: {weights_path}")
    else:
        print(f"\nEvaluating {len(problems)} problems" + (f" ({n_jobs} jobs)" if n_jobs > 1 else ""))

    # Always collect traces for proofatlas
    collect_trace = (prover == "proofatlas")

    # Prepare work items
    work_items = [
        (problem, base_dir, prover, preset, tptp_root, weights_path, collect_trace, trace_preset, binary, preset_name, rerun)
        for problem in problems
    ]

    if n_jobs > 1:
        # Parallel execution
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(_run_single_problem, item): i for i, item in enumerate(work_items)}
            completed = 0

            for future in as_completed(futures):
                completed += 1
                try:
                    status, result = future.result()
                    if status == "skip":
                        stats[result.status] = stats.get(result.status, 0) + 1
                        stats["skip"] += 1
                        symbol = {"proof": "+", "saturated": "~", "timeout": "T", "error": "!"}[result.status]
                        print(f"[{completed}/{len(problems)}] S{symbol} {result.problem} (cached)")
                    else:
                        stats[result.status] = stats.get(result.status, 0) + 1
                        symbol = {"proof": "+", "saturated": "~", "timeout": "T", "error": "!"}[result.status]
                        print(f"[{completed}/{len(problems)}] {symbol} {result.problem} ({result.time_s:.2f}s)")

                    log_file.write(f"PROGRESS:{completed}:{len(problems)}:{stats['proof']}:{stats['timeout']}\n")
                    log_file.flush()
                    sys.stdout.flush()
                    export_benchmark_progress(base_dir, prover, preset_name, stats, completed, len(problems))
                except Exception as e:
                    print(f"ERROR: {e}")
                    stats["error"] += 1
    else:
        # Sequential execution
        for i, item in enumerate(work_items, 1):
            # Periodic garbage collection to prevent OOM
            if i % 100 == 0:
                import gc
                gc.collect()

            try:
                status, result = _run_single_problem(item)
                if status == "skip":
                    stats[result.status] = stats.get(result.status, 0) + 1
                    stats["skip"] += 1
                    symbol = {"proof": "+", "saturated": "~", "timeout": "T", "error": "!"}[result.status]
                    print(f"[{i}/{len(problems)}] S{symbol} {result.problem} (cached)")
                else:
                    stats[result.status] = stats.get(result.status, 0) + 1
                    symbol = {"proof": "+", "saturated": "~", "timeout": "T", "error": "!"}[result.status]
                    print(f"[{i}/{len(problems)}] {symbol} {result.problem} ({result.time_s:.2f}s)")

                log_file.write(f"PROGRESS:{i}:{len(problems)}:{stats['proof']}:{stats['timeout']}\n")
                log_file.flush()
                sys.stdout.flush()
                export_benchmark_progress(base_dir, prover, preset_name, stats, i, len(problems))
            except Exception as e:
                print(f"ERROR processing {item[0].name}: {e}")
                sys.stdout.flush()
                import traceback
                traceback.print_exc()
                sys.stdout.flush()

    # Print summary (individual results saved to .data/runs/)
    # Note: skip count is separate (skipped problems are also counted in their status)
    total = len(problems)
    proof_rate = 100 * stats["proof"] / total if total else 0

    print(f"\n{'='*60}")
    skip_str = f" S{stats['skip']}" if stats["skip"] else ""
    print(f"Results: +{stats['proof']} ~{stats['saturated']} T{stats['timeout']}{skip_str} ({proof_rate:.1f}% proofs)")

    return stats


def get_available_provers(base_dir: Path) -> dict:
    """Get available provers and their configs."""
    provers = {}

    # proofatlas is always available (Python bindings)
    proofatlas_config_path = base_dir / "configs" / "proofatlas.json"
    if proofatlas_config_path.exists():
        provers["proofatlas"] = {
            "config": load_config(proofatlas_config_path),
            "binary": None,  # Uses Python bindings
        }

    # Check for vampire
    vampire_config_path = base_dir / "configs" / "vampire.json"
    if vampire_config_path.exists():
        vampire_config = load_config(vampire_config_path)
        vampire_binary = base_dir / vampire_config["paths"]["binary"]
        if vampire_binary.exists():
            provers["vampire"] = {
                "config": vampire_config,
                "binary": vampire_binary,
            }

    # Check for spass
    spass_config_path = base_dir / "configs" / "spass.json"
    if spass_config_path.exists():
        spass_config = load_config(spass_config_path)
        spass_binary = base_dir / spass_config["paths"]["binary"]
        if spass_binary.exists():
            provers["spass"] = {
                "config": spass_config,
                "binary": spass_binary,
            }

    return provers


def main():
    parser = argparse.ArgumentParser(description="Benchmark and train theorem provers")
    parser.add_argument("--prover",
                       help="Prover to run (default: all available)")
    parser.add_argument("--preset", nargs="*",
                       help="Solver preset(s) (default: all)")
    parser.add_argument("--problem-set",
                       help="Problem set from tptp.json (default: from config)")
    parser.add_argument("--force-train", action="store_true",
                       help="Force retrain even if weights exist")
    parser.add_argument("--base-only", action="store_true",
                       help="Only run base configs (skip learned selectors)")
    parser.add_argument("--rerun", action="store_true",
                       help="Re-evaluate problems even if cached results exist")
    parser.add_argument("--trace-preset",
                       help="Preset name for trace collection (default: solver preset)")
    parser.add_argument("--n-jobs", type=int, default=1,
                       help="Number of parallel jobs (default: 1)")

    # Job management
    parser.add_argument("--track", action="store_true",
                       help="Track progress")
    parser.add_argument("--status", action="store_true",
                       help="Check job status")
    parser.add_argument("--kill", action="store_true",
                       help="Stop running job")

    args = parser.parse_args()
    base_dir = find_project_root()

    # Job management
    if args.status:
        print_job_status(base_dir)
        return

    if args.track:
        job = get_job_status(base_dir)
        if job:
            track_job(base_dir)
            return

    if args.kill:
        had_job = kill_job(base_dir)
        if had_job:
            print("Job killed.")
        else:
            print("No job file found. Killing any orphaned processes...")
        return

    existing = get_job_status(base_dir)
    if existing:
        print(f"Error: Job already running (PID: {existing['pid']})")
        print("Use --status, --track, or --kill")
        sys.exit(1)

    # Load configs
    tptp_config = load_config(base_dir / "configs" / "tptp.json")
    tptp_root = base_dir / tptp_config["paths"]["root"]

    # Determine problem set (use default from config if not specified)
    problem_set = args.problem_set
    if problem_set is None:
        problem_set = tptp_config.get("defaults", {}).get("problem_set")
        if problem_set is None:
            print("Error: No --problem-set specified and no default in tptp.json")
            sys.exit(1)

    # Get available provers
    available_provers = get_available_provers(base_dir)
    if not available_provers:
        print("Error: No provers available")
        sys.exit(1)

    # Filter provers if specified
    if args.prover:
        if args.prover not in available_provers:
            print(f"Error: Prover '{args.prover}' not available")
            print(f"Available: {', '.join(available_provers.keys())}")
            sys.exit(1)
        available_provers = {args.prover: available_provers[args.prover]}

    # Build list of (prover, preset_name, preset, binary) combinations
    runs = []
    for prover_name, prover_info in available_provers.items():
        presets = prover_info["config"].get("presets", {})

        if args.preset:
            # Filter to only presets that exist for this prover
            preset_names = [p for p in args.preset if p in presets]
            if not preset_names:
                continue  # Skip this prover if no matching presets
        else:
            preset_names = list(presets.keys())

        for preset_name in preset_names:
            preset = presets[preset_name]

            # Skip learned selectors if --base-only
            if args.base_only and is_learned_selector(preset):
                continue

            runs.append({
                "prover": prover_name,
                "preset_name": preset_name,
                "preset": preset,
                "binary": prover_info["binary"],
                "config": prover_info["config"],
            })

    if not runs:
        print("Error: No matching prover/preset combinations")
        sys.exit(1)

    # Get problems
    problems = get_problems(base_dir, tptp_config, problem_set)

    log_file_path = get_log_file(base_dir)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Double fork to fully daemonize (survives terminal close, SSH disconnect)
    # Use a pipe to communicate grandchild PID back to parent
    read_fd, write_fd = os.pipe()

    pid = os.fork()
    if pid > 0:
        # First parent: wait for intermediate child and read grandchild PID
        os.close(write_fd)
        os.waitpid(pid, 0)
        grandchild_pid = int(os.read(read_fd, 32).decode().strip())
        os.close(read_fd)

        # Job status is saved by grandchild - just wait briefly for it
        time.sleep(0.1)
        prover_names = sorted(set(r["prover"] for r in runs))
        print(f"Started job (PID: {grandchild_pid})")
        print(f"Provers: {', '.join(prover_names)}, Configs: {len(runs)}, Problems: {len(problems)}")
        print("Use --track to monitor, --status to check, --kill to stop")
        if args.track:
            time.sleep(0.5)
            print()
            track_job(base_dir)
        return

    # First child: become session leader and fork again
    os.close(read_fd)
    os.setsid()
    import signal
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    pid2 = os.fork()
    if pid2 > 0:
        # Intermediate child: send grandchild PID to parent and exit
        os.write(write_fd, f"{pid2}\n".encode())
        os.close(write_fd)
        os._exit(0)

    # Second child (grandchild): the actual daemon
    os.close(write_fd)

    # Close stdin and redirect stdout/stderr to log file early
    # so any errors during startup are captured
    sys.stdin.close()
    os.close(0)
    sys.stdout = open(log_file_path, "w")
    sys.stderr = sys.stdout

    # Log startup info
    print(f"Benchmark daemon started (PID: {os.getpid()})")
    print(f"Working directory: {base_dir}")
    print(f"Configs: {len(runs)}, Problems: {len(problems)}")
    sys.stdout.flush()

    # Clear any stale PID tracking and save job status
    clear_pids(base_dir)
    job_file_error = None
    try:
        save_job_status(base_dir, os.getpid(), sys.argv, len(runs))
        print(f"Job file saved: {get_job_file(base_dir)}")
    except Exception as e:
        job_file_error = str(e)
        print(f"WARNING: Failed to save job status: {e}")
        print("Use 'ps aux | grep proofatlas' to find this process")
    sys.stdout.flush()

    # Set up signal handlers to log unexpected termination
    def signal_handler(signum, frame):
        sig_names = {signal.SIGTERM: "SIGTERM", signal.SIGINT: "SIGINT",
                     signal.SIGQUIT: "SIGQUIT", signal.SIGABRT: "SIGABRT"}
        sig_name = sig_names.get(signum, f"signal {signum}")
        print(f"\nRECEIVED {sig_name} - exiting")
        sys.stdout.flush()
        clear_pids(base_dir)
        clear_job_status(base_dir)
        sys.stdout.close()
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGQUIT, signal_handler)

    # Extract problem names for filtering traces
    problem_names = {p.stem for p in problems}
    num_runs = len(runs)

    try:
        for run_idx, run in enumerate(runs, 1):
            prover = run["prover"]
            preset_name = run["preset_name"]
            preset = run["preset"]
            binary = run["binary"]
            prover_config = run["config"]

            config_label = f"{prover}/{preset_name}"
            trace_preset = args.trace_preset or preset.get("traces") or preset_name

            # Log config progress for --status parsing
            print(f"CONFIG:{config_label}:{run_idx}:{num_runs}")
            sys.stdout.flush()

            print(f"\n{'='*60}")
            print(f"Running: {config_label} ({run_idx}/{num_runs})")
            print(f"{'='*60}\n")

            weights_path = None
            current_preset = preset

            # Training only supported for proofatlas
            if prover == "proofatlas" and is_learned_selector(preset):
                existing_weights = find_weights(base_dir, preset_name)

                if existing_weights and not args.force_train:
                    print(f"Using existing weights: {existing_weights}")
                    weights_path = existing_weights
                else:
                    print(f"Training {preset_name}...")

                    # First collect traces with age_weight if none exist
                    traces_dir = base_dir / ".data" / "traces" / trace_preset
                    proofatlas_presets = prover_config.get("presets", {})
                    if not traces_dir.exists() or not list(traces_dir.glob("*.json")):
                        print("Collecting traces with age_weight...")
                        trace_source_preset = proofatlas_presets.get(trace_preset, preset)
                        run_evaluation(
                            base_dir, problems, tptp_root,
                            prover="proofatlas", preset=trace_source_preset,
                            log_file=sys.stdout,
                            preset_name=trace_preset, trace_preset=trace_preset,
                            rerun=True,  # Always run for trace collection
                        )

                    # Load traces and train (filtered by problem set)
                    data = load_traces(base_dir, trace_preset, problem_names)
                    if data["num_problems"] > 0:
                        weights_path = run_training(
                            base_dir, preset_name, preset, data, sys.stdout,
                            init_weights=existing_weights,
                        )
                    else:
                        print("No traces collected, using age_weight")
                        current_preset = proofatlas_presets.get(trace_preset, preset)
                        weights_path = None

            # Run evaluation
            run_evaluation(
                base_dir, problems, tptp_root,
                prover=prover, preset=current_preset,
                log_file=sys.stdout,
                preset_name=preset_name, weights_path=str(weights_path) if weights_path else None,
                binary=binary, trace_preset=trace_preset,
                rerun=args.rerun, n_jobs=args.n_jobs,
            )

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        clear_pids(base_dir)
        clear_job_status(base_dir)
        sys.stdout.close()
    os._exit(0)


if __name__ == "__main__":
    main()
