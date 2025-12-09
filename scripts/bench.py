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


def get_job_file(base_dir: Path) -> Path:
    return base_dir / JOB_FILE


def get_log_file(base_dir: Path) -> Path:
    return base_dir / LOG_FILE


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
    job = get_job_status(base_dir)
    if not job:
        return False

    try:
        os.kill(job["pid"], signal.SIGTERM)
        clear_job_status(base_dir)
        return True
    except (OSError, ProcessLookupError):
        clear_job_status(base_dir)
        return False


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

            if "status" in filters and filters["status"]:
                if meta.get("status") not in filters["status"]:
                    continue
            if "format" in filters and filters["format"]:
                if meta.get("format") not in filters["format"]:
                    continue
            if "max_rating" in filters and filters["max_rating"] is not None:
                if meta.get("rating", 1.0) > filters["max_rating"]:
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
    """Check if selector requires trained weights (has a 'model' field)."""
    return "model" in selector_config


# Trace collection and training

def save_trace(base_dir: Path, preset: str, problem: str, trace_data: dict):
    """Save proof trace for training."""
    try:
        import torch
        traces_dir = base_dir / ".data" / "traces" / preset
        traces_dir.mkdir(parents=True, exist_ok=True)
        problem_name = Path(problem).stem
        torch.save(trace_data, traces_dir / f"{problem_name}.pt")
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
    import torch

    traces_dir = base_dir / ".data" / "traces" / preset
    if not traces_dir.exists():
        return {"problems": [], "num_problems": 0}

    problems = []
    for trace_file in sorted(traces_dir.glob("*.pt")):
        # Filter by problem set if specified
        if problem_names is not None and trace_file.stem not in problem_names:
            continue

        try:
            trace = torch.load(trace_file, weights_only=False)
        except Exception:
            continue

        if not trace.get("proof_found") or not trace.get("graphs"):
            continue

        problems.append({
            "name": trace_file.stem,
            "graphs": trace["graphs"],
            "labels": trace["labels"],
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
    with open(base_dir / "configs" / "models.json") as f:
        models_config = json.load(f)
    with open(base_dir / "configs" / "training.json") as f:
        training_config = json.load(f)

    # Get model and training config from preset
    model_name = preset.get("model")
    training_name = preset.get("training", "standard")

    # Get model architecture
    arch = models_config["architectures"].get(model_name)
    if not arch:
        raise ValueError(f"Unknown model: {model_name}")

    # Get training config
    training_defaults = training_config.get("defaults", {})
    training_overrides = training_config.get("configs", {}).get(training_name, {})

    # Merge: training defaults < training overrides < model architecture
    config = {**training_defaults, **training_overrides, **arch}
    config["input_dim"] = models_config.get("input_dim", 13)

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

def run_proofatlas(problem: Path, base_dir: Path, preset: dict, tptp_root: Path,
                   weights_path: str = None, collect_trace: bool = False,
                   trace_preset: str = None) -> BenchResult:
    """Run ProofAtlas on a problem."""
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

    max_clauses = preset.get("max_clauses", 10000)
    is_learned = "model" in preset
    age_weight_ratio = preset.get("age_weight_ratio", 0.167)
    selector = preset.get("model", "age_weight") if is_learned else "age_weight"

    # Remaining time after parsing
    elapsed_parsing = time.time() - start
    remaining_timeout = max(0.1, timeout - elapsed_parsing)

    try:
        proof_found = state.run_saturation(
            max_clauses,
            float(remaining_timeout),
            float(age_weight_ratio) if not is_learned else None,
            selector,
            weights_path,
        )
    except Exception:
        return BenchResult(problem=problem.name, status="error", time_s=time.time() - start)

    elapsed = time.time() - start

    if proof_found:
        status = "proof"
    elif elapsed >= timeout:
        status = "timeout"
    else:
        status = "saturated"

    # Collect trace for training
    if collect_trace and proof_found and trace_preset:
        try:
            from proofatlas.ml.graph_utils import to_torch_tensors
            examples = state.extract_training_examples()
            if examples:
                clause_ids = [e.clause_idx for e in examples]
                graphs = state.clauses_to_graphs(clause_ids)
                graph_tensors = [to_torch_tensors(g) for g in graphs]
                labels = [e.label for e in examples]
                trace_data = {
                    "proof_found": True,
                    "time": elapsed,
                    "graphs": graph_tensors,
                    "labels": labels,
                }
                save_trace(base_dir, trace_preset, problem.name, trace_data)
        except Exception:
            pass

    return BenchResult(problem=problem.name, status=status, time_s=elapsed)


def run_vampire(problem: Path, base_dir: Path, preset: dict, binary: Path, tptp_root: Path) -> BenchResult:
    """Run Vampire on a problem."""
    import subprocess

    timeout = preset.get("time_limit", 10)
    selection = preset.get("selection", 21)
    avatar = preset.get("avatar", "off")

    cmd = [
        str(binary),
        "--include", str(tptp_root),
        "--time_limit", str(timeout),
        "--selection", str(selection),
        "--avatar", avatar,
        str(problem),
    ]

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 5,  # grace period
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
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
    else:
        status = "error"

    return BenchResult(problem=problem.name, status=status, time_s=elapsed)


def run_spass(problem: Path, base_dir: Path, preset: dict, binary: Path, tptp_root: Path) -> BenchResult:
    """Run SPASS on a problem."""
    import subprocess

    timeout = preset.get("TimeLimit", 10)
    selection = preset.get("Select", 1)

    # SPASS requires TPTP format with -TPTP flag
    cmd = [
        str(binary),
        "-TPTP",
        f"-TimeLimit={timeout}",
        f"-Select={selection}",
        str(problem),
    ]

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 5,
            env={**os.environ, "TPTP": str(tptp_root)},
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
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


def run_evaluation(base_dir: Path, problems: list[Path], tptp_root: Path,
                   prover: str, preset: dict, log_file,
                   preset_name: str = None, weights_path: str = None,
                   binary: Path = None, trace_preset: str = None,
                   rerun: bool = False):
    """Run evaluation on problems with the specified prover."""
    stats = {"proof": 0, "saturated": 0, "timeout": 0, "error": 0, "skip": 0}

    if prover == "proofatlas":
        selector_type = "learned" if "model" in preset else "age_weight"
        print(f"\nEvaluating {len(problems)} problems with {selector_type}")
        if weights_path:
            print(f"Weights: {weights_path}")
    else:
        print(f"\nEvaluating {len(problems)} problems")

    # Always collect traces for proofatlas
    collect_trace = (prover == "proofatlas")

    for i, problem in enumerate(problems, 1):
        # Check if already evaluated (skip unless --rerun)
        existing = load_run_result(base_dir, prover, preset_name, problem)
        if existing and not rerun:
            stats[existing.status] = stats.get(existing.status, 0) + 1
            stats["skip"] += 1

            log_file.write(f"SKIP:{i}:{len(problems)}:{problem.name}\n")
            log_file.flush()

            symbol = {"proof": "+", "saturated": "~", "timeout": "T", "error": "!"}[existing.status]
            print(f"[{i}/{len(problems)}] S{symbol} {existing.problem} (cached)")
            continue

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

        stats[result.status] = stats.get(result.status, 0) + 1

        # Save individual result to .data/runs/
        save_run_result(base_dir, prover, preset_name, result)

        log_file.write(f"PROGRESS:{i}:{len(problems)}:{stats['proof']}:{stats['timeout']}\n")
        log_file.flush()

        symbol = {"proof": "+", "saturated": "~", "timeout": "T", "error": "!"}[result.status]
        print(f"[{i}/{len(problems)}] {symbol} {result.problem} ({result.time_s:.2f}s)")

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
    parser.add_argument("--preset",
                       help="Solver preset (default: all)")
    parser.add_argument("--problem-set", default="default",
                       help="Problem set from tptp.json")
    parser.add_argument("--force-train", action="store_true",
                       help="Force retrain even if weights exist")
    parser.add_argument("--base-only", action="store_true",
                       help="Only run base configs (skip learned selectors)")
    parser.add_argument("--rerun", action="store_true",
                       help="Re-evaluate problems even if cached results exist")
    parser.add_argument("--trace-preset",
                       help="Preset name for trace collection (default: solver preset)")

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
        if kill_job(base_dir):
            print("Job killed.")
        else:
            print("No job running.")
        return

    existing = get_job_status(base_dir)
    if existing:
        print(f"Error: Job already running (PID: {existing['pid']})")
        print("Use --status, --track, or --kill")
        sys.exit(1)

    # Load configs
    tptp_config = load_config(base_dir / "configs" / "tptp.json")
    tptp_root = base_dir / tptp_config["paths"]["root"]

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
            if args.preset not in presets:
                continue  # Skip this prover if preset not available
            preset_names = [args.preset]
        else:
            preset_names = list(presets.keys())

        for preset_name in preset_names:
            preset = presets[preset_name]

            # Skip learned selectors if --base-only
            if args.base_only and "model" in preset:
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
    problems = get_problems(base_dir, tptp_config, args.problem_set)

    log_file_path = get_log_file(base_dir)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Fork to background
    pid = os.fork()
    if pid > 0:
        save_job_status(base_dir, pid, sys.argv, len(runs))
        prover_names = sorted(set(r["prover"] for r in runs))
        print(f"Started job (PID: {pid})")
        print(f"Provers: {', '.join(prover_names)}, Configs: {len(runs)}, Problems: {len(problems)}")
        print("Use --track to monitor, --status to check, --kill to stop")
        if args.track:
            time.sleep(0.5)
            print()
            track_job(base_dir)
    else:
        os.setsid()
        sys.stdout = open(log_file_path, "w")
        sys.stderr = sys.stdout

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
                        if not traces_dir.exists() or not list(traces_dir.glob("*.pt")):
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
                    rerun=args.rerun,
                )

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            clear_job_status(base_dir)
            sys.stdout.close()
        os._exit(0)


if __name__ == "__main__":
    main()
