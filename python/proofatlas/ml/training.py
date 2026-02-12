"""
Training infrastructure for clause selection models.

Provides the training loop, model persistence, and trace management.
Loss functions, datasets, collate functions, and the web logger have been
extracted to dedicated modules (losses, datasets, logger, export).
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any

import torch
import torch.nn as nn

from ..selectors import create_model
from .config import TrainingConfig

# Re-export from new modules for backward compatibility
from .losses import (
    info_nce_loss,
    info_nce_loss_per_proof,
    margin_ranking_loss,
    compute_loss,
)
from .datasets import (
    ProofBatchDataset,
    collate_proof_batch,
    collate_tokenized_batch,
    scan_trace_files,
)
from .logger import JSONLogger
from .export import export_model


# =============================================================================
# Model Persistence
# =============================================================================


def save_model(model: nn.Module, path: Path, config: Optional[TrainingConfig] = None):
    """Save model checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.to_dict() if config else None,
    }, path)


def load_model(path: Path, device: Optional[torch.device] = None) -> nn.Module:
    """Load model from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device)
    config_dict = checkpoint.get("config", {})

    # Handle both old flat config and new nested config
    model_config = config_dict.get("model", config_dict)

    model = create_model(
        model_type=model_config.get("type", model_config.get("model_type", "gcn")),
        hidden_dim=model_config.get("hidden_dim", 64),
        num_layers=model_config.get("num_layers", 3),
        num_heads=model_config.get("num_heads", 4),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


# =============================================================================
# Trace Management
# =============================================================================


def save_tensor_trace(traces_dir, preset, problem, graph_dict, sentence_dict):
    """Save tensor trace data as per-state .npz files.

    Splits the full-problem trace (all clauses + all states) into per-state
    files, each containing only the U+P clauses for that state.

    Layout: traces_dir/preset/PROBLEM/idx.{graph,sentence}.npz

    Args:
        traces_dir: Base directory for traces (e.g., .data/traces/)
        preset: Preset name for trace subdirectory
        problem: Problem file name
        graph_dict: Dict of numpy arrays for graph trace
        sentence_dict: Dict of numpy arrays for sentence trace
    """
    import numpy as np

    try:
        stem = Path(problem).stem
        problem_dir = Path(traces_dir) / preset / stem
        problem_dir.mkdir(parents=True, exist_ok=True)

        # Extract state arrays from graph_dict
        state_u_offsets = graph_dict["state_u_offsets"]
        state_p_offsets = graph_dict["state_p_offsets"]
        state_u_indices = graph_dict["state_u_indices"]
        state_p_indices = graph_dict["state_p_indices"]
        num_states = len(graph_dict["state_selected"])

        node_features = graph_dict["node_features"]
        edge_src = graph_dict["edge_src"]
        edge_dst = graph_dict["edge_dst"]
        node_offsets = graph_dict["node_offsets"]
        edge_offsets = graph_dict["edge_offsets"]
        clause_features = graph_dict["clause_features"]
        labels = graph_dict["labels"]

        # Sentence data
        clause_strings = sentence_dict.get("clause_strings")
        sent_labels = sentence_dict["labels"]

        for si in range(num_states):
            u_start, u_end = int(state_u_offsets[si]), int(state_u_offsets[si + 1])
            p_start, p_end = int(state_p_offsets[si]), int(state_p_offsets[si + 1])
            u_idx = state_u_indices[u_start:u_end].astype(np.int64)
            p_idx = state_p_indices[p_start:p_end].astype(np.int64)

            if len(u_idx) == 0:
                continue

            all_idx = np.concatenate([u_idx, p_idx])
            num_u = len(u_idx)

            # --- Graph per-state file ---
            all_nf = []
            all_es = []
            all_ed = []
            s_node_off = [0]
            s_edge_off = [0]
            node_cursor = 0

            for idx in all_idx:
                i = int(idx)
                ns, ne = int(node_offsets[i]), int(node_offsets[i + 1])
                es, ee = int(edge_offsets[i]), int(edge_offsets[i + 1])
                nn = ne - ns
                nedges = ee - es

                if nn > 0:
                    all_nf.append(node_features[ns:ne])
                if nedges > 0:
                    all_es.append(edge_src[es:ee].astype(np.int64) - ns + node_cursor)
                    all_ed.append(edge_dst[es:ee].astype(np.int64) - ns + node_cursor)

                node_cursor += nn
                s_node_off.append(node_cursor)
                s_edge_off.append(s_edge_off[-1] + nedges)

            cnf = np.concatenate(all_nf) if all_nf else np.zeros((0, node_features.shape[1]), dtype=node_features.dtype)
            ces = np.concatenate(all_es) if all_es else np.zeros(0, dtype=np.int64)
            ced = np.concatenate(all_ed) if all_ed else np.zeros(0, dtype=np.int64)

            np.savez_compressed(
                problem_dir / f"{si}.graph.npz",
                node_features=cnf,
                edge_src=ces,
                edge_dst=ced,
                node_offsets=np.array(s_node_off, dtype=np.int64),
                edge_offsets=np.array(s_edge_off, dtype=np.int64),
                clause_features=clause_features[all_idx],
                labels=labels[all_idx],
                num_u=np.array(num_u, dtype=np.int64),
            )

            # --- Sentence per-state file ---
            if clause_strings is not None:
                s_strings = np.array([clause_strings[int(i)] for i in all_idx], dtype=object)
                np.savez_compressed(
                    problem_dir / f"{si}.sentence.npz",
                    clause_strings=s_strings,
                    labels=sent_labels[all_idx],
                    num_u=np.array(num_u, dtype=np.int64),
                )
    except Exception:
        pass


def load_trace_files(
    traces_dir: Path,
    preset: str,
    encoder_type: str = "graph",
    problem_names: Optional[set] = None,
) -> List[Path]:
    """Get list of per-state trace files for a preset.

    Globs PROBLEM/idx.{graph,sentence}.npz in subdirectories.

    Args:
        traces_dir: Base directory for traces (e.g., .data/traces/)
        preset: Trace preset name (subdirectory in traces_dir)
        encoder_type: "graph" or "string" — determines which .npz suffix to glob
        problem_names: Optional set of problem names to include. If None, loads all.

    Returns:
        List of per-state trace file paths.
    """
    preset_dir = Path(traces_dir) / preset
    if not preset_dir.exists():
        return []

    suffix = "sentence.npz" if encoder_type == "string" else "graph.npz"
    trace_files = sorted(preset_dir.glob(f"**/*.{suffix}"))
    if problem_names is not None:
        # Parent directory name is the problem stem
        trace_files = [f for f in trace_files if f.parent.name in problem_names]

    return trace_files


# =============================================================================
# Forward Pass (shared between train and val)
# =============================================================================


def _edge_list_to(edge_list, device):
    """Move edge-list tuple/list or tensor to device."""
    if isinstance(edge_list, (tuple, list)):
        row, col, val, shape = edge_list
        return (row.to(device), col.to(device), val.to(device), shape)
    return edge_list.to(device)


def _forward_pass(model, batch, device, is_sentence_model, is_features_model=False):
    """Run forward pass through model, returning scores.

    Handles sentence, GNN, and features-only model types, encoding U and P
    sets separately.

    Args:
        model: The clause selection model
        batch: Collated batch dict from dataloader
        device: Torch device for computation
        is_sentence_model: Whether this is a sentence transformer model
        is_features_model: Whether this is a features-only model

    Returns:
        scores tensor on device
    """
    if is_sentence_model:
        u_ids = batch["u_input_ids"].to(device)
        u_mask = batch["u_attention_mask"].to(device)
        u_emb = model.encode_tokens(u_ids, u_mask)

        p_emb = None
        if "p_input_ids" in batch:
            p_ids = batch["p_input_ids"].to(device)
            p_mask = batch["p_attention_mask"].to(device)
            p_emb = model.encode_tokens(p_ids, p_mask)

        return model.scorer(u_emb, p_emb)
    elif is_features_model:
        u_cf = batch["u_clause_features"].to(device)
        u_emb = model.encode(u_cf)

        p_emb = None
        if batch.get("p_clause_features") is not None:
            p_cf = batch["p_clause_features"].to(device)
            p_emb = model.encode(p_cf)

        return model.scorer(u_emb, p_emb)
    else:
        u_x = batch["u_node_features"].to(device)
        u_adj = _edge_list_to(batch["u_adj"], device)
        u_pool = _edge_list_to(batch["u_pool_matrix"], device)
        u_cf = batch.get("u_clause_features")
        if u_cf is not None:
            u_cf = u_cf.to(device)

        u_emb = model.encode(u_x, u_adj, u_pool, u_cf)

        p_emb = None
        if "p_node_features" in batch:
            p_x = batch["p_node_features"].to(device)
            p_adj = _edge_list_to(batch["p_adj"], device)
            p_pool = _edge_list_to(batch["p_pool_matrix"], device)
            p_cf = batch.get("p_clause_features")
            if p_cf is not None:
                p_cf = p_cf.to(device)
            p_emb = model.encode(p_x, p_adj, p_pool, p_cf)

        return model.scorer(u_emb, p_emb)


# =============================================================================
# Training Loop
# =============================================================================


def run_training(
    preset: dict,
    trace_dir: Path,
    weights_dir: Path,
    configs_dir: Path,
    problem_names: Optional[set] = None,
    log_callback: Optional[callable] = None,
    web_data_dir: Optional[Path] = None,
    log_file: Optional[Any] = None,
    cpu_workers: int = 0,
    rank: int = 0,
    world_size: int = 1,
    max_batch_bytes: Optional[int] = None,
    accumulate_batches: Optional[int] = None,
    force_cpu: bool = False,
    max_epochs: Optional[int] = None,
) -> Path:
    """Train a model and return the weights path.

    Args:
        preset: Preset config dict with embedding/scorer fields
        trace_dir: Directory containing trace JSON files
        weights_dir: Directory to save weights (.weights/)
        configs_dir: Directory containing config files (embeddings.json, etc.)
        problem_names: Optional set of problem names to filter traces
        log_callback: Optional callback(epoch, max_epochs, train_loss) for logging
        web_data_dir: Directory for web data (web/data/) - enables live web updates
        log_file: File object for logging (e.g., bench.log)
        cpu_workers: Number of DataLoader workers (default: 0)
        rank: GPU rank for DDP (default: 0)
        world_size: Total GPU count for DDP (default: 1)
        max_batch_bytes: Override max batch size in bytes (default: from config)
        accumulate_batches: Override gradient accumulation steps (default: from config)

    Returns:
        Path to weights directory (not the file) to match find_weights().
    """
    import random
    import time

    import torch.optim as optim
    from torch.utils.data import DataLoader

    from .weights import get_model_name, get_encoder_type

    start_time = time.time()
    is_main = rank == 0
    use_ddp = world_size > 1

    def log_msg(msg: str):
        """Log message with timestamp (only on rank 0)."""
        if not is_main:
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}"
        if log_file and log_file is not sys.stdout:
            print(line, flush=True)
            log_file.write(line + "\n")
            log_file.flush()
        else:
            print(line, flush=True)

    # DDP setup
    if use_ddp:
        import torch.distributed as dist
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    elif force_cpu:
        device = torch.device("cpu")
    else:
        try:
            use_cuda = torch.cuda.is_available()
        except Exception:
            use_cuda = False
        device = torch.device("cuda" if use_cuda else "cpu")

    try:
        return _run_training_inner(
            preset=preset,
            trace_dir=trace_dir,
            weights_dir=weights_dir,
            configs_dir=configs_dir,
            problem_names=problem_names,
            log_callback=log_callback,
            web_data_dir=web_data_dir,
            log_msg=log_msg,
            cpu_workers=cpu_workers,
            rank=rank,
            world_size=world_size,
            device=device,
            is_main=is_main,
            use_ddp=use_ddp,
            max_batch_bytes=max_batch_bytes,
            accumulate_batches=accumulate_batches,
            start_time=start_time,
            max_epochs_override=max_epochs,
        )
    finally:
        if use_ddp:
            import torch.distributed as dist
            dist.destroy_process_group()


def _run_training_inner(
    *,
    preset,
    trace_dir,
    weights_dir,
    configs_dir,
    problem_names,
    log_callback,
    web_data_dir,
    log_msg,
    cpu_workers,
    rank,
    world_size,
    device,
    is_main,
    use_ddp,
    max_batch_bytes,
    accumulate_batches,
    start_time,
    max_epochs_override=None,
):
    """Inner training loop, separated for clean DDP cleanup."""
    import random
    import time

    import torch.optim as optim
    from torch.utils.data import DataLoader

    from .weights import get_model_name, get_encoder_type

    # Load configs
    configs_dir = Path(configs_dir)
    with open(configs_dir / "embeddings.json") as f:
        embeddings_config = json.load(f)
    with open(configs_dir / "scorers.json") as f:
        scorers_config = json.load(f)
    with open(configs_dir / "training.json") as f:
        training_config = json.load(f)

    # Get model and training config from preset
    embedding_name = preset.get("encoder") or preset.get("embedding") or preset.get("model")
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

    # Determine output type based on embedding
    embedding_type = get_encoder_type(preset)
    if embedding_type == "string":
        output_type = "tokenized"
    else:
        # Both "graph" and "features" use graph.npz traces
        output_type = "graph"

    model_name = get_model_name(preset)

    # Get per-state trace files (.npz in PROBLEM/ subdirectories)
    trace_dir = Path(trace_dir)
    suffix = "sentence.npz" if embedding_type == "string" else "graph.npz"
    trace_files = sorted(trace_dir.glob(f"**/*.{suffix}"))
    if problem_names is not None:
        trace_files = [f for f in trace_files if f.parent.name in problem_names]

    if not trace_files:
        raise ValueError(f"No trace files found in {trace_dir} (looking for **/*.{suffix})")

    # Count distinct problems (parent directories)
    problems = {f.parent.name for f in trace_files}
    log_msg(f"Found {len(trace_files)} state files from {len(problems)} problems")

    # Validate trace files
    valid_files = scan_trace_files(trace_files)
    if not valid_files:
        raise ValueError("No valid traces found (all filtered or empty)")
    valid_problems = {f.parent.name for f in valid_files}
    log_msg(f"Validated {len(valid_files)} states from {len(valid_problems)} problems")

    # Problem-level split: group by problem, split problems, then flatten
    val_ratio = config.get("val_ratio", 0.0)
    random.seed(42)
    problem_list = sorted(valid_problems)
    random.shuffle(problem_list)
    n_val_problems = int(len(problem_list) * val_ratio)
    val_problem_set = set(problem_list[:n_val_problems])

    train_files = [f for f in valid_files if f.parent.name not in val_problem_set]
    val_files = [f for f in valid_files if f.parent.name in val_problem_set]
    log_msg(f"Training {model_name}: {len(train_files)} train states, {len(val_files)} val states")

    # Resolve batch size and accumulation: CLI overrides > config
    is_sentence_model_type = (embedding_type == "string")

    if max_batch_bytes is None:
        # Fall back to config-based defaults
        default_max_clauses = 512 if is_sentence_model_type else 8192
        max_clauses = config.get("max_clauses_per_batch", default_max_clauses)
        # Estimate: ~1KB per clause for graphs, ~0.5KB for tokenized
        bytes_per_clause = 512 if is_sentence_model_type else 1024
        resolved_max_batch_bytes = max_clauses * bytes_per_clause
    else:
        resolved_max_batch_bytes = max_batch_bytes

    default_accumulate = 4 if is_sentence_model_type else 1
    accumulate_steps = accumulate_batches if accumulate_batches is not None else config.get("accumulate_steps", default_accumulate)

    log_msg(f"Batch: {_fmt_bytes(resolved_max_batch_bytes)} max, {accumulate_steps} accumulation steps, {cpu_workers} workers")

    # Create datasets using ProofBatchDataset
    log_msg(f"Loading training data ({output_type} format)...")
    train_ds = ProofBatchDataset(
        files=train_files,
        output_type=output_type,
        max_batch_bytes=resolved_max_batch_bytes,
        shuffle=True,
    )

    val_ds = None
    if val_files:
        log_msg(f"Loading validation data ({output_type} format)...")
        val_ds = ProofBatchDataset(
            files=val_files,
            output_type=output_type,
            max_batch_bytes=resolved_max_batch_bytes,
            shuffle=False,
        )

    log_msg(f"Train: {train_ds.num_problems} problems ({train_ds.num_files} states), Val: {val_ds.num_problems if val_ds else 0} problems")

    # DataLoader: batch_size=None because IterableDataset yields complete batches
    # collate_fn passes through the single batch dict from each yield
    train_loader = DataLoader(
        train_ds,
        batch_size=None,
        num_workers=cpu_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = None
    if val_ds:
        val_loader = DataLoader(
            val_ds,
            batch_size=None,
            num_workers=cpu_workers,
            pin_memory=(device.type == "cuda"),
        )

    # Create model
    emb_config = config.get("embedding", {})
    scorer_config = config.get("scorer", {})
    model_type = emb_config.get("type", config.get("type", "gcn"))
    hidden_dim = emb_config.get("hidden_dim", config.get("hidden_dim", 64))
    model = create_model(
        model_type=model_type,
        node_feature_dim=config.get("input_dim", 13),
        hidden_dim=hidden_dim,
        num_layers=emb_config.get("num_layers", config.get("num_layers", 3)),
        num_heads=emb_config.get("num_heads", config.get("num_heads", 4)),
        scorer_type=scorer_config.get("type", "mlp"),
        scorer_num_heads=scorer_config.get("num_heads", 4),
        scorer_num_layers=scorer_config.get("num_layers", 2),
        freeze_encoder=emb_config.get("freeze_encoder", False),
    )

    needs_adj = model_type in ["gcn", "gat", "graphsage", "gnn_transformer"]
    is_sentence_model = model_type == "sentence"
    is_features_model = model_type == "features"
    model = model.to(device)

    # DDP wrapping
    if use_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[rank])

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

    max_epochs = max_epochs_override if max_epochs_override is not None else config.get("max_epochs", 100)

    # Initialize web logger for live updates (rank 0 only)
    web_logger = None
    if web_data_dir and is_main:
        web_logger = JSONLogger(web_data_dir, model_name)
        web_logger.log_config(
            model_config={
                "type": model_type,
                "hidden_dim": config.get("hidden_dim", 64),
                "num_layers": config.get("num_layers", 3),
                "input_dim": config.get("input_dim", 13),
                "scorer_type": scorer_name,
            },
            training_config={
                "batch_size": _fmt_bytes(resolved_max_batch_bytes),
                "accumulate_steps": accumulate_steps,
                "learning_rate": lr,
                "max_epochs": max_epochs,
                "optimizer": optimizer_type,
                "weight_decay": weight_decay,
                "margin": config.get("margin", 0.1),
                "cpu_workers": cpu_workers,
                "gpu_workers": world_size,
            },
        )

    # Track best model and early stopping
    best_val_loss = float("inf")
    best_epoch = None
    best_state_dict = None
    patience = config.get("patience", 10)
    patience_counter = 0

    # Loss configuration
    loss_type = config.get("loss_type", "info_nce")
    temperature = config.get("temperature", 1.0)
    margin = config.get("margin", 0.1)
    gradient_clip = config.get("gradient_clip", 1.0)

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0
        num_batches = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            if batch is None:
                continue

            labels = batch["labels"].to(device)
            proof_ids = batch.get("proof_ids")
            if proof_ids is not None:
                proof_ids = proof_ids.to(device)

            # Use no_sync for intermediate accumulation steps with DDP
            is_accumulation_step = accumulate_steps > 1 and (step + 1) % accumulate_steps != 0
            ctx = model.no_sync() if (use_ddp and is_accumulation_step) else _nullcontext()

            with ctx:
                raw_model = model.module if use_ddp else model
                scores = _forward_pass(raw_model, batch, device, is_sentence_model, is_features_model)

                loss = compute_loss(scores, labels, proof_ids, loss_type, temperature, margin)
                if accumulate_steps > 1:
                    loss = loss / accumulate_steps
                loss.backward()

            train_loss += loss.item() * (accumulate_steps if accumulate_steps > 1 else 1)
            num_batches += 1

            # Step optimizer after accumulating gradients
            if accumulate_steps <= 1 or (step + 1) % accumulate_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                optimizer.zero_grad()

        # Flush remaining accumulated gradients at end of epoch
        if accumulate_steps > 1 and num_batches % accumulate_steps != 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            optimizer.zero_grad()

        train_loss /= num_batches if num_batches > 0 else 1

        val_loss = 0
        if val_loader is not None:
            model.eval()
            num_val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    if batch is None:
                        continue

                    labels = batch["labels"].to(device)
                    val_proof_ids = batch.get("proof_ids")
                    if val_proof_ids is not None:
                        val_proof_ids = val_proof_ids.to(device)

                    raw_model = model.module if use_ddp else model
                    scores = _forward_pass(raw_model, batch, device, is_sentence_model, is_features_model)

                    val_loss += compute_loss(scores, labels, val_proof_ids, loss_type, temperature, margin).item()
                    num_val_batches += 1
            val_loss /= num_val_batches if num_val_batches > 0 else 1

        # Track best model and early stopping (rank 0 only tracks, but all ranks continue)
        current_val = val_loss if val_loader is not None else train_loss
        if current_val < best_val_loss:
            best_val_loss = current_val
            best_epoch = epoch
            if is_main:
                raw_model = model.module if use_ddp else model
                best_state_dict = {k: v.cpu().clone() for k, v in raw_model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # Log to console and file
        if val_loader is not None:
            log_msg(f"Epoch {epoch}/{max_epochs} | train={train_loss:.4f} | val={val_loss:.4f}")
        else:
            log_msg(f"Epoch {epoch}/{max_epochs} | train={train_loss:.4f}")

        # Log to web for live updates
        if web_logger:
            web_logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss if val_loader is not None else None,
                val_acc=None,
                lr=lr,
            )

        if log_callback and is_main:
            log_callback(epoch, max_epochs, train_loss)

        # Early stopping check
        if patience_counter >= patience:
            log_msg(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    # Restore best model and export (rank 0 only)
    if is_main:
        raw_model = model.module if use_ddp else model
        if best_state_dict is not None:
            raw_model.load_state_dict(best_state_dict)
            log_msg(f"Restored best model from epoch {best_epoch}")

        # Log final results
        total_time = time.time() - start_time
        early_stopped = patience_counter >= patience
        if web_logger:
            web_logger.log_final(
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                total_time=total_time,
                termination_reason="early_stopped" if early_stopped else "completed",
            )

        # Export to TorchScript for Rust inference
        weights_path = export_model(
            model=raw_model,
            weights_dir=weights_dir,
            model_name=model_name,
            config=config,
            is_sentence_model=is_sentence_model,
            needs_adj=needs_adj,
        )

        log_msg(f"TorchScript model saved: {weights_path}")
        log_msg(f"Training completed in {total_time:.1f}s (best epoch: {best_epoch})")

    # Return the weights directory (not the file path) to match find_weights()
    return Path(weights_dir)


class _nullcontext:
    """Minimal no-op context manager (avoids importing contextlib)."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def _fmt_bytes(n: int) -> str:
    """Format byte count as human-readable string."""
    if n >= 1024 * 1024 * 1024:
        return f"{n / (1024**3):.1f}G"
    elif n >= 1024 * 1024:
        return f"{n / (1024**2):.1f}M"
    elif n >= 1024:
        return f"{n / 1024:.1f}K"
    else:
        return f"{n}B"
