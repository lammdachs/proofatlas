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
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

from ..selectors import create_model

from .losses import compute_loss
from .datasets import ProofBatchDataset, scan_trace_files
from .logger import JSONLogger
from .export import export_model


# =============================================================================
# Forward Pass (shared between train and val)
# =============================================================================


def _edge_list_to(edge_list, device):
    """Move edge-list tuple/list or tensor to device."""
    if isinstance(edge_list, (tuple, list)):
        row, col, val, shape = edge_list
        return (row.to(device), col.to(device), val.to(device), shape)
    return edge_list.to(device)


def _build_sym_emb(model, batch, prefix, device):
    """Build symbol embeddings from pre-computed frozen embeddings + learned sentinels.

    If the batch contains pre-computed node_embeddings and node_sentinel_type,
    replaces sentinel positions with the model's learned sentinel embeddings.
    Otherwise falls back to node_names for on-the-fly computation.

    Returns (sym_emb, node_names) — one will be non-None, the other None.
    """
    # Only use pre-computed embeddings if the model has symbol_embedding
    if not hasattr(model, 'symbol_embedding') or model.symbol_embedding is None:
        return None, batch.get(f"{prefix}_node_names")

    emb_key = f"{prefix}_node_embeddings"
    st_key = f"{prefix}_node_sentinel_type"

    if emb_key in batch:
        sym_emb = batch[emb_key].clone().to(device)
        sentinel = batch[st_key].to(device)
        # Replace sentinel positions with learned embeddings
        for st in range(3):  # VAR=0, CLAUSE=1, LIT=2
            mask = sentinel == st
            if mask.any():
                sym_emb[mask] = model.symbol_embedding.sentinel_embeddings.weight[st]
        return sym_emb, None
    else:
        return None, batch.get(f"{prefix}_node_names")


def _forward_pass(model, batch, device, is_sentence_model, is_features_model=False):
    """Run forward pass through model, returning scores.

    Handles sentence, GNN, and features-only model types, encoding U and P
    sets separately. Supports pre-computed embeddings (from NPZ traces) as
    well as on-the-fly computation.

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
        u_raw = batch["u_embeddings"].to(device)
        u_emb = model.projection(u_raw)

        p_emb = None
        if "p_embeddings" in batch:
            p_raw = batch["p_embeddings"].to(device)
            p_emb = model.projection(p_raw)

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

        # Pre-computed node embeddings or fallback to node_names
        u_sym_emb, u_node_names = _build_sym_emb(model, batch, "u", device)
        u_emb = model.encode(u_x, u_adj, u_pool, u_cf, node_names=u_node_names, sym_emb=u_sym_emb)

        p_emb = None
        if "p_node_features" in batch:
            p_x = batch["p_node_features"].to(device)
            p_adj = _edge_list_to(batch["p_adj"], device)
            p_pool = _edge_list_to(batch["p_pool_matrix"], device)
            p_cf = batch.get("p_clause_features")
            if p_cf is not None:
                p_cf = p_cf.to(device)
            p_sym_emb, p_node_names = _build_sym_emb(model, batch, "p", device)
            p_emb = model.encode(p_x, p_adj, p_pool, p_cf, node_names=p_node_names, sym_emb=p_sym_emb)

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
    max_clauses: Optional[int] = None,
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
        max_clauses: Override max U clauses per batch (default: from config)
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
            max_clauses=max_clauses,
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
    max_clauses,
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
        output_type = "sentence"
    else:
        # Both "graph" and "features" use graph.npz traces
        output_type = "graph"

    model_name = get_model_name(preset)

    # Get per-problem trace files (flat: STEM.graph.npz or STEM.sentence.npz)
    trace_dir = Path(trace_dir)
    suffix = "sentence.npz" if embedding_type == "string" else "graph.npz"
    trace_files = sorted(trace_dir.glob(f"*.{suffix}"))
    if problem_names is not None:
        trace_files = [f for f in trace_files if f.stem.rsplit(".", 1)[0] in problem_names]

    if not trace_files:
        raise ValueError(f"No trace files found in {trace_dir} (looking for *.{suffix})")

    log_msg(f"Found {len(trace_files)} problem traces")

    # Problem-level split
    val_ratio = config.get("val_ratio", 0.0)
    random.seed(42)
    problem_list = sorted(f.stem.rsplit(".", 1)[0] for f in trace_files)
    random.shuffle(problem_list)
    n_val_problems = int(len(problem_list) * val_ratio)
    val_problem_set = set(problem_list[:n_val_problems])

    train_files = [f for f in trace_files if f.stem.rsplit(".", 1)[0] not in val_problem_set]
    val_files = [f for f in trace_files if f.stem.rsplit(".", 1)[0] in val_problem_set]
    log_msg(f"Training {model_name}: {len(train_files)} train, {len(val_files)} val problems")

    # Resolve batch size and accumulation: CLI overrides > config
    is_sentence_model_type = (embedding_type == "string")

    if max_clauses is None:
        default_max_clauses = 512 if is_sentence_model_type else 8192
        resolved_max_clauses = config.get("max_clauses_per_batch", default_max_clauses)
    else:
        resolved_max_clauses = max_clauses

    default_accumulate = 4 if is_sentence_model_type else 1
    accumulate_steps = accumulate_batches if accumulate_batches is not None else config.get("accumulate_steps", default_accumulate)

    log_msg(f"Batch: {resolved_max_clauses} max clauses, {accumulate_steps} accumulation steps, {cpu_workers} workers")

    # Create datasets using ProofBatchDataset
    log_msg(f"Loading training data ({output_type} format)...")
    train_ds = ProofBatchDataset(
        files=train_files,
        output_type=output_type,
        max_clauses=resolved_max_clauses,
        shuffle=True,
    )

    val_ds = None
    if val_files:
        log_msg(f"Loading validation data ({output_type} format)...")
        val_ds = ProofBatchDataset(
            files=val_files,
            output_type=output_type,
            max_clauses=resolved_max_clauses,
            shuffle=False,
        )

    log_msg(f"Train: {train_ds.num_problems} problems, Val: {val_ds.num_problems if val_ds else 0} problems")

    # DataLoader: batch_size=None because IterableDataset yields complete batches
    # collate_fn passes through the single batch dict from each yield
    # With DDP, each rank creates its own DataLoader, so divide workers accordingly
    loader_workers = max(1, cpu_workers // world_size) if world_size > 1 else cpu_workers
    train_loader = DataLoader(
        train_ds,
        batch_size=None,
        num_workers=loader_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = None
    if val_ds:
        val_loader = DataLoader(
            val_ds,
            batch_size=None,
            num_workers=loader_workers,
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
        node_info=emb_config.get("node_info", "features"),
        use_clause_features=emb_config.get("use_clause_features", True),
        sin_dim=emb_config.get("sin_dim", 8),
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
                "max_clauses": resolved_max_clauses,
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

    log_every_steps = config.get("log_every_steps", 100)

    for epoch in range(1, max_epochs + 1):
        log_msg(f"Epoch {epoch}/{max_epochs} starting...")
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

            # Intra-epoch progress
            if is_main and num_batches % log_every_steps == 0:
                avg_loss = train_loss / num_batches
                log_msg(f"  Epoch {epoch}/{max_epochs} step {num_batches} | loss={avg_loss:.4f}")

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


