"""
Training infrastructure for clause selection models.

Provides the training loop, model persistence, and trace management.
Loss functions, datasets, collate functions, and the web logger have been
extracted to dedicated modules (losses, datasets, logger, export).
"""

import json
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
    ProofDataset,
    DynamicBatchSampler,
    collate_proof_batch,
    collate_tokenized_batch,
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


def save_trace(traces_dir: Path, preset: str, problem: str, trace_json: str):
    """Save proof trace for training in structured JSON format.

    Args:
        traces_dir: Base directory for traces (e.g., .data/traces/)
        preset: Preset name for trace subdirectory
        problem: Problem file name
        trace_json: Structured JSON string from extract_structured_trace()
    """
    try:
        preset_dir = Path(traces_dir) / preset
        preset_dir.mkdir(parents=True, exist_ok=True)
        problem_name = Path(problem).stem
        json_path = preset_dir / f"{problem_name}.json"
        with open(json_path, "w") as f:
            f.write(trace_json)
    except Exception:
        pass


def load_trace_files(
    traces_dir: Path,
    preset: str,
    problem_names: Optional[set] = None,
) -> List[Path]:
    """Get list of trace files for a preset.

    Args:
        traces_dir: Base directory for traces (e.g., .data/traces/)
        preset: Trace preset name (subdirectory in traces_dir)
        problem_names: Optional set of problem names to include. If None, loads all.

    Returns:
        List of trace file paths.
    """
    preset_dir = Path(traces_dir) / preset
    if not preset_dir.exists():
        return []

    trace_files = sorted(preset_dir.glob("*.json"))
    if problem_names is not None:
        trace_files = [f for f in trace_files if f.stem in problem_names]

    return trace_files


# =============================================================================
# Forward Pass (shared between train and val)
# =============================================================================


def _forward_pass(model, batch, device, is_sentence_model):
    """Run forward pass through model, returning scores.

    Handles both sentence and GNN model types, encoding U and P sets separately.

    Args:
        model: The clause selection model
        batch: Collated batch dict from dataloader
        device: Torch device for computation
        is_sentence_model: Whether this is a sentence transformer model

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
    else:
        u_x = batch["u_node_features"].to(device)
        u_adj = batch["u_adj"].to(device)
        u_pool = batch["u_pool_matrix"].to(device)
        u_cf = batch.get("u_clause_features")
        if u_cf is not None:
            u_cf = u_cf.to(device)

        u_emb = model.encode(u_x, u_adj, u_pool, u_cf)

        p_emb = None
        if "p_node_features" in batch:
            p_x = batch["p_node_features"].to(device)
            p_adj = batch["p_adj"].to(device)
            p_pool = batch["p_pool_matrix"].to(device)
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

    Returns:
        Path to weights directory (not the file) to match find_weights().
    """
    import random
    import time

    import torch.optim as optim
    from torch.utils.data import DataLoader

    from .weights import get_model_name, get_encoder_type

    start_time = time.time()

    def log_msg(msg: str):
        """Log message with timestamp to stdout and optionally to log_file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}"
        if log_file and log_file is not sys.stdout:
            print(line, flush=True)
            log_file.write(line + "\n")
            log_file.flush()
        else:
            print(line, flush=True)

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

    # Get trace files
    trace_dir = Path(trace_dir)
    trace_files = sorted(trace_dir.glob("*.json"))
    if problem_names is not None:
        trace_files = [f for f in trace_files if f.stem in problem_names]

    if not trace_files:
        raise ValueError(f"No trace files found in {trace_dir}")

    log_msg(f"Found {len(trace_files)} trace files")

    # Problem-level split
    val_ratio = config.get("val_ratio", 0.0)
    random.seed(42)
    random.shuffle(trace_files)

    if val_ratio > 0:
        val_count = max(1, int(len(trace_files) * val_ratio))
        train_files = trace_files[val_count:]
        val_files = trace_files[:val_count]
    else:
        train_files = trace_files
        val_files = []

    # Determine output type based on embedding
    embedding_type = get_encoder_type(preset)
    if embedding_type == "string":
        output_type = "tokenized"
    else:
        output_type = "graph"

    # Create datasets (loads all data in parallel)
    log_msg(f"Loading training data ({output_type} format)...")
    train_ds = ProofDataset(
        trace_dir=None,
        trace_files=train_files,
        output_type=output_type,
    )

    val_ds = None
    if val_files:
        log_msg(f"Loading validation data ({output_type} format)...")
        val_ds = ProofDataset(
            trace_dir=None,
            trace_files=val_files,
            output_type=output_type,
        )

    log_msg(f"Train: {len(train_ds)} traces, Val: {len(val_ds) if val_ds else 0} traces")

    model_name = get_model_name(preset)
    log_msg(f"Training {model_name}: {len(train_ds)} traces")

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Dynamic batching
    default_max_clauses = 512 if is_sentence_model else 8192
    max_clauses = config.get("max_clauses_per_batch", default_max_clauses)
    default_accumulate = 4 if is_sentence_model else 1
    accumulate_steps = config.get("accumulate_steps", default_accumulate)

    collate_fn = collate_tokenized_batch if output_type == "tokenized" else collate_proof_batch

    train_sampler = DynamicBatchSampler(train_ds, max_clauses=max_clauses, shuffle=True)
    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=0
    )
    val_loader = None
    if val_ds:
        val_sampler = DynamicBatchSampler(val_ds, max_clauses=max_clauses, shuffle=False)
        val_loader = DataLoader(
            val_ds, batch_sampler=val_sampler, collate_fn=collate_fn, num_workers=0
        )

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

    # Initialize web logger for live updates
    web_logger = None
    if web_data_dir:
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
                "batch_size": accumulate_steps,
                "learning_rate": lr,
                "max_epochs": max_epochs,
                "optimizer": optimizer_type,
                "weight_decay": weight_decay,
                "margin": config.get("margin", 0.1),
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

            scores = _forward_pass(model, batch, device, is_sentence_model)

            loss = compute_loss(scores, labels, proof_ids, loss_type, temperature, margin)
            if accumulate_steps > 1:
                loss = loss / accumulate_steps
            loss.backward()
            train_loss += loss.item() * (accumulate_steps if accumulate_steps > 1 else 1)
            num_batches += 1

            # Step optimizer after accumulating gradients
            if accumulate_steps <= 1 or (step + 1) % accumulate_steps == 0 or (step + 1) == len(train_loader):
                gradient_clip = config.get("gradient_clip", 1.0)
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                optimizer.zero_grad()

        train_loss /= num_batches if num_batches > 0 else 1

        val_loss = 0
        if val_loader:
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

                    scores = _forward_pass(model, batch, device, is_sentence_model)

                    val_loss += compute_loss(scores, labels, val_proof_ids, loss_type, temperature, margin).item()
                    num_val_batches += 1
            val_loss /= num_val_batches if num_val_batches > 0 else 1

        # Track best model and early stopping
        current_val = val_loss if val_loader else train_loss
        if current_val < best_val_loss:
            best_val_loss = current_val
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # Log to console and file
        if val_loader:
            log_msg(f"Epoch {epoch}/{max_epochs} | train={train_loss:.4f} | val={val_loss:.4f}")
        else:
            log_msg(f"Epoch {epoch}/{max_epochs} | train={train_loss:.4f}")

        # Log to web for live updates
        if web_logger:
            web_logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss if val_loader else None,
                val_acc=None,
                lr=lr,
            )

        if log_callback:
            log_callback(epoch, max_epochs, train_loss)

        # Early stopping check
        if patience_counter >= patience:
            log_msg(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    # Restore best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
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
        model=model,
        weights_dir=weights_dir,
        model_name=model_name,
        config=config,
        is_sentence_model=is_sentence_model,
        needs_adj=needs_adj,
    )

    log_msg(f"TorchScript model saved: {weights_path}")
    log_msg(f"Training completed in {total_time:.1f}s (best epoch: {best_epoch})")

    # Return the weights directory (not the file path) to match find_weights()
    # Rust expects a directory and constructs the full path itself
    return Path(weights_dir)
