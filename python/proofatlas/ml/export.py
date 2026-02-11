"""TorchScript model export for Rust inference.

Exports trained PyTorch models to TorchScript format (.pt files) that
can be loaded by the Rust scoring server via tch-rs.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def export_model(
    model: nn.Module,
    weights_dir: Path,
    model_name: str,
    config: dict,
    is_sentence_model: bool,
    needs_adj: bool,
) -> Path:
    """Export a trained model to TorchScript for Rust inference.

    Args:
        model: Trained PyTorch model (already has best weights loaded)
        weights_dir: Directory to save the .pt file
        model_name: Name for the weights file (without extension)
        config: Training config dict (needs "input_dim" key)
        is_sentence_model: Whether this is a sentence transformer model
        needs_adj: Whether the model needs adjacency matrix input

    Returns:
        Path to the saved weights file
    """
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / f"{model_name}.pt"

    model.eval()

    # Trace on CPU — torch.jit.trace bakes in device placement for tensor
    # creation ops. CPU is the default eval backend (bench --gpu-workers 0).
    model.cpu()
    trace_device = torch.device("cpu")

    if is_sentence_model:
        model.export_torchscript(str(weights_path), save_tokenizer=True)
    else:
        # GNN models: trace with example inputs (must match Rust call signature)
        # Script GraphNorm modules before tracing — their forward() uses
        # data-dependent shapes (batch.max().item()) that trace bakes as constants
        if hasattr(model, 'norms'):
            for i in range(len(model.norms)):
                model.norms[i] = torch.jit.script(model.norms[i])

        # Rust sends sparse COO tensors for adj and pool_matrix
        num_nodes, num_clauses = 10, 3
        example_x = torch.randn(num_nodes, config.get("input_dim", 13), device=trace_device)
        example_adj = torch.eye(num_nodes, device=trace_device).to_sparse()
        example_pool = (torch.ones(num_clauses, num_nodes, device=trace_device) / num_nodes).to_sparse()
        example_clause_features = torch.randn(num_clauses, 3, device=trace_device)

        if needs_adj:
            traced = torch.jit.trace(model, (example_x, example_adj, example_pool, example_clause_features), check_trace=False)
        else:
            traced = torch.jit.trace(model, (example_x, example_pool), check_trace=False)

        traced.save(str(weights_path))

    return weights_path
