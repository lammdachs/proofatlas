"""Model weight management utilities.

Functions for finding, loading, and managing trained model weights.
"""

from pathlib import Path
from typing import Optional


# Encoders that use string input (clause text)
STRING_ENCODERS = {"sentence"}

# Encoders that use graph input (node features, adjacency, pooling)
GRAPH_ENCODERS = {"gcn", "gat", "graphsage"}

# Keep old names as aliases for backwards compatibility in training code
STRING_EMBEDDINGS = STRING_ENCODERS
GRAPH_EMBEDDINGS = GRAPH_ENCODERS


def get_model_name(preset: dict) -> str:
    """Get model file name from preset config.

    Uses modular naming: {encoder}_{scorer}
    """
    encoder = preset.get("encoder")
    scorer = preset.get("scorer")
    if encoder and scorer:
        return f"{encoder}_{scorer}"
    raise ValueError("Preset must have 'encoder' and 'scorer' fields")


def get_encoder_type(preset: dict) -> Optional[str]:
    """Get encoder type category from preset config.

    Returns "graph", "string", or None for non-ML presets.
    """
    encoder = preset.get("encoder")
    if not encoder:
        return None

    if encoder in STRING_ENCODERS:
        return "string"
    elif encoder in GRAPH_ENCODERS:
        return "graph"
    else:
        raise ValueError(f"Unknown encoder: {encoder}")


# Keep old name as alias
get_embedding_type = get_encoder_type


def is_learned_selector(preset: dict) -> bool:
    """Check if preset requires trained weights."""
    return "encoder" in preset and "scorer" in preset


def find_weights(weights_dir: Path, preset: dict) -> Optional[Path]:
    """Find TorchScript model for a learned selector.

    Args:
        weights_dir: Directory containing model weights (.weights/)
        preset: Preset config dict with encoder/scorer fields

    Returns:
        Path to weights directory if model exists, or None if not found.
    """
    if not weights_dir.exists():
        return None

    model_name = get_model_name(preset)
    encoder_type = get_encoder_type(preset)

    if encoder_type == "graph":
        model_path = weights_dir / f"{model_name}.pt"
        if model_path.exists():
            return weights_dir
    elif encoder_type == "string":
        model_path = weights_dir / f"{model_name}.pt"
        tokenizer_path = weights_dir / f"{model_name}_tokenizer" / "tokenizer.json"
        if model_path.exists() and tokenizer_path.exists():
            return weights_dir

    return None
