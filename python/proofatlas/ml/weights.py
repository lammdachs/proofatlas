"""Model weight management utilities.

Functions for finding, loading, and managing trained model weights.
"""

from pathlib import Path
from typing import Optional


# Embeddings that use string input (clause text)
STRING_EMBEDDINGS = {"sentence"}

# Embeddings that use graph input (node features, adjacency, pooling)
GRAPH_EMBEDDINGS = {"gcn", "gat", "graphsage"}


def get_model_name(preset: dict) -> str:
    """Get model file name from preset config.

    Uses modular naming: {embedding}_{scorer}.pt
    Falls back to legacy naming for backwards compatibility.
    """
    # Modular design: embedding + scorer
    embedding = preset.get("embedding")
    scorer = preset.get("scorer")
    if embedding and scorer:
        return f"{embedding}_{scorer}"

    # Legacy: explicit model field
    if "model" in preset:
        return preset["model"]

    # Legacy: embedding field only
    if embedding:
        return embedding

    # Default
    return "gcn"


def get_embedding_type(preset: dict) -> Optional[str]:
    """Get embedding type category from preset config.

    Returns "graph", "string", or None for non-ML presets.
    """
    embedding = preset.get("embedding")
    if not embedding:
        return None

    if embedding in STRING_EMBEDDINGS:
        return "string"
    elif embedding in GRAPH_EMBEDDINGS:
        return "graph"
    else:
        raise ValueError(f"Unknown embedding: {embedding}")


def is_learned_selector(preset: dict) -> bool:
    """Check if preset requires trained weights."""
    # Check for modular embedding+scorer design
    if "embedding" in preset and "scorer" in preset:
        return True
    # Legacy: check for model field
    return "model" in preset


def find_weights(weights_dir: Path, preset: dict) -> Optional[Path]:
    """Find TorchScript model for a learned selector.

    Args:
        weights_dir: Directory containing model weights (.weights/)
        preset: Preset config dict with embedding/scorer fields

    Returns:
        Path to weights directory if model exists, or None if not found.
    """
    if not weights_dir.exists():
        return None

    model_name = get_model_name(preset)
    embedding_type = get_embedding_type(preset)

    if embedding_type == "graph":
        model_path = weights_dir / f"{model_name}.pt"
        if model_path.exists():
            return weights_dir
    elif embedding_type == "string":
        model_path = weights_dir / f"{model_name}.pt"
        tokenizer_path = weights_dir / f"{model_name}_tokenizer" / "tokenizer.json"
        if model_path.exists() and tokenizer_path.exists():
            return weights_dir

    return None
