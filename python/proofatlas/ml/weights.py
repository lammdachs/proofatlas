"""Model weight management utilities.

Functions for finding, loading, and managing trained model weights.
"""

from pathlib import Path
from typing import Optional


# Encoders that use string input (clause text)
STRING_ENCODERS = {"sentence"}

# Encoders that use graph input (node features, adjacency, pooling)
GRAPH_ENCODERS = {"gcn", "gcn_struct"}

# Encoders that use only clause-level features (no structure or text)
FEATURES_ENCODERS = {"features"}


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
    elif encoder in FEATURES_ENCODERS:
        return "features"
    else:
        raise ValueError(f"Unknown encoder: {encoder}")



def is_learned_selector(preset: dict) -> bool:
    """Check if preset requires trained weights."""
    return "encoder" in preset and "scorer" in preset


def find_weights(weights_dir: Path, preset: dict, preset_name: str = None) -> Optional[Path]:
    """Find TorchScript model for a learned selector.

    Args:
        weights_dir: Directory containing model weights (.weights/)
        preset: Preset config dict with encoder/scorer fields
        preset_name: If given, look for {preset_name}.pt first, then
                     fall back to {encoder}_{scorer}.pt

    Returns:
        Path to weights directory if model exists, or None if not found.
    """
    if not weights_dir.exists():
        return None

    encoder_type = get_encoder_type(preset)

    # Try preset-specific name first, then generic encoder_scorer name
    names = []
    if preset_name:
        names.append(preset_name)
    names.append(get_model_name(preset))

    generic_name = get_model_name(preset)

    for model_name in names:
        if encoder_type in ("graph", "features"):
            if (weights_dir / f"{model_name}.pt").exists():
                # Ensure Rust can find it as {encoder}_{scorer}.pt
                if model_name != generic_name:
                    _ensure_symlink(weights_dir / f"{generic_name}.pt",
                                    weights_dir / f"{model_name}.pt")
                return weights_dir
        elif encoder_type == "string":
            model_path = weights_dir / f"{model_name}.pt"
            tokenizer_path = weights_dir / f"{model_name}_tokenizer" / "tokenizer.json"
            if model_path.exists() and tokenizer_path.exists():
                if model_name != generic_name:
                    _ensure_symlink(weights_dir / f"{generic_name}.pt",
                                    weights_dir / f"{model_name}.pt")
                return weights_dir

    return None


def _ensure_symlink(link_path: Path, target_path: Path):
    """Create or update a symlink."""
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(target_path.name)
