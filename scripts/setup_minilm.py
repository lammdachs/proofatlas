#!/usr/bin/env python3
"""
Download and export base MiniLM model for trace embedding.

The exported model produces raw 384-D embeddings used by the Rust backend
to pre-compute node/clause embeddings when saving proof traces.

Usage:
    python scripts/setup_minilm.py          # Install to .weights/
    python scripts/setup_minilm.py --force  # Re-download even if exists

Output files:
    .weights/base_minilm.pt                 - TorchScript model
    .weights/base_minilm_tokenizer/         - HuggingFace tokenizer
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Setup base MiniLM for trace embedding")
    parser.add_argument("--force", "-f", action="store_true", help="Force re-download")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    weights_dir = root / ".weights"
    model_path = weights_dir / "base_minilm.pt"
    tokenizer_path = weights_dir / "base_minilm_tokenizer" / "tokenizer.json"

    if model_path.exists() and tokenizer_path.exists() and not args.force:
        print(f"Base MiniLM already installed at {weights_dir}")
        print(f"  Model:     {model_path} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"  Tokenizer: {tokenizer_path.parent}")
        print()
        print("To reinstall, use --force")
        return

    # Check dependencies
    try:
        import torch  # noqa: F401
    except ImportError:
        print("Error: PyTorch is required. Install it first:", file=sys.stderr)
        print("  pip install torch", file=sys.stderr)
        sys.exit(1)

    try:
        import transformers  # noqa: F401
    except ImportError:
        print("Error: transformers is required. Install it first:", file=sys.stderr)
        print("  pip install transformers", file=sys.stderr)
        sys.exit(1)

    # Add package to path
    sys.path.insert(0, str(root / "python"))
    from proofatlas.ml.export import ensure_base_minilm

    print("Base MiniLM Setup")
    print("=" * 50)
    print(f"Model:  sentence-transformers/all-MiniLM-L6-v2")
    print(f"Target: {weights_dir}")
    print()

    if args.force:
        # Remove existing files so ensure_base_minilm re-downloads
        model_path.unlink(missing_ok=True)
        if tokenizer_path.exists():
            import shutil
            shutil.rmtree(tokenizer_path.parent)

    print("Downloading and exporting...")
    ensure_base_minilm(weights_dir)

    print()
    print(f"Base MiniLM installed successfully!")
    print(f"  Model:     {model_path} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  Tokenizer: {tokenizer_path.parent}")


if __name__ == "__main__":
    main()
