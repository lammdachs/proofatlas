#!/usr/bin/env python3
"""
CLI entry point for proofatlas-train.

This module provides the entry point for the `proofatlas-train` command.
It imports and runs the main function from scripts/train.py.
"""

import sys
from pathlib import Path


def find_project_root() -> Path:
    """Find the proofatlas project root."""
    root = Path(__file__).parent.parent.parent.parent.parent

    candidates = [root, Path.cwd()]
    for candidate in candidates:
        if (candidate / "configs" / "proofatlas.json").exists():
            return candidate

    path = Path.cwd()
    while path != path.parent:
        if (path / "configs" / "proofatlas.json").exists():
            return path
        path = path.parent

    return Path.cwd()


def main():
    root = find_project_root()
    scripts_dir = root / "scripts"
    if scripts_dir.exists():
        sys.path.insert(0, str(scripts_dir))

    try:
        from train import main as train_main
        train_main()
    except ImportError:
        import subprocess
        script = root / "scripts" / "train.py"
        if script.exists():
            sys.exit(subprocess.call([sys.executable, str(script)] + sys.argv[1:]))
        else:
            print(f"Error: Could not find train.py script at {script}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
