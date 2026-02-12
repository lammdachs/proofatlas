#!/usr/bin/env python3
"""
CLI entry point for proofatlas-pipeline.

This module provides the entry point for the `proofatlas-pipeline` command.
It imports and runs the main function from scripts/pipeline.py.
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
        from pipeline import main as pipeline_main
        pipeline_main()
    except ImportError:
        import subprocess
        script = root / "scripts" / "pipeline.py"
        if script.exists():
            sys.exit(subprocess.call([sys.executable, str(script)] + sys.argv[1:]))
        else:
            print(f"Error: Could not find pipeline.py script at {script}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
