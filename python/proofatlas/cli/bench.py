#!/usr/bin/env python3
"""
CLI entry point for proofatlas-bench.

This module provides the entry point for the `proofatlas-bench` command.
It imports and runs the main function from scripts/bench.py.
"""

import sys
from pathlib import Path

from proofatlas.paths import find_project_root


def main():
    root = find_project_root()
    scripts_dir = root / "scripts"
    if scripts_dir.exists():
        sys.path.insert(0, str(scripts_dir))

    try:
        from bench import main as bench_main
        bench_main()
    except ImportError:
        import subprocess
        script = root / "scripts" / "bench.py"
        if script.exists():
            sys.exit(subprocess.call([sys.executable, str(script)] + sys.argv[1:]))
        else:
            print(f"Error: Could not find bench.py script at {script}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
