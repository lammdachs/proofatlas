#!/usr/bin/env python3
"""
CLI entry point for proofatlas-bench.

This module provides the entry point for the `proofatlas-bench` command.
It imports and runs the main function from scripts/bench.py.
"""

import sys
from pathlib import Path


def main():
    # Find project root and add scripts to path
    root = Path(__file__).parent.parent.parent.parent.parent  # cli -> proofatlas -> python -> project root

    # Try to find the project root by looking for configs
    candidates = [root, Path.cwd()]
    for candidate in candidates:
        if (candidate / "configs" / "tptp.json").exists():
            root = candidate
            break
    else:
        # Walk up from cwd
        path = Path.cwd()
        while path != path.parent:
            if (path / "configs" / "tptp.json").exists():
                root = path
                break
            path = path.parent

    scripts_dir = root / "scripts"
    if scripts_dir.exists():
        sys.path.insert(0, str(scripts_dir))

    # Import and run
    try:
        from bench import main as bench_main
        bench_main()
    except ImportError:
        # Fallback: execute the script directly
        import subprocess
        script = root / "scripts" / "bench.py"
        if script.exists():
            sys.exit(subprocess.call([sys.executable, str(script)] + sys.argv[1:]))
        else:
            print(f"Error: Could not find bench.py script at {script}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
