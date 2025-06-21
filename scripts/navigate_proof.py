#!/usr/bin/env python
"""Command-line script for navigating proofs."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from proofatlas.navigator.proof_navigator import main

if __name__ == "__main__":
    main()