"""Canonical project root resolution for all ProofAtlas entry points.

Works regardless of the current working directory by resolving from the
installed package location. Supports ``PROOFATLAS_ROOT`` env-var override.
"""

import os
from pathlib import Path

_MARKER = ("configs", "proofatlas.json")


def _has_marker(path: Path) -> bool:
    return (path / _MARKER[0] / _MARKER[1]).exists()


def find_project_root() -> Path:
    """Return the ProofAtlas project root directory.

    Resolution order:
    1. ``PROOFATLAS_ROOT`` environment variable (if set and valid).
    2. Relative to the installed package (``python/proofatlas/`` -> 2 parents up).
    3. Walk upward from the current working directory.

    Raises ``FileNotFoundError`` with an actionable message when the root
    cannot be determined.
    """
    # 1. Explicit env-var override
    env = os.environ.get("PROOFATLAS_ROOT")
    if env:
        p = Path(env).resolve()
        if _has_marker(p):
            return p
        raise FileNotFoundError(
            f"PROOFATLAS_ROOT={env} does not contain {'/'.join(_MARKER)}"
        )

    # 2. Relative to this file: python/proofatlas/paths.py -> project root
    pkg_root = Path(__file__).resolve().parent.parent.parent
    if _has_marker(pkg_root):
        return pkg_root

    # 3. Walk upward from cwd
    path = Path.cwd().resolve()
    while True:
        if _has_marker(path):
            return path
        parent = path.parent
        if parent == path:
            break
        path = parent

    raise FileNotFoundError(
        "Could not find proofatlas project root (no configs/proofatlas.json found).\n"
        "Either run from inside the project tree or set PROOFATLAS_ROOT."
    )
