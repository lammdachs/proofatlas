#!/usr/bin/env python3
"""
Setup TPTP library and extract problem metadata.

Downloads TPTP to .tptp/ and writes .tptp/index.json

Usage:
    python scripts/setup_tptp.py
    python scripts/setup_tptp.py --force  # Re-download and re-extract
"""

import json
import re
import tarfile
import time
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


INCLUDE_RE = re.compile(r"^include\('([^']+)'\s*(?:,\s*\[[^\]]*\])?\s*\)\.", re.MULTILINE)


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def load_tptp_config() -> dict:
    config_path = get_project_root() / "configs" / "tptp.json"
    with open(config_path) as f:
        return json.load(f)


# =============================================================================
# TPTP Download
# =============================================================================

def download_tptp(force: bool = False) -> Path:
    """Download and extract TPTP library. Returns the Problems directory."""
    config = load_tptp_config()
    version = config["version"]
    source = config["source"]

    target_dir = get_project_root() / ".tptp"
    tptp_dir = target_dir / f"TPTP-v{version}"
    problems_dir = tptp_dir / "Problems"

    print("TPTP Setup")
    print("=" * 50)
    print(f"Version: {version}")
    print(f"Source:  {source}")
    print(f"Target:  {target_dir}")
    print()

    if problems_dir.exists() and not force:
        print(f"TPTP v{version} is already installed.")
        print("Use --force to re-download.")
        return problems_dir

    target_dir.mkdir(parents=True, exist_ok=True)

    archive_path = target_dir / f"TPTP-v{version}.tgz"
    if not archive_path.exists() or force:
        print(f"Downloading TPTP v{version}...")
        urllib.request.urlretrieve(source, archive_path, _download_progress)
        print()
    else:
        print("Archive already exists, skipping download.")

    print("Extracting...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(target_dir)

    if not problems_dir.exists():
        raise RuntimeError("Installation verification failed - Problems directory not found")

    problem_count = sum(1 for _ in problems_dir.rglob("*.p"))
    print(f"\nTPTP v{version} installed successfully!")
    print(f"Problems directory: {problems_dir}")
    print(f"Total problems: {problem_count}")

    return problems_dir


def _download_progress(block_num: int, block_size: int, total_size: int):
    if total_size > 0:
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        bar_len = 30
        filled = int(bar_len * percent / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\r[{bar}] {percent:5.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)


# =============================================================================
# Metadata Extraction
#
# All metadata is extracted from the TPTP header comments and include
# directives — we do not parse clause bodies.
# =============================================================================

@dataclass
class ProblemMetadata:
    path: str
    domain: str
    status: str       # unsatisfiable, satisfiable, unknown
    format: str       # cnf, fof
    has_equality: bool
    rating: float
    file_size: int
    includes: List[str]
    total_size: int


def extract_metadata(problem_path: Path, tptp_root: Path) -> Optional[ProblemMetadata]:
    """Extract metadata from header comments and include directives."""
    try:
        content = problem_path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return None

    # Determine format — reject tff/thf
    fmt = _detect_format(content)
    if fmt is None:
        return None

    status, rating, has_equality = _parse_header(content)

    file_size = problem_path.stat().st_size
    includes, axiom_size = _resolve_includes(content, tptp_root)

    problems_dir = tptp_root / "Problems"
    return ProblemMetadata(
        path=str(problem_path.relative_to(problems_dir)),
        domain=problem_path.parent.name,
        status=status,
        format=fmt,
        has_equality=has_equality,
        rating=rating,
        file_size=file_size,
        includes=includes,
        total_size=file_size + axiom_size,
    )


def _detect_format(content: str) -> Optional[str]:
    """Return 'cnf' or 'fof' if the file uses only those formats, else None."""
    # Check first 8KB for a quick decision; fall back to full scan for
    # files where formulas start late (many include directives).
    probe = content[:8192]
    if 'tff(' in probe or 'thf(' in probe:
        return None
    if 'fof(' in probe:
        return 'fof'
    if 'cnf(' in probe:
        return 'cnf'

    # Full scan for files with large headers
    if 'tff(' in content or 'thf(' in content:
        return None
    if 'fof(' in content:
        return 'fof'
    if 'cnf(' in content:
        return 'cnf'
    return None


def _parse_header(content: str) -> Tuple[str, float, bool]:
    """Extract status, rating, and has_equality from TPTP header comments."""
    header = content[:4096]

    status = 'unknown'
    idx = header.lower().find('% status')
    if idx != -1:
        line_end = header.find('\n', idx)
        line = header[idx:line_end if line_end != -1 else idx + 100].lower()
        if 'unsatisfiable' in line or 'theorem' in line or 'contrasatisfiable' in line:
            status = 'unsatisfiable'
        elif 'satisfiable' in line or 'countersatisfiable' in line:
            status = 'satisfiable'

    rating = 0.0
    idx = header.lower().find('% rating')
    if idx != -1:
        portion = header[idx:idx + 50]
        colon_idx = portion.find(':')
        if colon_idx != -1:
            num_str = ''
            for c in portion[colon_idx + 1:]:
                if c.isdigit() or c == '.':
                    num_str += c
                elif num_str:
                    break
            if num_str:
                try:
                    rating = float(num_str)
                except ValueError:
                    pass

    # "N equ" in the syntax section where N > 0 means equality is present
    has_equality = False
    equ_match = re.search(r'(\d+)\s+equ', header)
    if equ_match and int(equ_match.group(1)) > 0:
        has_equality = True

    return status, rating, has_equality


def _resolve_includes(content: str, tptp_root: Path) -> Tuple[List[str], int]:
    """Extract include directives and sum referenced axiom file sizes."""
    includes = []
    axiom_size = 0
    for m in INCLUDE_RE.finditer(content):
        rel_path = m.group(1)
        includes.append(rel_path)
        ax_path = tptp_root / rel_path
        if ax_path.exists():
            axiom_size += ax_path.stat().st_size
    return includes, axiom_size


# =============================================================================
# Batch extraction
# =============================================================================

def extract_all_metadata(problems_dir: Path) -> Tuple[List[ProblemMetadata], float]:
    """Extract metadata from all TPTP problems."""
    print("\nExtracting problem metadata...")
    print("=" * 50)

    tptp_root = problems_dir.parent

    all_files = []
    domains = sorted([d for d in problems_dir.iterdir() if d.is_dir()])
    for domain_dir in domains:
        all_files.extend(sorted(domain_dir.glob("*.p")))

    total_files = len(all_files)
    print(f"Found {total_files} problem files in {len(domains)} domains")

    problems = []
    start_time = time.time()

    for i, problem_file in enumerate(all_files):
        pct = 100 * (i + 1) / total_files
        filled = int(30 * (i + 1) / total_files)
        bar = "█" * filled + "░" * (30 - filled)
        print(f"\r[{bar}] {pct:5.1f}% ({i+1}/{total_files})", end="", flush=True)

        metadata = extract_metadata(problem_file, tptp_root)
        if metadata is not None:
            problems.append(metadata)

    elapsed = time.time() - start_time
    print()

    return problems, elapsed


def save_metadata(problems: List[ProblemMetadata], tptp_dir: Path, elapsed: float):
    """Save extracted metadata to .tptp/index.json."""
    output_path = tptp_dir / "index.json"

    output = {
        "tptp_version": "9.0.0",
        "generated": datetime.now().isoformat(),
        "problems": [asdict(p) for p in problems],
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    num_cnf = sum(1 for p in problems if p.format == 'cnf')
    num_fof = sum(1 for p in problems if p.format == 'fof')
    num_unsat = sum(1 for p in problems if p.status == 'unsatisfiable')
    num_sat = sum(1 for p in problems if p.status == 'satisfiable')
    num_unknown = sum(1 for p in problems if p.status == 'unknown')

    print(f"\nExtracted metadata for {len(problems)} problems in {elapsed:.1f}s")
    print(f"  CNF: {num_cnf}, FOF: {num_fof}")
    print(f"  Unsatisfiable: {num_unsat}, Satisfiable: {num_sat}, Unknown: {num_unknown}")
    print(f"\nSaved to {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Setup TPTP library and extract problem metadata"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download and re-extraction"
    )
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Skip metadata extraction (download only)"
    )

    args = parser.parse_args()

    problems_dir = download_tptp(force=args.force)

    if not args.skip_metadata:
        problems, elapsed = extract_all_metadata(problems_dir)
        save_metadata(problems, problems_dir.parent, elapsed)

    print("\nSetup complete!")


if __name__ == "__main__":
    main()
