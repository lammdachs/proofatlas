#!/usr/bin/env python3
"""
Setup TPTP library and extract problem metadata.

Downloads TPTP to .tptp/ and extracts metadata to .data/problem_metadata.json

Usage:
    python scripts/setup_tptp.py
    python scripts/setup_tptp.py --force  # Re-download and re-extract
"""

import json
import tarfile
import time
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set, Tuple


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_tptp_config() -> dict:
    """Load TPTP configuration."""
    config_path = get_project_root() / "configs" / "tptp.json"
    with open(config_path) as f:
        return json.load(f)


# =============================================================================
# TPTP Download
# =============================================================================

def download_tptp(force: bool = False) -> Path:
    """Download and extract TPTP library.

    Returns the path to the Problems directory.
    """
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

    # Check if already installed
    if problems_dir.exists() and not force:
        print(f"TPTP v{version} is already installed.")
        print("Use --force to re-download.")
        return problems_dir

    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)

    # Download
    archive_path = target_dir / f"TPTP-v{version}.tgz"
    if not archive_path.exists() or force:
        print(f"Downloading TPTP v{version}...")
        urllib.request.urlretrieve(source, archive_path, _download_progress)
        print()  # newline after progress
    else:
        print("Archive already exists, skipping download.")

    # Extract
    print("Extracting...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(target_dir)

    # Verify
    if not problems_dir.exists():
        raise RuntimeError("Installation verification failed - Problems directory not found")

    # Count problems
    problem_count = sum(1 for _ in problems_dir.rglob("*.p"))
    print(f"\nTPTP v{version} installed successfully!")
    print(f"Problems directory: {problems_dir}")
    print(f"Total problems: {problem_count}")

    return problems_dir


def _download_progress(block_num: int, block_size: int, total_size: int):
    """Progress callback for urlretrieve."""
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
# =============================================================================

@dataclass
class ProblemMetadata:
    path: str
    domain: str
    status: str  # unsatisfiable, satisfiable, unknown
    format: str  # cnf, fof
    has_equality: bool
    is_unit_only: bool
    rating: float
    num_clauses: int
    num_axioms: int
    num_conjectures: int
    max_clause_size: int
    num_predicates: int
    num_functions: int
    num_constants: int
    max_term_depth: int


def extract_header_info(content: str) -> Tuple[str, float]:
    """Extract status and rating from the header comments."""
    header = content[:4096]

    # Extract status
    status = 'unknown'
    idx = header.lower().find('% status')
    if idx != -1:
        line_end = header.find('\n', idx)
        line = header[idx:line_end if line_end != -1 else idx + 100].lower()
        if 'unsatisfiable' in line or 'theorem' in line or 'contrasatisfiable' in line:
            status = 'unsatisfiable'
        elif 'satisfiable' in line or 'countersatisfiable' in line:
            status = 'satisfiable'

    # Extract rating
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

    return status, rating


def quick_format_check(content: str) -> Optional[str]:
    """Check format from first 8KB of content."""
    header = content[:8192]

    if 'tff(' in header or 'thf(' in header:
        return None

    if 'fof(' in header:
        return 'fof'
    elif 'cnf(' in header:
        return 'cnf'

    return None


def parse_clauses_fast(content: str) -> Tuple[int, int, int, int, bool, bool, int, Set[str], Set[str], Set[str]]:
    """Fast single-pass parser for TPTP clauses."""
    num_clauses = 0
    num_axioms = 0
    num_conjectures = 0
    max_clause_size = 0
    all_unit = True
    has_equality = False
    max_depth = 0

    predicates: Set[str] = set()
    functions: Set[str] = set()
    constants: Set[str] = set()

    i = 0
    n = len(content)

    while i < n:
        while i < n and content[i] in ' \t\n\r':
            i += 1

        if i >= n:
            break

        if content[i] == '%':
            while i < n and content[i] != '\n':
                i += 1
            continue

        if content[i:i+7] == 'include':
            while i < n and content[i] != '.':
                i += 1
            i += 1
            continue

        is_cnf = content[i:i+4] == 'cnf('
        is_fof = content[i:i+4] == 'fof('

        if not is_cnf and not is_fof:
            while i < n and content[i] != '\n':
                i += 1
            continue

        i += 4
        num_clauses += 1

        while i < n and content[i] != ',':
            i += 1
        i += 1

        while i < n and content[i] in ' \t\n\r':
            i += 1

        role_start = i
        while i < n and content[i] not in ', \t\n\r':
            i += 1
        role = content[role_start:i].lower()

        if role in ('axiom', 'hypothesis', 'definition', 'lemma'):
            num_axioms += 1
        elif role in ('conjecture', 'negated_conjecture'):
            num_conjectures += 1

        while i < n and content[i] != ',':
            i += 1
        i += 1

        paren_depth = 0
        literal_count = 1
        current_max_depth = 0
        ident = ''
        prev_non_space = ''

        while i < n:
            c = content[i]

            if c == '(':
                paren_depth += 1
                current_max_depth = max(current_max_depth, paren_depth)
                if ident:
                    if prev_non_space in ('', '(', '|', '&', '~', ',', '['):
                        predicates.add(ident)
                    else:
                        functions.add(ident)
                    ident = ''
                prev_non_space = c
                i += 1

            elif c == ')':
                if paren_depth == 0:
                    break
                paren_depth -= 1
                if ident and ident[0].islower():
                    if ident not in predicates and ident not in functions:
                        if ident not in ('true', 'false', 'and', 'or', 'not', 'implies'):
                            constants.add(ident)
                ident = ''
                prev_non_space = c
                i += 1

            elif c == '|' and paren_depth == 0:
                literal_count += 1
                if ident and ident[0].islower():
                    if ident not in predicates and ident not in functions:
                        if ident not in ('true', 'false', 'and', 'or', 'not', 'implies'):
                            constants.add(ident)
                ident = ''
                prev_non_space = c
                i += 1

            elif c == '=' and i + 1 < n:
                next_c = content[i + 1] if i + 1 < n else ''
                prev_c = content[i - 1] if i > 0 else ''
                if prev_c not in ('<', '!', '=') and next_c not in ('>', '='):
                    has_equality = True
                if prev_c == '!' and next_c != '>':
                    has_equality = True
                if ident and ident[0].islower():
                    if ident not in predicates and ident not in functions:
                        if ident not in ('true', 'false', 'and', 'or', 'not', 'implies'):
                            constants.add(ident)
                ident = ''
                prev_non_space = c
                i += 1

            elif c.isalnum() or c == '_':
                ident += c
                i += 1

            elif c in ' \t\n\r':
                if ident and ident[0].islower():
                    j = i
                    while j < n and content[j] in ' \t\n\r':
                        j += 1
                    if j < n and content[j] == '(':
                        pass
                    else:
                        if ident not in predicates and ident not in functions:
                            if ident not in ('true', 'false', 'and', 'or', 'not', 'implies'):
                                constants.add(ident)
                        ident = ''
                i += 1

            elif c == ',':
                if ident and ident[0].islower():
                    if ident not in predicates and ident not in functions:
                        if ident not in ('true', 'false', 'and', 'or', 'not', 'implies'):
                            constants.add(ident)
                ident = ''
                prev_non_space = c
                i += 1

            else:
                if ident and ident[0].islower():
                    if ident not in predicates and ident not in functions:
                        if ident not in ('true', 'false', 'and', 'or', 'not', 'implies'):
                            constants.add(ident)
                ident = ''
                if c not in ' \t\n\r':
                    prev_non_space = c
                i += 1

        while i < n and content[i] != '.':
            i += 1
        i += 1

        max_clause_size = max(max_clause_size, literal_count)
        max_depth = max(max_depth, current_max_depth)

        if literal_count > 1:
            all_unit = False

    return (num_clauses, num_axioms, num_conjectures, max_clause_size,
            all_unit, has_equality, max_depth, predicates, functions, constants)


def extract_metadata(problem_path: Path, tptp_root: Path) -> Optional[ProblemMetadata]:
    """Extract metadata from a single problem file."""
    try:
        content = problem_path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return None

    fmt = quick_format_check(content)
    if fmt is None:
        return None

    if 'tff(' in content or 'thf(' in content:
        return None

    status, rating = extract_header_info(content)

    (num_clauses, num_axioms, num_conjectures, max_clause_size,
     all_unit, has_equality, max_depth, predicates, functions, constants) = parse_clauses_fast(content)

    relative_path = problem_path.relative_to(tptp_root)
    domain = problem_path.parent.name

    return ProblemMetadata(
        path=str(relative_path),
        domain=domain,
        status=status,
        format=fmt,
        has_equality=has_equality,
        is_unit_only=all_unit and num_clauses > 0,
        rating=rating,
        num_clauses=num_clauses,
        num_axioms=num_axioms,
        num_conjectures=num_conjectures,
        max_clause_size=max_clause_size,
        num_predicates=len(predicates),
        num_functions=len(functions),
        num_constants=len(constants),
        max_term_depth=max_depth,
    )


def extract_all_metadata(problems_dir: Path) -> List[ProblemMetadata]:
    """Extract metadata from all TPTP problems."""
    print("\nExtracting problem metadata...")
    print("=" * 50)

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
        bar_len = 30
        filled = int(bar_len * (i + 1) / total_files)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r[{bar}] {pct:5.1f}% ({i+1}/{total_files})", end="", flush=True)

        metadata = extract_metadata(problem_file, problems_dir)
        if metadata is not None:
            problems.append(metadata)

    elapsed = time.time() - start_time
    print()

    return problems, elapsed


def save_metadata(problems: List[ProblemMetadata], elapsed: float):
    """Save extracted metadata to JSON."""
    output_path = get_project_root() / ".data" / "problem_metadata.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Summary statistics
    num_cnf = sum(1 for p in problems if p.format == 'cnf')
    num_fof = sum(1 for p in problems if p.format == 'fof')
    num_unsat = sum(1 for p in problems if p.status == 'unsatisfiable')
    num_sat = sum(1 for p in problems if p.status == 'satisfiable')
    num_unknown = sum(1 for p in problems if p.status == 'unknown')
    num_with_eq = sum(1 for p in problems if p.has_equality)
    num_unit = sum(1 for p in problems if p.is_unit_only)

    output = {
        "version": "1.0",
        "tptp_version": "9.0.0",
        "generated": datetime.now().isoformat(),
        "num_problems": len(problems),
        "summary": {
            "total": len(problems),
            "by_format": {"cnf": num_cnf, "fof": num_fof},
            "by_status": {"unsatisfiable": num_unsat, "satisfiable": num_sat, "unknown": num_unknown},
            "with_equality": num_with_eq,
            "unit_only": num_unit,
        },
        "problems": [asdict(p) for p in problems],
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nExtracted metadata for {len(problems)} problems in {elapsed:.1f}s")
    print(f"  CNF: {num_cnf}, FOF: {num_fof}")
    print(f"  Unsatisfiable: {num_unsat}, Satisfiable: {num_sat}, Unknown: {num_unknown}")
    print(f"  With equality: {num_with_eq}, Unit only: {num_unit}")
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

    # Download TPTP
    problems_dir = download_tptp(force=args.force)

    # Extract metadata
    if not args.skip_metadata:
        problems, elapsed = extract_all_metadata(problems_dir)
        save_metadata(problems, elapsed)

    print("\nSetup complete!")


if __name__ == "__main__":
    main()
