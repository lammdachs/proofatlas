#!/usr/bin/env python3
"""
Extract metadata from TPTP problems into a single JSON file.

Fast version - uses efficient single-pass parsing without expensive regex.

Extracts:
- path, domain, status, format (cnf/fof only)
- has_equality, is_unit_only, rating
- num_clauses, num_axioms, num_conjectures, max_clause_size
- num_predicates, num_functions, num_constants, max_term_depth
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set, Tuple
from dataclasses import dataclass, asdict


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


def get_tptp_path() -> Path:
    """Get the TPTP directory path."""
    base_path = Path(__file__).parent.parent / ".tptp/TPTP-v9.0.0/Problems"
    if not base_path.exists():
        raise FileNotFoundError(f"TPTP directory not found: {base_path}")
    return base_path


def extract_header_info(content: str) -> Tuple[str, float]:
    """Extract status and rating from the header comments (first 4KB)."""
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
        # Find digits after colon
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

    # Check for TFF/THF first - if present, skip
    if 'tff(' in header or 'thf(' in header:
        return None

    # Check for CNF/FOF
    if 'fof(' in header:
        return 'fof'
    elif 'cnf(' in header:
        return 'cnf'

    return None


def parse_clauses_fast(content: str) -> Tuple[int, int, int, int, bool, bool, int, Set[str], Set[str], Set[str]]:
    """
    Fast single-pass parser for TPTP clauses.

    Returns: (num_clauses, num_axioms, num_conjectures, max_clause_size,
              all_unit, has_equality, max_depth, predicates, functions, constants)
    """
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
        # Skip whitespace and comments
        while i < n and content[i] in ' \t\n\r':
            i += 1

        if i >= n:
            break

        # Skip comment lines
        if content[i] == '%':
            while i < n and content[i] != '\n':
                i += 1
            continue

        # Skip include statements
        if content[i:i+7] == 'include':
            while i < n and content[i] != '.':
                i += 1
            i += 1
            continue

        # Check for cnf( or fof(
        is_cnf = content[i:i+4] == 'cnf('
        is_fof = content[i:i+4] == 'fof('

        if not is_cnf and not is_fof:
            # Skip unknown content until next line
            while i < n and content[i] != '\n':
                i += 1
            continue

        i += 4  # Skip 'cnf(' or 'fof('
        num_clauses += 1

        # Skip clause name (until comma)
        while i < n and content[i] != ',':
            i += 1
        i += 1  # Skip comma

        # Skip whitespace
        while i < n and content[i] in ' \t\n\r':
            i += 1

        # Read role (until comma)
        role_start = i
        while i < n and content[i] not in ', \t\n\r':
            i += 1
        role = content[role_start:i].lower()

        if role in ('axiom', 'hypothesis', 'definition', 'lemma'):
            num_axioms += 1
        elif role in ('conjecture', 'negated_conjecture'):
            num_conjectures += 1

        # Skip to start of formula (after comma)
        while i < n and content[i] != ',':
            i += 1
        i += 1  # Skip comma

        # Now parse the formula until closing ).
        # Track: paren depth, literal count, max depth, symbols, equality
        paren_depth = 0
        literal_count = 1
        current_max_depth = 0
        formula_start = i

        # Track identifier being built
        ident = ''
        ident_start = -1
        prev_non_space = ''

        while i < n:
            c = content[i]

            if c == '(':
                paren_depth += 1
                current_max_depth = max(current_max_depth, paren_depth)

                # If we were building an identifier, it's a predicate or function
                if ident:
                    if prev_non_space in ('', '(', '|', '&', '~', ',', '['):
                        predicates.add(ident)
                    else:
                        functions.add(ident)
                    ident = ''

                prev_non_space = c
                i += 1

            elif c == ')':
                # Check if this is the end of the clause
                if paren_depth == 0:
                    # End of formula - look for the closing dot
                    break
                paren_depth -= 1

                # If we were building an identifier (constant)
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
                # Check for real equality (not =>, <=, ==)
                if prev_c not in ('<', '!', '=') and next_c not in ('>', '='):
                    has_equality = True
                # Also != is equality
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
                # Whitespace - save identifier if any
                if ident and ident[0].islower():
                    # Check next non-whitespace to see if it's a function call
                    j = i
                    while j < n and content[j] in ' \t\n\r':
                        j += 1
                    if j < n and content[j] == '(':
                        # It's a function/predicate, will be handled when we see (
                        pass
                    else:
                        # It's a constant
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

        # Skip to end of clause (the dot)
        while i < n and content[i] != '.':
            i += 1
        i += 1  # Skip dot

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

    # Quick format check
    fmt = quick_format_check(content)
    if fmt is None:
        return None

    # Also check full content for TFF/THF (might appear after header)
    if 'tff(' in content or 'thf(' in content:
        return None

    # Extract header info
    status, rating = extract_header_info(content)

    # Parse clauses in single pass
    (num_clauses, num_axioms, num_conjectures, max_clause_size,
     all_unit, has_equality, max_depth, predicates, functions, constants) = parse_clauses_fast(content)

    # Get relative path
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


def extract_all_metadata(verbose: bool = True) -> List[ProblemMetadata]:
    """Extract metadata from all TPTP problems."""
    tptp_path = get_tptp_path()
    problems = []

    # Precompute all problem files for progress bar
    if verbose:
        print("Scanning for problem files...")

    all_files = []
    domains = sorted([d for d in tptp_path.iterdir() if d.is_dir()])
    for domain_dir in domains:
        all_files.extend(sorted(domain_dir.glob("*.p")))

    total_files = len(all_files)
    if verbose:
        print(f"Found {total_files} problem files in {len(domains)} domains")

    # Process with progress bar
    for i, problem_file in enumerate(all_files):
        if verbose:
            pct = 100 * (i + 1) / total_files
            bar_len = 30
            filled = int(bar_len * (i + 1) / total_files)
            bar = "█" * filled + "░" * (bar_len - filled)
            # Show current file being processed
            print(f"\r[{bar}] {pct:5.1f}% ({i+1}/{total_files}) {problem_file.name:<30}", end="", flush=True)

        metadata = extract_metadata(problem_file, tptp_path)
        if metadata is not None:
            problems.append(metadata)

    if verbose:
        print()  # newline after progress bar

    return problems


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract TPTP problem metadata")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path(__file__).parent.parent / ".data/problem_metadata.json",
        help="Output JSON file"
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    print("Extracting TPTP problem metadata...")
    print("=" * 50)

    start_time = time.time()
    problems = extract_all_metadata(verbose=not args.quiet)
    elapsed = time.time() - start_time

    # Create output
    output = {
        "version": "1.0",
        "tptp_version": "9.0.0",
        "generated": datetime.now().isoformat(),
        "num_problems": len(problems),
        "problems": [asdict(p) for p in problems],
    }

    # Add summary statistics
    num_cnf = sum(1 for p in problems if p.format == 'cnf')
    num_fof = sum(1 for p in problems if p.format == 'fof')
    num_unsat = sum(1 for p in problems if p.status == 'unsatisfiable')
    num_sat = sum(1 for p in problems if p.status == 'satisfiable')
    num_unknown = sum(1 for p in problems if p.status == 'unknown')
    num_with_eq = sum(1 for p in problems if p.has_equality)
    num_unit = sum(1 for p in problems if p.is_unit_only)

    output["summary"] = {
        "total": len(problems),
        "by_format": {"cnf": num_cnf, "fof": num_fof},
        "by_status": {"unsatisfiable": num_unsat, "satisfiable": num_sat, "unknown": num_unknown},
        "with_equality": num_with_eq,
        "unit_only": num_unit,
    }

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nExtracted metadata for {len(problems)} problems in {elapsed:.1f}s")
    print(f"  CNF: {num_cnf}, FOF: {num_fof}")
    print(f"  Unsatisfiable: {num_unsat}, Satisfiable: {num_sat}, Unknown: {num_unknown}")
    print(f"  With equality: {num_with_eq}, Unit only: {num_unit}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
