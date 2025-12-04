#!/usr/bin/env python3
"""
Compare ProofAtlas with PyRes on TPTP problems.
Tests both provers on the same set of problems with equal timeouts.
"""

import os
import subprocess
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple

def get_tptp_base() -> Path:
    """Get the TPTP base directory."""
    return Path(__file__).parent.parent / ".tptp/TPTP-v9.0.0/Problems"

def get_problem_status(problem_path: Path) -> str:
    """Get the status of a TPTP problem."""
    try:
        with open(problem_path, 'r') as f:
            for line in f.readlines()[:50]:
                if 'Status' in line and ':' in line:
                    import re
                    match = re.search(r'Status\s*:\s*(\w+)', line)
                    if match:
                        return match.group(1).strip()
        return "Unknown"
    except:
        return "Unknown"

def load_problem_lists() -> Dict[str, List[str]]:
    """Load categorized problem lists, filtering for Unsatisfiable problems only."""
    lists_dir = Path(__file__).parent.parent / ".data/benchmark_lists"
    tptp_base = get_tptp_base()
    categories = {
        "unit_equalities": "unit_equalities_problems.txt",
        "cnf_without_equality": "cnf_without_equality_problems.txt",
        "cnf_with_equality": "cnf_with_equality_problems.txt",
        "fof_without_equality": "fof_without_equality_problems.txt",
        "fof_with_equality": "fof_with_equality_problems.txt"
    }

    problem_lists = {}
    for category, filename in categories.items():
        filepath = lists_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                lines = f.readlines()
                all_problems = [line.strip() for line in lines if line.strip() and not line.startswith('#')]

                # Filter for Unsatisfiable problems only
                unsatisfiable_problems = []
                for problem in all_problems:
                    problem_path = tptp_base / problem
                    if problem_path.exists():
                        status = get_problem_status(problem_path)
                        if status in ['Unsatisfiable', 'Theorem']:
                            unsatisfiable_problems.append(problem)

                problem_lists[category] = unsatisfiable_problems
                print(f"  {category}: {len(unsatisfiable_problems)}/{len(all_problems)} are Unsatisfiable")

    return problem_lists

def run_proofatlas(problem_path: Path, timeout: int = 5) -> Tuple[bool, float]:
    """Run ProofAtlas on a problem with timeout."""
    start_time = time.time()

    original_cwd = os.getcwd()
    rust_dir = Path(__file__).parent.parent / "rust"
    os.chdir(rust_dir)

    try:
        tptp_root = Path(__file__).parent.parent / ".tptp/TPTP-v9.0.0"
        cmd = ["cargo", "run", "--release", "--bin", "prove", "--",
               str(problem_path), "--timeout", str(timeout), "--include", str(tptp_root)]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 1)
        time_taken = time.time() - start_time

        output = result.stdout + result.stderr
        solved = "THEOREM PROVED" in output or "Proof found!" in output

        return solved, time_taken

    except subprocess.TimeoutExpired:
        return False, timeout
    except Exception:
        return False, timeout
    finally:
        os.chdir(original_cwd)

def run_pyres(problem_path: Path, timeout: int = 5) -> Tuple[bool, float]:
    """Run PyRes on a problem with timeout."""
    start_time = time.time()

    pyres_dir = Path(__file__).parent.parent / ".pyres"

    # Determine if problem is CNF or FOF based on extension
    if problem_path.suffix == ".p":
        # Check file content for fof() declarations
        try:
            with open(problem_path, 'r') as f:
                content = f.read(500)  # Read first 500 chars
                is_fof = 'fof(' in content
        except:
            is_fof = False
    else:
        is_fof = problem_path.name.endswith('+1.p') or problem_path.name.endswith('+2.p')

    if is_fof:
        prover = pyres_dir / "pyres-fof.py"
        args = ["-tifbp", "-HPickGiven5", "-nlargest"]
    else:
        prover = pyres_dir / "pyres-cnf.py"
        args = ["-tfb", "-HPickGiven5", "-nsmallest"]

    try:
        cmd = [str(prover)] + args + [str(problem_path)]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 1)
        time_taken = time.time() - start_time

        output = result.stdout + result.stderr
        # PyRes outputs "SZS status" on success
        solved = ("SZS status Unsatisfiable" in output or
                  "SZS status Theorem" in output or
                  "$false" in output)

        return solved, time_taken

    except subprocess.TimeoutExpired:
        return False, timeout
    except Exception:
        return False, timeout

def compare_on_problems(category: str, problems: List[str], timeout: int = 5):
    """Compare both provers on a set of problems."""
    tptp_base = get_tptp_base()

    print(f"\n{'='*70}")
    print(f"{category.replace('_', ' ').upper()}")
    print(f"{'='*70}")
    print(f"{'Problem':<30} {'ProofAtlas':<15} {'PyRes':<15} {'Winner':<10}")
    print(f"{'-'*70}")

    results = {
        'proofatlas': {'solved': 0, 'time': 0},
        'pyres': {'solved': 0, 'time': 0},
        'both': 0,
        'neither': 0
    }

    for problem in problems:
        problem_path = tptp_base / problem

        if not problem_path.exists():
            continue

        # Run both provers
        pa_solved, pa_time = run_proofatlas(problem_path, timeout)
        pr_solved, pr_time = run_pyres(problem_path, timeout)

        # Format results
        pa_status = f"✓ {pa_time:.2f}s" if pa_solved else "✗"
        pr_status = f"✓ {pr_time:.2f}s" if pr_solved else "✗"

        if pa_solved and pr_solved:
            if pa_time < pr_time:
                winner = "ProofAtlas"
            elif pr_time < pa_time:
                winner = "PyRes"
            else:
                winner = "Tie"
            results['both'] += 1
        elif pa_solved:
            winner = "ProofAtlas"
        elif pr_solved:
            winner = "PyRes"
        else:
            winner = "-"
            results['neither'] += 1

        if pa_solved:
            results['proofatlas']['solved'] += 1
            results['proofatlas']['time'] += pa_time
        if pr_solved:
            results['pyres']['solved'] += 1
            results['pyres']['time'] += pr_time

        # Shorten problem name for display
        short_name = problem.split('/')[-1]
        print(f"{short_name:<30} {pa_status:<15} {pr_status:<15} {winner:<10}")

    # Category summary
    total = len(problems)
    print(f"\n{'-'*70}")
    print(f"Category Summary:")
    print(f"  ProofAtlas: {results['proofatlas']['solved']}/{total} solved " +
          f"({results['proofatlas']['solved']/total*100:.0f}%)")
    if results['proofatlas']['solved'] > 0:
        print(f"    Avg time: {results['proofatlas']['time']/results['proofatlas']['solved']:.2f}s")

    print(f"  PyRes:      {results['pyres']['solved']}/{total} solved " +
          f"({results['pyres']['solved']/total*100:.0f}%)")
    if results['pyres']['solved'] > 0:
        print(f"    Avg time: {results['pyres']['time']/results['pyres']['solved']:.2f}s")

    print(f"  Both solved: {results['both']}")
    print(f"  Neither solved: {results['neither']}")

    return results

def main():
    """Main comparison function."""
    import argparse

    parser = argparse.ArgumentParser(description='Compare ProofAtlas with PyRes')
    parser.add_argument('--problems', type=int, default=10,
                       help='Number of problems per category (default: 10)')
    parser.add_argument('--timeout', type=int, default=5,
                       help='Timeout per problem in seconds (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()
    random.seed(args.seed)

    print("="*70)
    print(f"PROOFATLAS vs PYRES COMPARISON")
    print(f"Testing {args.problems} problems per category, {args.timeout}s timeout")
    print("="*70)

    # Load problem lists
    problem_lists = load_problem_lists()

    if not problem_lists:
        print("Error: No problem lists found.")
        return

    # Overall results
    overall = {
        'proofatlas': {'solved': 0, 'time': 0},
        'pyres': {'solved': 0, 'time': 0}
    }

    # Test each category
    for category, problems in problem_lists.items():
        sample = random.sample(problems, min(args.problems, len(problems)))
        results = compare_on_problems(category, sample, args.timeout)

        overall['proofatlas']['solved'] += results['proofatlas']['solved']
        overall['proofatlas']['time'] += results['proofatlas']['time']
        overall['pyres']['solved'] += results['pyres']['solved']
        overall['pyres']['time'] += results['pyres']['time']

    # Overall summary
    total_problems = args.problems * len(problem_lists)
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")

    pa_solved = overall['proofatlas']['solved']
    pr_solved = overall['pyres']['solved']

    print(f"ProofAtlas: {pa_solved}/{total_problems} solved ({pa_solved/total_problems*100:.1f}%)")
    if pa_solved > 0:
        print(f"  Average solve time: {overall['proofatlas']['time']/pa_solved:.2f}s")

    print(f"PyRes:      {pr_solved}/{total_problems} solved ({pr_solved/total_problems*100:.1f}%)")
    if pr_solved > 0:
        print(f"  Average solve time: {overall['pyres']['time']/pr_solved:.2f}s")

    print(f"\nWinner: ", end="")
    if pa_solved > pr_solved:
        print(f"ProofAtlas (+{pa_solved - pr_solved} problems)")
    elif pr_solved > pa_solved:
        print(f"PyRes (+{pr_solved - pa_solved} problems)")
    else:
        print("Tie!")

if __name__ == "__main__":
    main()