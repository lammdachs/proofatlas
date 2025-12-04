#!/usr/bin/env python3
"""
Compare ProofAtlas (with SelectLargestNegative) with Vampire on TPTP problems.
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

def run_proofatlas_with_largest_neg(problem_path: Path, timeout: int = 5) -> Tuple[bool, float]:
    """Run ProofAtlas with SelectLargestNegative on a problem with timeout."""
    start_time = time.time()

    original_cwd = os.getcwd()
    rust_dir = Path(__file__).parent.parent / "rust"
    os.chdir(rust_dir)

    try:
        tptp_root = Path(__file__).parent.parent / ".tptp/TPTP-v9.0.0"
        cmd = ["cargo", "run", "--release", "--bin", "prove", "--",
               str(problem_path),
               "--timeout", str(timeout),
               "--literal-selection", "largest_negative",
               "--include", str(tptp_root)]

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

def run_vampire(problem_path: Path, timeout: int = 5, selection: str = "0") -> Tuple[bool, float]:
    """Run Vampire on a problem with timeout."""
    start_time = time.time()

    vampire_bin = Path(__file__).parent.parent / ".vampire" / "vampire"

    try:
        # Use specified clause selection and disable avatar (--avatar off)
        cmd = [str(vampire_bin), "-s", selection, "--avatar", "off", "-t", str(timeout), str(problem_path)]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 2)
        time_taken = time.time() - start_time

        output = result.stdout + result.stderr
        # Vampire outputs "SZS status Unsatisfiable" or "SZS status Theorem" on success
        solved = ("SZS status Unsatisfiable" in output or
                  "SZS status Theorem" in output or
                  "Refutation found" in output)

        return solved, time_taken

    except subprocess.TimeoutExpired:
        return False, timeout
    except Exception:
        return False, timeout

def compare_on_problems(category: str, problems: List[str], timeout: int = 5, vampire_selection: str = "0"):
    """Compare both provers on a set of problems."""
    tptp_base = get_tptp_base()

    print(f"\n{'='*70}")
    print(f"{category.replace('_', ' ').upper()}")
    print(f"{'='*70}")
    print(f"{'Problem':<30} {'PA(LargestNeg)':<18} {'Vampire':<15} {'Winner':<10}")
    print(f"{'-'*70}")

    results = {
        'proofatlas': {'solved': 0, 'time': 0},
        'vampire': {'solved': 0, 'time': 0},
        'both': 0,
        'neither': 0
    }

    for problem in problems:
        problem_path = tptp_base / problem

        if not problem_path.exists():
            continue

        # Run both provers
        pa_solved, pa_time = run_proofatlas_with_largest_neg(problem_path, timeout)
        vamp_solved, vamp_time = run_vampire(problem_path, timeout, vampire_selection)

        # Format results
        pa_status = f"✓ {pa_time:.2f}s" if pa_solved else "✗"
        vamp_status = f"✓ {vamp_time:.2f}s" if vamp_solved else "✗"

        if pa_solved and vamp_solved:
            if pa_time < vamp_time:
                winner = "ProofAtlas"
            elif vamp_time < pa_time:
                winner = "Vampire"
            else:
                winner = "Tie"
            results['both'] += 1
        elif pa_solved:
            winner = "ProofAtlas"
        elif vamp_solved:
            winner = "Vampire"
        else:
            winner = "-"
            results['neither'] += 1

        if pa_solved:
            results['proofatlas']['solved'] += 1
            results['proofatlas']['time'] += pa_time
        if vamp_solved:
            results['vampire']['solved'] += 1
            results['vampire']['time'] += vamp_time

        # Shorten problem name for display
        short_name = problem.split('/')[-1]
        print(f"{short_name:<30} {pa_status:<18} {vamp_status:<15} {winner:<10}")

    # Category summary
    total = len(problems)
    print(f"\n{'-'*70}")
    print(f"Category Summary:")
    print(f"  ProofAtlas (LargestNeg): {results['proofatlas']['solved']}/{total} solved " +
          f"({results['proofatlas']['solved']/total*100:.0f}%)")
    if results['proofatlas']['solved'] > 0:
        print(f"    Avg time: {results['proofatlas']['time']/results['proofatlas']['solved']:.2f}s")

    print(f"  Vampire:                 {results['vampire']['solved']}/{total} solved " +
          f"({results['vampire']['solved']/total*100:.0f}%)")
    if results['vampire']['solved'] > 0:
        print(f"    Avg time: {results['vampire']['time']/results['vampire']['solved']:.2f}s")

    print(f"  Both solved: {results['both']}")
    print(f"  Neither solved: {results['neither']}")

    return results

def main():
    """Main comparison function."""
    import argparse

    parser = argparse.ArgumentParser(description='Compare ProofAtlas (LargestNeg) with Vampire')
    parser.add_argument('--problems', type=int, default=20,
                       help='Number of problems per category (default: 20)')
    parser.add_argument('--timeout', type=int, default=10,
                       help='Timeout per problem in seconds (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--vampire-selection', type=str, default='0',
                       help='Vampire clause selection strategy (default: 0)')

    args = parser.parse_args()
    random.seed(args.seed)

    print("="*70)
    print(f"PROOFATLAS (SelectLargestNegative) vs VAMPIRE COMPARISON")
    print(f"Vampire options: -s {args.vampire_selection} --avatar off")
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
        'vampire': {'solved': 0, 'time': 0}
    }

    # Test each category (CNF only)
    for category, problems in problem_lists.items():
        sample = random.sample(problems, min(args.problems, len(problems)))
        results = compare_on_problems(category, sample, args.timeout, args.vampire_selection)

        overall['proofatlas']['solved'] += results['proofatlas']['solved']
        overall['proofatlas']['time'] += results['proofatlas']['time']
        overall['vampire']['solved'] += results['vampire']['solved']
        overall['vampire']['time'] += results['vampire']['time']

    # Overall summary
    total_problems = sum(min(args.problems, len(probs)) for probs in problem_lists.values())
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")

    pa_solved = overall['proofatlas']['solved']
    vamp_solved = overall['vampire']['solved']

    print(f"ProofAtlas (LargestNeg): {pa_solved}/{total_problems} solved ({pa_solved/total_problems*100:.1f}%)")
    if pa_solved > 0:
        print(f"  Average solve time: {overall['proofatlas']['time']/pa_solved:.2f}s")

    print(f"Vampire:                 {vamp_solved}/{total_problems} solved ({vamp_solved/total_problems*100:.1f}%)")
    if vamp_solved > 0:
        print(f"  Average solve time: {overall['vampire']['time']/vamp_solved:.2f}s")

    print(f"\nWinner: ", end="")
    if pa_solved > vamp_solved:
        print(f"ProofAtlas (+{pa_solved - vamp_solved} problems)")
    elif vamp_solved > pa_solved:
        print(f"Vampire (+{vamp_solved - pa_solved} problems)")
    else:
        print("Tie!")

if __name__ == "__main__":
    main()
