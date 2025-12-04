#!/usr/bin/env python3
"""
Test literal selection strategies on CNF problems from benchmark.
Compares SelectAll vs SelectLargestNegative.
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
    """Load CNF problem lists, filtering for Unsatisfiable problems only."""
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

def run_compare_selection(problem_path: Path, timeout: int = 5) -> Dict[str, Tuple[bool, float]]:
    """Run compare_selection binary on a problem.

    Returns: dict mapping strategy name to (solved, time)
    """
    original_cwd = os.getcwd()
    rust_dir = Path(__file__).parent.parent / "rust"
    os.chdir(rust_dir)

    try:
        tptp_root = Path(__file__).parent.parent / ".tptp/TPTP-v9.0.0"
        cmd = ["cargo", "run", "--release", "--bin", "compare_selection", "--",
               str(problem_path), "--include", str(tptp_root)]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout * 3 + 5)
        output = result.stdout + result.stderr

        # Parse output for each strategy
        results = {}
        lines = output.split('\n')

        current_strategy = None
        for line in lines:
            if "Testing with" in line:
                if "Select All" in line:
                    current_strategy = "SelectAll"
                elif "Select Largest Negative" in line:
                    current_strategy = "SelectLargestNegative"
                elif "Select Max Weight" in line:
                    current_strategy = "SelectMaxWeight"
            elif current_strategy and "✓ THEOREM PROVED" in line:
                time_str = line.split("in ")[1].split("s")[0]
                results[current_strategy] = (True, float(time_str))
                current_strategy = None
            elif current_strategy and ("✗" in line or "TIMEOUT" in line or "SATURATED" in line or "RESOURCE" in line):
                results[current_strategy] = (False, timeout)
                current_strategy = None

        return results

    except subprocess.TimeoutExpired:
        return {
            "SelectAll": (False, timeout),
            "SelectLargestNegative": (False, timeout),
            "SelectMaxWeight": (False, timeout),
        }
    except Exception as e:
        print(f"Error: {e}")
        return {}
    finally:
        os.chdir(original_cwd)

def main():
    """Run comparison on CNF problems."""
    import argparse

    parser = argparse.ArgumentParser(description='Compare literal selection strategies')
    parser.add_argument('--problems', type=int, default=10,
                       help='Number of problems per category (default: 10)')
    parser.add_argument('--timeout', type=int, default=5,
                       help='Timeout per problem in seconds (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()
    random.seed(args.seed)

    print("="*70)
    print(f"LITERAL SELECTION STRATEGY COMPARISON (CNF Problems)")
    print(f"Testing {args.problems} problems per category, {args.timeout}s timeout per strategy")
    print("="*70)

    problem_lists = load_problem_lists()
    if not problem_lists:
        print("Error: No problem lists found.")
        return

    tptp_base = get_tptp_base()

    # Overall statistics
    overall_stats = {
        "SelectAll": {"solved": 0, "time": 0},
        "SelectLargestNegative": {"solved": 0, "time": 0},
        "SelectMaxWeight": {"solved": 0, "time": 0},
    }

    total_problems = 0

    # Test each category
    for category, problems in problem_lists.items():
        print(f"\n{'='*70}")
        print(f"{category.replace('_', ' ').upper()}")
        print(f"{'='*70}")
        print(f"{'Problem':<30} {'SelectAll':<15} {'LargestNeg':<15} {'MaxWeight':<15}")
        print(f"{'-'*70}")

        sample = random.sample(problems, min(args.problems, len(problems)))

        for problem in sample:
            problem_path = tptp_base / problem

            if not problem_path.exists():
                continue

            short_name = problem.split('/')[-1]
            print(f"{short_name:<30}", end="", flush=True)

            # Run comparison
            results = run_compare_selection(problem_path, args.timeout)

            if not results:
                print(" ERROR")
                continue

            total_problems += 1

            # Display results
            for strategy in ["SelectAll", "SelectLargestNegative", "SelectMaxWeight"]:
                if strategy in results:
                    solved, time_taken = results[strategy]
                    if solved:
                        overall_stats[strategy]["solved"] += 1
                        overall_stats[strategy]["time"] += time_taken
                        status = f"✓ {time_taken:.2f}s"
                    else:
                        status = "✗"
                    print(f" {status:<15}", end="")
                else:
                    print(f" {'ERROR':<15}", end="")

            print()  # newline

        # Category summary
        print(f"\n{'-'*70}")
        print(f"Category Summary:")
        for strategy in ["SelectAll", "SelectLargestNegative", "SelectMaxWeight"]:
            solved = overall_stats[strategy]["solved"]
            time_total = overall_stats[strategy]["time"]
            print(f"  {strategy:25s}: {solved:2d} solved", end="")
            if solved > 0:
                print(f" (avg: {time_total/solved:.2f}s)")
            else:
                print()

    # Overall summary
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY (CNF Problems Only)")
    print(f"{'='*70}")

    for strategy in ["SelectAll", "SelectLargestNegative", "SelectMaxWeight"]:
        solved = overall_stats[strategy]["solved"]
        time_total = overall_stats[strategy]["time"]
        print(f"{strategy:25s}: {solved:2d}/{total_problems} solved ({solved/total_problems*100 if total_problems > 0 else 0:.1f}%)", end="")
        if solved > 0:
            print(f" - avg: {time_total/solved:.2f}s")
        else:
            print()

    # Winner
    print(f"\n{'-'*70}")
    best_strategy = max(overall_stats.keys(), key=lambda k: overall_stats[k]["solved"])
    print(f"Winner: {best_strategy} (+{overall_stats[best_strategy]['solved'] - min(s['solved'] for s in overall_stats.values())} problems)")

if __name__ == "__main__":
    main()