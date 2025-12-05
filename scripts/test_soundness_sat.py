#!/usr/bin/env python3
"""
Soundness test: Verify ProofAtlas never proves Satisfiable problems.
Runs all Satisfiable CNF problems from benchmark lists with 1s timeout.
"""

import os
import subprocess
import time
from pathlib import Path
from typing import List, Tuple
import re

def get_tptp_base() -> Path:
    """Get the TPTP base directory."""
    return Path(__file__).parent.parent / ".tptp/TPTP-v9.0.0/Problems"

def get_problem_status(problem_path: Path) -> str:
    """Get the status of a TPTP problem."""
    try:
        with open(problem_path, 'r') as f:
            for line in f.readlines()[:50]:
                if 'Status' in line and ':' in line:
                    match = re.search(r'Status\s*:\s*(\w+)', line)
                    if match:
                        return match.group(1).strip()
        return "Unknown"
    except:
        return "Unknown"

def load_satisfiable_problems() -> List[str]:
    """Load all Satisfiable CNF problems from benchmark lists."""
    lists_dir = Path(__file__).parent.parent / ".data/benchmark_lists"
    tptp_base = get_tptp_base()

    categories = {
        "unit_equalities": "unit_equalities_problems.txt",
        "cnf_without_equality": "cnf_without_equality_problems.txt",
        "cnf_with_equality": "cnf_with_equality_problems.txt",
    }

    satisfiable_problems = []

    print("Loading Satisfiable problems from benchmark lists...")
    for category, filename in categories.items():
        filepath = lists_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                lines = f.readlines()
                all_problems = [line.strip() for line in lines if line.strip() and not line.startswith('#')]

                # Filter for Satisfiable problems only
                for problem in all_problems:
                    problem_path = tptp_base / problem
                    if problem_path.exists():
                        status = get_problem_status(problem_path)
                        if status == 'Satisfiable':
                            satisfiable_problems.append(problem)

                sat_count = len([p for p in all_problems
                                if get_problem_status(tptp_base / p) == 'Satisfiable'])
                print(f"  {category}: {sat_count} Satisfiable problems")

    return satisfiable_problems

def test_problem(problem_path: Path, timeout: int = 1) -> Tuple[bool, str]:
    """
    Test a single problem with ProofAtlas.
    Returns (claimed_proof, result_description).
    """
    rust_dir = Path(__file__).parent.parent / "rust"
    tptp_root = Path(__file__).parent.parent / ".tptp/TPTP-v9.0.0"

    cmd = [
        "cargo", "run", "--release", "--bin", "prove", "--",
        str(problem_path),
        "--timeout", str(timeout),
        "--literal-selection", "22",
        "--include", str(tptp_root)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 2,
            cwd=rust_dir
        )

        output = result.stdout + result.stderr

        if "THEOREM PROVED" in output or "Proof found!" in output:
            return (True, "CLAIMED PROOF (UNSOUND!)")
        elif "TIMEOUT" in output:
            return (False, "Timeout")
        elif "SATURATED" in output:
            return (False, "Saturated")
        else:
            return (False, "Other")

    except subprocess.TimeoutExpired:
        return (False, "Timeout")
    except Exception as e:
        return (False, f"Error: {e}")

def main():
    """Run soundness test on all Satisfiable problems."""
    print("="*70)
    print("SOUNDNESS TEST: Satisfiable Problems")
    print("Testing that ProofAtlas never proves Satisfiable problems")
    print("="*70)
    print()

    # Load Satisfiable problems
    satisfiable_problems = load_satisfiable_problems()

    if not satisfiable_problems:
        print("No Satisfiable problems found in benchmark lists.")
        return

    print(f"\nTesting {len(satisfiable_problems)} Satisfiable problems (1s timeout each)...")
    print()

    tptp_base = get_tptp_base()
    unsound_proofs = []
    results = {"Timeout": 0, "Saturated": 0, "Other": 0}

    for i, problem in enumerate(satisfiable_problems, 1):
        problem_path = tptp_base / problem

        if not problem_path.exists():
            continue

        print(f"[{i}/{len(satisfiable_problems)}] Testing {problem}...", end=" ", flush=True)

        claimed_proof, result = test_problem(problem_path, timeout=1)

        if claimed_proof:
            print(f"❌ {result}")
            unsound_proofs.append((problem, result))
        else:
            print(f"✓ {result}")
            results[result] = results.get(result, 0) + 1

    # Summary
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total Satisfiable problems tested: {len(satisfiable_problems)}")
    print(f"Problems that timed out: {results.get('Timeout', 0)}")
    print(f"Problems that saturated: {results.get('Saturated', 0)}")
    print(f"Other results: {results.get('Other', 0)}")
    print()

    if unsound_proofs:
        print(f"❌ SOUNDNESS VIOLATION: {len(unsound_proofs)} Satisfiable problems were 'proved'!")
        print()
        print("Unsound proofs:")
        for problem, result in unsound_proofs:
            print(f"  - {problem}: {result}")
        print()
        print("⚠️  ProofAtlas has a CRITICAL SOUNDNESS BUG!")
        return 1
    else:
        print("✅ SOUNDNESS TEST PASSED")
        print("   ProofAtlas correctly did not prove any Satisfiable problems.")
        return 0

if __name__ == "__main__":
    exit(main())
