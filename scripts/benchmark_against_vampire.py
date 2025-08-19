#!/usr/bin/env python3
"""
Benchmark ProofAtlas against Vampire on categorized TPTP problems.
"""

import os
import subprocess
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import signal
import resource

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Timeout!")

def run_prover(prover_cmd: List[str], timeout: int) -> Tuple[bool, float, str]:
    """
    Run a prover with timeout.
    Returns: (success, time_taken, output)
    """
    start_time = time.time()
    
    try:
        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        # Run the prover
        result = subprocess.run(
            prover_cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        # Cancel the alarm
        signal.alarm(0)
        
        time_taken = time.time() - start_time
        
        # Check for success
        output = result.stdout + result.stderr
        success = False
        
        # Check for proof found
        if "Proof found" in output or "SZS status Theorem" in output or "SZS status Unsatisfiable" in output:
            success = True
        elif "SZS status Satisfiable" in output or "SZS status CounterSatisfiable" in output:
            success = False  # Problem is satisfiable
        
        return success, time_taken, output
        
    except TimeoutException:
        signal.alarm(0)
        return False, timeout, "TIMEOUT"
    except Exception as e:
        signal.alarm(0)
        return False, time.time() - start_time, f"ERROR: {str(e)}"

def run_proofatlas(problem_path: Path, timeout: int, use_superposition: bool = True) -> Tuple[bool, float, str]:
    """Run ProofAtlas on a problem."""
    prove_binary = Path(__file__).parent.parent / "rust/target/release/prove"
    
    if not prove_binary.exists():
        # Try debug build
        prove_binary = Path(__file__).parent.parent / "rust/target/debug/prove"
    
    if not prove_binary.exists():
        return False, 0, "ProofAtlas binary not found. Run 'cargo build --release' in rust/ directory."
    
    cmd = [str(prove_binary), str(problem_path), f"--timeout={timeout}"]
    if not use_superposition:
        cmd.append("--no-superposition")
    
    return run_prover(cmd, timeout)

def run_vampire(problem_path: Path, timeout: int, mode: str = "casc") -> Tuple[bool, float, str]:
    """Run Vampire on a problem."""
    # Check if vampire is in PATH
    vampire_cmd = "vampire"
    
    try:
        subprocess.run([vampire_cmd, "--version"], capture_output=True, check=True)
    except:
        return False, 0, "Vampire not found in PATH. Please install Vampire theorem prover."
    
    cmd = [vampire_cmd, f"--mode={mode}", f"--time_limit={timeout}", str(problem_path)]
    
    return run_prover(cmd, timeout)

def benchmark_problem_list(
    problem_list_file: Path,
    tptp_base: Path,
    timeout: int,
    max_problems: Optional[int] = None,
    use_superposition: bool = True
) -> Dict:
    """Benchmark a list of problems."""
    results = {
        "category": problem_list_file.stem.replace("_problems", ""),
        "timeout": timeout,
        "problems": []
    }
    
    # Read problem list
    with open(problem_list_file, 'r') as f:
        lines = f.readlines()
    
    problems = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    
    if max_problems:
        problems = problems[:max_problems]
    
    print(f"\nBenchmarking {len(problems)} problems from {problem_list_file.name}")
    print("=" * 60)
    
    proofatlas_solved = 0
    vampire_solved = 0
    both_solved = 0
    
    for i, problem_rel_path in enumerate(problems):
        problem_path = tptp_base / problem_rel_path
        
        if not problem_path.exists():
            print(f"Problem not found: {problem_path}")
            continue
        
        print(f"\n[{i+1}/{len(problems)}] {problem_rel_path}")
        
        # Run ProofAtlas
        print("  Running ProofAtlas...", end='', flush=True)
        pa_success, pa_time, pa_output = run_proofatlas(problem_path, timeout, use_superposition)
        print(f" {'SOLVED' if pa_success else 'FAILED'} ({pa_time:.2f}s)")
        
        # Run Vampire
        print("  Running Vampire...", end='', flush=True)
        v_success, v_time, v_output = run_vampire(problem_path, timeout)
        print(f" {'SOLVED' if v_success else 'FAILED'} ({v_time:.2f}s)")
        
        # Update counters
        if pa_success:
            proofatlas_solved += 1
        if v_success:
            vampire_solved += 1
        if pa_success and v_success:
            both_solved += 1
        
        # Store result
        results["problems"].append({
            "problem": problem_rel_path,
            "proofatlas": {
                "solved": pa_success,
                "time": pa_time
            },
            "vampire": {
                "solved": v_success,
                "time": v_time
            }
        })
    
    # Summary statistics
    total = len(results["problems"])
    results["summary"] = {
        "total_problems": total,
        "proofatlas_solved": proofatlas_solved,
        "vampire_solved": vampire_solved,
        "both_solved": both_solved,
        "proofatlas_unique": proofatlas_solved - both_solved,
        "vampire_unique": vampire_solved - both_solved,
        "proofatlas_success_rate": proofatlas_solved / total if total > 0 else 0,
        "vampire_success_rate": vampire_solved / total if total > 0 else 0
    }
    
    print(f"\n{'=' * 60}")
    print(f"Summary for {results['category']}:")
    print(f"  Total problems: {total}")
    print(f"  ProofAtlas solved: {proofatlas_solved} ({proofatlas_solved/total*100:.1f}%)")
    print(f"  Vampire solved: {vampire_solved} ({vampire_solved/total*100:.1f}%)")
    print(f"  Both solved: {both_solved}")
    print(f"  ProofAtlas unique: {proofatlas_solved - both_solved}")
    print(f"  Vampire unique: {vampire_solved - both_solved}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark ProofAtlas against Vampire")
    parser.add_argument("--timeout", type=int, default=10, help="Timeout per problem in seconds")
    parser.add_argument("--max-problems", type=int, help="Maximum number of problems per category")
    parser.add_argument("--categories", nargs="+", 
                        choices=["unit_equalities", "cnf_without_equality", "cnf_with_equality", 
                                "fof_without_equality", "fof_with_equality"],
                        help="Categories to benchmark (default: all)")
    parser.add_argument("--no-superposition", action="store_true", 
                        help="Disable superposition for ProofAtlas")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                        help="Output file for results")
    
    args = parser.parse_args()
    
    # Paths
    base_path = Path(__file__).parent.parent
    tptp_base = base_path / ".data/problems/tptp/TPTP-v9.0.0/Problems"
    lists_dir = base_path / ".data/benchmark_lists"
    
    if not lists_dir.exists():
        print("Problem lists not found. Running categorization script first...")
        subprocess.run([str(base_path / "scripts/categorize_tptp_problems.py")], check=True)
    
    # Determine which categories to run
    if args.categories:
        categories = args.categories
    else:
        categories = ["unit_equalities", "cnf_without_equality", "cnf_with_equality", 
                     "fof_without_equality", "fof_with_equality"]
    
    all_results = {}
    
    # Run benchmarks for each category
    for category in categories:
        list_file = lists_dir / f"{category}_problems.txt"
        if not list_file.exists():
            print(f"Warning: {list_file} not found, skipping category")
            continue
        
        results = benchmark_problem_list(
            list_file, 
            tptp_base, 
            args.timeout,
            args.max_problems,
            not args.no_superposition
        )
        
        all_results[category] = results
    
    # Save results
    output_path = base_path / ".data" / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Print overall summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    
    total_pa = 0
    total_v = 0
    total_problems = 0
    
    for category, results in all_results.items():
        summary = results["summary"]
        print(f"\n{category.replace('_', ' ').title()}:")
        print(f"  ProofAtlas: {summary['proofatlas_solved']}/{summary['total_problems']} ({summary['proofatlas_success_rate']*100:.1f}%)")
        print(f"  Vampire: {summary['vampire_solved']}/{summary['total_problems']} ({summary['vampire_success_rate']*100:.1f}%)")
        
        total_pa += summary['proofatlas_solved']
        total_v += summary['vampire_solved']
        total_problems += summary['total_problems']
    
    if total_problems > 0:
        print(f"\nOverall:")
        print(f"  ProofAtlas: {total_pa}/{total_problems} ({total_pa/total_problems*100:.1f}%)")
        print(f"  Vampire: {total_v}/{total_problems} ({total_v/total_problems*100:.1f}%)")

if __name__ == "__main__":
    main()