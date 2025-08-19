#!/usr/bin/env python3
"""
Benchmark Vampire vs ProofAtlas on TPTP problems.
Runs 600 random problems from each category with 10s timeout.
Generates cactus plots (cumulative solve time graphs).
"""

import os
import json
import subprocess
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def get_tptp_base() -> Path:
    """Get the TPTP base directory."""
    return Path(__file__).parent.parent / ".data/problems/tptp/TPTP-v9.0.0/Problems"

def load_problem_lists() -> Dict[str, List[str]]:
    """Load categorized problem lists."""
    lists_dir = Path(__file__).parent.parent / ".data/benchmark_lists"
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
        with open(filepath, 'r') as f:
            # Skip header lines
            lines = f.readlines()
            problems = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
            problem_lists[category] = problems
    
    return problem_lists

def sample_problems(problem_lists: Dict[str, List[str]], sample_size: int = 600) -> Dict[str, List[str]]:
    """Sample random problems from each category."""
    sampled = {}
    for category, problems in problem_lists.items():
        # Sample min(sample_size, len(problems)) to handle categories with fewer problems
        n_samples = min(sample_size, len(problems))
        sampled[category] = random.sample(problems, n_samples)
        print(f"Sampled {n_samples} problems from {category}")
    return sampled

def run_prover(prover: str, problem_path: Path, timeout: int = 10) -> Tuple[bool, float, str]:
    """Run a prover on a problem with timeout.
    
    Returns: (solved, time_taken, output)
    """
    start_time = time.time()
    
    try:
        if prover == "vampire":
            # Run Vampire - try different possible locations
            vampire_cmd = None
            for cmd_name in ["vampire", "/usr/local/bin/vampire", "~/vampire/vampire"]:
                try:
                    subprocess.run([cmd_name, "--version"], capture_output=True, timeout=1)
                    vampire_cmd = cmd_name
                    break
                except:
                    continue
            
            if not vampire_cmd:
                return False, timeout, "ERROR: Vampire not found"
            
            cmd = [vampire_cmd, "--mode", "casc", "--time_limit", str(timeout), str(problem_path)]
        else:  # proofatlas
            # Run ProofAtlas
            tptp_root = Path(__file__).parent.parent / ".data/problems/tptp/TPTP-v9.0.0"
            cmd = ["cargo", "run", "--release", "--bin", "prove", "--", 
                   str(problem_path), "--timeout", str(timeout), "--include", str(tptp_root)]
            # Change to rust directory for cargo
            original_cwd = os.getcwd()
            os.chdir(Path(__file__).parent.parent / "rust")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 1)
        
        if prover == "proofatlas":
            os.chdir(original_cwd)
        
        time_taken = time.time() - start_time
        
        # Check if solved
        output = result.stdout + result.stderr
        if prover == "vampire":
            # Vampire outputs "% SZS status Theorem" or "% SZS status Unsatisfiable"
            solved = "% SZS status Theorem" in output or "% SZS status Unsatisfiable" in output
        else:  # proofatlas
            # ProofAtlas outputs "Proof found!"
            solved = "Proof found!" in output
        
        return solved, time_taken, output
        
    except subprocess.TimeoutExpired:
        if prover == "proofatlas":
            os.chdir(original_cwd)
        return False, timeout, "TIMEOUT"
    except Exception as e:
        if prover == "proofatlas" and 'original_cwd' in locals():
            os.chdir(original_cwd)
        return False, timeout, f"ERROR: {str(e)}"

def benchmark_category(category: str, problems: List[str], timeout: int = 10) -> Dict[str, List[Tuple[str, bool, float]]]:
    """Benchmark a category of problems with both provers.
    
    Returns: {prover: [(problem, solved, time), ...]}
    """
    tptp_base = get_tptp_base()
    results = {"vampire": [], "proofatlas": []}
    
    print(f"\nBenchmarking {category}...")
    print("=" * 80)
    
    for i, problem in enumerate(problems):
        problem_path = tptp_base / problem
        print(f"[{i+1}/{len(problems)}] {problem}", end="", flush=True)
        
        # Run Vampire
        v_solved, v_time, _ = run_prover("vampire", problem_path, timeout)
        results["vampire"].append((problem, v_solved, v_time))
        
        # Run ProofAtlas
        p_solved, p_time, _ = run_prover("proofatlas", problem_path, timeout)
        results["proofatlas"].append((problem, p_solved, p_time))
        
        # Print summary
        v_status = f"✓ {v_time:.2f}s" if v_solved else "✗"
        p_status = f"✓ {p_time:.2f}s" if p_solved else "✗"
        print(f" - Vampire: {v_status}, ProofAtlas: {p_status}")
    
    return results

def save_results(all_results: Dict[str, Dict[str, List[Tuple[str, bool, float]]]], output_dir: Path):
    """Save benchmark results to JSON."""
    # Convert to serializable format
    serializable = {}
    for category, results in all_results.items():
        serializable[category] = {}
        for prover, prover_results in results.items():
            serializable[category][prover] = [
                {"problem": p, "solved": s, "time": t} 
                for p, s, t in prover_results
            ]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"benchmark_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    return output_file

def generate_cactus_plot(results: Dict[str, List[Tuple[str, bool, float]]], 
                        category: str, timeout: int, output_dir: Path):
    """Generate a cactus plot (cumulative solve time graph) for a category."""
    plt.figure(figsize=(10, 6))
    
    for prover, prover_results in results.items():
        # Extract solved problems and their times
        solve_times = sorted([t for _, solved, t in prover_results if solved])
        
        if solve_times:
            # Add a point at (0,0) for better visualization
            x_values = [0] + list(range(1, len(solve_times) + 1))
            y_values = [0] + solve_times
            
            plt.plot(x_values, y_values, linewidth=2, label=prover.capitalize())
    
    plt.xlabel('Number of Problems Solved', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title(f'Cactus Plot - {category.replace("_", " ").title()}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, timeout * 1.1)
    
    # Add summary statistics
    for prover, prover_results in results.items():
        solved_count = sum(1 for _, solved, _ in prover_results if solved)
        total_count = len(prover_results)
        plt.text(0.02, 0.98 - (0.05 if prover == "vampire" else 0.1), 
                f'{prover.capitalize()}: {solved_count}/{total_count} solved',
                transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    output_file = output_dir / f"cactus_plot_{category}.png"
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    return output_file

def generate_combined_cactus_plot(all_results: Dict[str, Dict[str, List[Tuple[str, bool, float]]]], 
                                 timeout: int, output_dir: Path):
    """Generate a combined cactus plot for all categories."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    categories = list(all_results.keys())
    
    for idx, category in enumerate(categories):
        ax = axes[idx]
        results = all_results[category]
        
        for prover, prover_results in results.items():
            solve_times = sorted([t for _, solved, t in prover_results if solved])
            
            if solve_times:
                x_values = [0] + list(range(1, len(solve_times) + 1))
                y_values = [0] + solve_times
                ax.plot(x_values, y_values, linewidth=2, label=prover.capitalize())
        
        ax.set_xlabel('Problems Solved')
        ax.set_ylabel('Time (s)')
        ax.set_title(category.replace("_", " ").title())
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, timeout * 1.1)
    
    # Hide the 6th subplot if we only have 5 categories
    if len(categories) < 6:
        axes[-1].set_visible(False)
    
    plt.suptitle('Vampire vs ProofAtlas - Cactus Plots', fontsize=16)
    plt.tight_layout()
    
    output_file = output_dir / "cactus_plot_combined.png"
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    return output_file

def print_summary(all_results: Dict[str, Dict[str, List[Tuple[str, bool, float]]]]):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    total_vampire = 0
    total_proofatlas = 0
    
    for category, results in all_results.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        
        for prover, prover_results in results.items():
            solved = sum(1 for _, s, _ in prover_results if s)
            total = len(prover_results)
            avg_time = np.mean([t for _, s, t in prover_results if s]) if solved > 0 else 0
            
            print(f"  {prover.capitalize()}: {solved}/{total} solved ({solved/total*100:.1f}%), "
                  f"avg time: {avg_time:.2f}s")
            
            if prover == "vampire":
                total_vampire += solved
            else:
                total_proofatlas += solved
    
    print(f"\nOverall:")
    total_problems = sum(len(results["vampire"]) for results in all_results.values())
    print(f"  Vampire: {total_vampire}/{total_problems} solved ({total_vampire/total_problems*100:.1f}%)")
    print(f"  ProofAtlas: {total_proofatlas}/{total_problems} solved ({total_proofatlas/total_problems*100:.1f}%)")

def main():
    """Main benchmarking function."""
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / ".data/benchmark_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading problem lists...")
    problem_lists = load_problem_lists()
    
    print("\nSampling problems...")
    sampled_problems = sample_problems(problem_lists, sample_size=600)
    
    # Save sampled problems for reproducibility
    with open(output_dir / "sampled_problems.json", 'w') as f:
        json.dump(sampled_problems, f, indent=2)
    
    # Run benchmarks
    all_results = {}
    timeout = 10
    
    for category, problems in sampled_problems.items():
        results = benchmark_category(category, problems, timeout)
        all_results[category] = results
        
        # Generate individual cactus plot
        plot_file = generate_cactus_plot(results, category, timeout, output_dir)
        print(f"Generated cactus plot: {plot_file}")
    
    # Save results
    results_file = save_results(all_results, output_dir)
    
    # Generate combined cactus plot
    combined_plot = generate_combined_cactus_plot(all_results, timeout, output_dir)
    print(f"Generated combined cactus plot: {combined_plot}")
    
    # Print summary
    print_summary(all_results)

if __name__ == "__main__":
    main()