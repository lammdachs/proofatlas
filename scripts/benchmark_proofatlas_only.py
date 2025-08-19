#!/usr/bin/env python3
"""
Benchmark ProofAtlas on TPTP problems.
Runs 600 random problems from each category with 10s timeout.
Generates timing analysis and cactus plots.
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

def run_proofatlas(problem_path: Path, timeout: int = 10) -> Tuple[bool, float, str]:
    """Run ProofAtlas on a problem with timeout.
    
    Returns: (solved, time_taken, output)
    """
    start_time = time.time()
    
    # Change to rust directory for cargo
    original_cwd = os.getcwd()
    rust_dir = Path(__file__).parent.parent / "rust"
    os.chdir(rust_dir)
    
    try:
        tptp_root = Path(__file__).parent.parent / ".data/problems/tptp/TPTP-v9.0.0"
        cmd = ["cargo", "run", "--release", "--bin", "prove", "--", 
               str(problem_path), "--timeout", str(timeout), "--include", str(tptp_root)]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 1)
        
        time_taken = time.time() - start_time
        
        # Check if solved
        output = result.stdout + result.stderr
        solved = "Proof found!" in output
        
        return solved, time_taken, output
        
    except subprocess.TimeoutExpired:
        return False, timeout, "TIMEOUT"
    except Exception as e:
        return False, timeout, f"ERROR: {str(e)}"
    finally:
        os.chdir(original_cwd)

def benchmark_category(category: str, problems: List[str], timeout: int = 10) -> List[Tuple[str, bool, float]]:
    """Benchmark a category of problems.
    
    Returns: [(problem, solved, time), ...]
    """
    tptp_base = get_tptp_base()
    results = []
    
    print(f"\nBenchmarking {category}...")
    print("=" * 80)
    
    solved_count = 0
    
    for i, problem in enumerate(problems):
        problem_path = tptp_base / problem
        print(f"[{i+1}/{len(problems)}] {problem}", end="", flush=True)
        
        # Run ProofAtlas
        solved, time_taken, _ = run_proofatlas(problem_path, timeout)
        results.append((problem, solved, time_taken))
        
        if solved:
            solved_count += 1
        
        # Print status
        status = f"✓ {time_taken:.2f}s" if solved else "✗"
        print(f" - {status} (Total solved: {solved_count}/{i+1})")
    
    return results

def save_results(all_results: Dict[str, List[Tuple[str, bool, float]]], output_dir: Path):
    """Save benchmark results to JSON."""
    # Convert to serializable format
    serializable = {}
    for category, results in all_results.items():
        serializable[category] = [
            {"problem": p, "solved": s, "time": t} 
            for p, s, t in results
        ]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"proofatlas_benchmark_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    return output_file

def generate_cactus_plot(all_results: Dict[str, List[Tuple[str, bool, float]]], 
                        timeout: int, output_dir: Path):
    """Generate a combined cactus plot for all categories."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    categories = list(all_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    
    # Also prepare data for combined plot
    all_solve_times = []
    
    for idx, (category, results) in enumerate(all_results.items()):
        ax = axes[idx]
        
        # Extract solved problems and their times
        solve_times = sorted([t for _, solved, t in results if solved])
        all_solve_times.extend(solve_times)
        
        if solve_times:
            # Add a point at (0,0) for better visualization
            x_values = [0] + list(range(1, len(solve_times) + 1))
            y_values = [0] + solve_times
            
            ax.plot(x_values, y_values, linewidth=2, color=colors[idx])
        
        # Statistics
        solved_count = len(solve_times)
        total_count = len(results)
        
        ax.set_xlabel('Problems Solved')
        ax.set_ylabel('Time (s)')
        ax.set_title(f'{category.replace("_", " ").title()}\n({solved_count}/{total_count} solved)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, timeout * 1.1)
    
    # Hide the 6th subplot if we only have 5 categories
    if len(categories) < 6:
        axes[-1].set_visible(False)
    
    plt.suptitle('ProofAtlas Performance - Cactus Plots by Category', fontsize=16)
    plt.tight_layout()
    
    output_file = output_dir / "cactus_plot_by_category.png"
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    # Generate overall cactus plot
    plt.figure(figsize=(10, 6))
    all_solve_times_sorted = sorted(all_solve_times)
    
    if all_solve_times_sorted:
        x_values = [0] + list(range(1, len(all_solve_times_sorted) + 1))
        y_values = [0] + all_solve_times_sorted
        plt.plot(x_values, y_values, linewidth=2, color='blue')
    
    total_problems = sum(len(results) for results in all_results.values())
    total_solved = len(all_solve_times_sorted)
    
    plt.xlabel('Number of Problems Solved', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title(f'ProofAtlas Overall Performance - Cactus Plot\n({total_solved}/{total_problems} solved)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, timeout * 1.1)
    
    plt.tight_layout()
    overall_output = output_dir / "cactus_plot_overall.png"
    plt.savefig(overall_output, dpi=300)
    plt.close()
    
    return output_file, overall_output

def generate_time_distribution(all_results: Dict[str, List[Tuple[str, bool, float]]], 
                              timeout: int, output_dir: Path):
    """Generate time distribution histogram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_solve_times = []
    for results in all_results.values():
        solve_times = [t for _, solved, t in results if solved]
        all_solve_times.extend(solve_times)
    
    if all_solve_times:
        # Create bins
        bins = np.logspace(-3, np.log10(timeout), 50)
        
        ax.hist(all_solve_times, bins=bins, alpha=0.7, color='blue', edgecolor='black')
        ax.set_xscale('log')
        ax.set_xlabel('Solve Time (seconds)', fontsize=12)
        ax.set_ylabel('Number of Problems', fontsize=12)
        ax.set_title('Distribution of Solve Times', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        median_time = np.median(all_solve_times)
        mean_time = np.mean(all_solve_times)
        ax.axvline(median_time, color='red', linestyle='--', label=f'Median: {median_time:.2f}s')
        ax.axvline(mean_time, color='green', linestyle='--', label=f'Mean: {mean_time:.2f}s')
        ax.legend()
    
    plt.tight_layout()
    output_file = output_dir / "time_distribution.png"
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    return output_file

def print_summary(all_results: Dict[str, List[Tuple[str, bool, float]]]):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("PROOFATLAS BENCHMARK SUMMARY")
    print("=" * 80)
    
    total_solved = 0
    total_problems = 0
    
    for category, results in all_results.items():
        solved = sum(1 for _, s, _ in results if s)
        total = len(results)
        total_solved += solved
        total_problems += total
        
        solve_times = [t for _, s, t in results if s]
        avg_time = np.mean(solve_times) if solve_times else 0
        median_time = np.median(solve_times) if solve_times else 0
        
        print(f"\n{category.replace('_', ' ').title()}:")
        print(f"  Solved: {solved}/{total} ({solved/total*100:.1f}%)")
        print(f"  Average solve time: {avg_time:.2f}s")
        print(f"  Median solve time: {median_time:.2f}s")
        
        # Time distribution
        if solve_times:
            time_ranges = [(0, 0.1), (0.1, 1), (1, 5), (5, 10)]
            print("  Time distribution:")
            for low, high in time_ranges:
                count = sum(1 for t in solve_times if low < t <= high)
                if count > 0:
                    print(f"    {low}-{high}s: {count} problems")
    
    print(f"\nOverall Performance:")
    print(f"  Total solved: {total_solved}/{total_problems} ({total_solved/total_problems*100:.1f}%)")

def main():
    """Main benchmarking function."""
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / ".data/benchmark_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("ProofAtlas Benchmark Tool")
    print("=" * 80)
    
    print("\nLoading problem lists...")
    problem_lists = load_problem_lists()
    
    print("\nSampling problems...")
    sampled_problems = sample_problems(problem_lists, sample_size=600)
    
    # Save sampled problems for reproducibility
    with open(output_dir / "sampled_problems_proofatlas.json", 'w') as f:
        json.dump(sampled_problems, f, indent=2)
    
    # Run benchmarks
    all_results = {}
    timeout = 10
    
    start_time = time.time()
    
    for category, problems in sampled_problems.items():
        results = benchmark_category(category, problems, timeout)
        all_results[category] = results
    
    total_time = time.time() - start_time
    print(f"\nTotal benchmark time: {total_time/60:.1f} minutes")
    
    # Save results
    results_file = save_results(all_results, output_dir)
    
    # Generate plots
    print("\nGenerating plots...")
    category_plot, overall_plot = generate_cactus_plot(all_results, timeout, output_dir)
    print(f"Generated cactus plots: {category_plot}, {overall_plot}")
    
    dist_plot = generate_time_distribution(all_results, timeout, output_dir)
    print(f"Generated time distribution: {dist_plot}")
    
    # Print summary
    print_summary(all_results)

if __name__ == "__main__":
    main()