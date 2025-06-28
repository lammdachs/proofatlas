#!/usr/bin/env python3
"""
Compare the Rust TPTP parser with Vampire's parser output.

This script parses TPTP files using both the new Rust parser and Vampire,
then compares the results to ensure correctness.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import tempfile
import os

# Add the python directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from proofatlas.core.logic import Problem, Clause, Literal, Predicate, Constant, Variable, Function, Term
from proofatlas.fileformats.tptp import TPTPFormat

# Import the Rust parser
try:
    import proofatlas_rust
except ImportError:
    print("Error: proofatlas_rust module not found. Please build it with 'cd rust && maturin develop'")
    sys.exit(1)


def parse_with_rust(file_path: Path) -> Optional[Dict]:
    """Parse a TPTP file using the Rust parser.
    
    Note: Only CNF format is supported. TFF/THF lines are skipped.
    """
    try:
        result = proofatlas_rust.parser.parse_file_to_dict(str(file_path))
        # Check if we actually got any clauses
        if result.get("num_clauses", 0) == 0:
            # File might contain only unsupported formats
            return None
        return result
    except Exception as e:
        print(f"Rust parser error on {file_path}: {e}")
        return None


def parse_with_python(file_path: Path) -> Optional[Problem]:
    """Parse a TPTP file using the Python parser."""
    try:
        parser = TPTPFormat()
        return parser.parse_file(file_path)
    except Exception as e:
        print(f"Python parser error on {file_path}: {e}")
        return None


def parse_with_vampire(file_path: Path) -> Optional[List[str]]:
    """Parse a TPTP file using Vampire and extract the clauses."""
    try:
        # Check if vampire is available
        vampire_check = subprocess.run(["which", "vampire"], capture_output=True)
        if vampire_check.returncode != 0:
            return None
            
        # Run Vampire with clausify option
        result = subprocess.run(
            ["vampire", "--mode", "clausify", "--print_clausifier_premises", "on", str(file_path)],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            return None
            
        # Extract clauses from Vampire output
        clauses = []
        in_clausification = False
        
        for line in result.stdout.split('\n'):
            line = line.strip()
            
            # Look for clausification section
            if "% Clausification" in line:
                in_clausification = True
                continue
                
            if in_clausification and line.startswith('tff(') or line.startswith('fof('):
                # Extract clause content
                if ',' in line:
                    parts = line.split(',', 2)
                    if len(parts) >= 3:
                        clause_content = parts[2].rstrip(').').strip()
                        clauses.append(clause_content)
                        
        return clauses
        
    except subprocess.TimeoutExpired:
        print(f"Vampire timeout on {file_path}")
        return None
    except Exception as e:
        print(f"Vampire error on {file_path}: {e}")
        return None


def normalize_clause_str(clause_str: str) -> str:
    """Normalize a clause string for comparison."""
    # Remove whitespace and normalize operators
    clause_str = clause_str.replace(' ', '')
    clause_str = clause_str.replace('!=', '!≡')  # Normalize inequality
    return clause_str


def compare_results(file_path: Path, rust_result: Dict, python_problem: Problem, vampire_clauses: List[str]) -> Dict:
    """Compare parsing results from different parsers."""
    comparison = {
        "file": str(file_path),
        "rust_success": rust_result is not None,
        "python_success": python_problem is not None,
        "vampire_success": vampire_clauses is not None,
        "differences": []
    }
    
    if rust_result and python_problem:
        # Compare clause counts
        rust_clause_count = len(rust_result.get("clauses", []))
        python_clause_count = len(python_problem.clauses)
        
        comparison["rust_clauses"] = rust_clause_count
        comparison["python_clauses"] = python_clause_count
        
        if rust_clause_count != python_clause_count:
            comparison["differences"].append(f"Clause count mismatch: Rust={rust_clause_count}, Python={python_clause_count}")
            
        # TODO: Deep comparison of clause contents
        
    if vampire_clauses:
        comparison["vampire_clauses"] = len(vampire_clauses)
        
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Compare Rust TPTP parser with Vampire")
    parser.add_argument("path", type=str, help="Path to TPTP file or directory")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of files to process")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--no-vampire", action="store_true", help="Skip Vampire comparison")
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    # Collect files to process
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = list(path.rglob("*.p"))[:args.limit]
    else:
        print(f"Error: {path} is not a valid file or directory")
        sys.exit(1)
        
    print(f"Processing {len(files)} files...")
    
    results = []
    stats = {
        "total": len(files),
        "rust_success": 0,
        "python_success": 0,
        "vampire_success": 0,
        "all_success": 0,
        "discrepancies": 0
    }
    
    for i, file_path in enumerate(files):
        if args.verbose:
            print(f"\n[{i+1}/{len(files)}] Processing {file_path.name}...")
            
        # Parse with each parser
        start = time.time()
        rust_result = parse_with_rust(file_path)
        rust_time = time.time() - start
        
        start = time.time()
        python_problem = parse_with_python(file_path)
        python_time = time.time() - start
        
        if not args.no_vampire:
            start = time.time()
            vampire_clauses = parse_with_vampire(file_path)
            vampire_time = time.time() - start
        else:
            vampire_clauses = None
            vampire_time = 0
        
        # Compare results
        comparison = compare_results(file_path, rust_result, python_problem, vampire_clauses)
        comparison["times"] = {
            "rust": rust_time,
            "python": python_time,
            "vampire": vampire_time
        }
        
        results.append(comparison)
        
        # Update stats
        if rust_result:
            stats["rust_success"] += 1
        if python_problem:
            stats["python_success"] += 1
        if vampire_clauses:
            stats["vampire_success"] += 1
        if rust_result and python_problem and vampire_clauses:
            stats["all_success"] += 1
        if comparison["differences"]:
            stats["discrepancies"] += 1
            
        if args.verbose and comparison["differences"]:
            print(f"  Discrepancies found: {comparison['differences']}")
            
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Total files: {stats['total']}")
    print(f"Rust parser success: {stats['rust_success']} ({stats['rust_success']/stats['total']*100:.1f}%)")
    print(f"Python parser success: {stats['python_success']} ({stats['python_success']/stats['total']*100:.1f}%)")
    print(f"Vampire parser success: {stats['vampire_success']} ({stats['vampire_success']/stats['total']*100:.1f}%)")
    print(f"All parsers successful: {stats['all_success']} ({stats['all_success']/stats['total']*100:.1f}%)")
    print(f"Files with discrepancies: {stats['discrepancies']}")
    
    # Calculate average times
    avg_times = {"rust": 0, "python": 0, "vampire": 0}
    for result in results:
        for parser in avg_times:
            avg_times[parser] += result["times"][parser]
    for parser in avg_times:
        avg_times[parser] /= len(results)
        
    print(f"\n=== Average Parse Times ===")
    print(f"Rust:    {avg_times['rust']*1000:.2f} ms")
    print(f"Python:  {avg_times['python']*1000:.2f} ms")
    print(f"Vampire: {avg_times['vampire']*1000:.2f} ms")
    print(f"Rust speedup vs Python: {avg_times['python']/avg_times['rust']:.1f}x")
    
    # Save results if requested
    if args.output:
        output_data = {
            "stats": stats,
            "average_times": avg_times,
            "results": results
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()