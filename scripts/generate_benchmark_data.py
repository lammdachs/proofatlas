#!/usr/bin/env python3
"""
Generate benchmark problem data for the web interface.
Outputs JSON with problem information for interactive display.
"""

import json
import random
from pathlib import Path
from typing import List, Dict

def get_tptp_base() -> Path:
    """Get the TPTP base directory."""
    return Path(__file__).parent.parent / ".data/problems/tptp/TPTP-v9.0.0/Problems"

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

def load_problem_list(category: str, filename: str, count: int = 50, seed: int = 42) -> List[Dict]:
    """Load and sample problems from a category."""
    lists_dir = Path(__file__).parent.parent / ".data/benchmark_lists"
    tptp_base = get_tptp_base()

    filepath = lists_dir / filename
    problems = []

    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return problems

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

    # Sample problems
    random.seed(seed)
    sampled = random.sample(unsatisfiable_problems, min(count, len(unsatisfiable_problems)))

    # Create problem data
    for problem in sampled:
        domain, filename = problem.split('/')
        problem_name = filename.replace('.p', '')

        # Generate TPTP URL
        tptp_url = f"https://tptp.org/cgi-bin/SeeTPTP?Category=Problems&Domain={domain}&File={filename}"

        problems.append({
            'name': problem_name,
            'file': problem,
            'domain': domain,
            'category': category,
            'tptp_url': tptp_url
        })

    return problems

def main():
    """Generate benchmark data."""
    categories = {
        "Unit Equalities": "unit_equalities_problems.txt",
        "CNF Without Equality": "cnf_without_equality_problems.txt",
        "CNF With Equality": "cnf_with_equality_problems.txt",
    }

    all_problems = []

    for category, filename in categories.items():
        print(f"Loading {category}...")
        problems = load_problem_list(category, filename, count=50, seed=42)
        all_problems.extend(problems)
        print(f"  Loaded {len(problems)} problems")

    # Output JSON
    output = {
        'problems': all_problems,
        'summary': {
            'total': len(all_problems),
            'categories': list(categories.keys())
        }
    }

    output_file = Path(__file__).parent.parent / "wasm" / "benchmark_problems.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nGenerated {len(all_problems)} problems")
    print(f"Output: {output_file}")

if __name__ == "__main__":
    main()
