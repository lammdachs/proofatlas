#!/usr/bin/env python3
"""
Generate benchmark problem data for the web interface.
Uses the unified problem_metadata.json for filtering.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional


def load_problem_metadata() -> Dict:
    """Load the unified problem metadata JSON."""
    metadata_path = Path(__file__).parent.parent / ".data/problem_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Problem metadata not found: {metadata_path}\n"
            "Run: python scripts/extract_problem_metadata.py"
        )
    with open(metadata_path, 'r') as f:
        return json.load(f)


def filter_problems(
    problems: List[Dict],
    status: Optional[str] = None,
    format: Optional[str] = None,
    has_equality: Optional[bool] = None,
    is_unit_only: Optional[bool] = None,
    max_rating: Optional[float] = None,
    min_clauses: Optional[int] = None,
    max_clauses: Optional[int] = None,
    domains: Optional[List[str]] = None,
) -> List[Dict]:
    """Filter problems based on criteria."""
    result = []
    for p in problems:
        if status and p['status'] != status:
            continue
        if format and p['format'] != format:
            continue
        if has_equality is not None and p['has_equality'] != has_equality:
            continue
        if is_unit_only is not None and p['is_unit_only'] != is_unit_only:
            continue
        if max_rating is not None and p['rating'] > max_rating:
            continue
        if min_clauses is not None and p['num_clauses'] < min_clauses:
            continue
        if max_clauses is not None and p['num_clauses'] > max_clauses:
            continue
        if domains and p['domain'] not in domains:
            continue
        result.append(p)
    return result


def sample_problems(problems: List[Dict], count: int, seed: int = 42) -> List[Dict]:
    """Randomly sample problems."""
    random.seed(seed)
    return random.sample(problems, min(count, len(problems)))


def format_for_web(problems: List[Dict], category: str) -> List[Dict]:
    """Format problems for web interface."""
    result = []
    for p in problems:
        domain = p['domain']
        filename = Path(p['path']).name
        problem_name = filename.replace('.p', '')
        tptp_url = f"https://tptp.org/cgi-bin/SeeTPTP?Category=Problems&Domain={domain}&File={filename}"

        result.append({
            'name': problem_name,
            'file': p['path'],
            'domain': domain,
            'category': category,
            'status': p['status'],
            'rating': p['rating'],
            'has_equality': p['has_equality'],
            'num_clauses': p['num_clauses'],
            'tptp_url': tptp_url,
        })
    return result


def main():
    """Generate benchmark data."""
    print("Loading problem metadata...")
    metadata = load_problem_metadata()
    all_problems = metadata['problems']
    print(f"Loaded {len(all_problems)} problems")

    # Define benchmark categories
    categories = [
        {
            'name': 'Unit Equalities (CNF)',
            'filters': {
                'status': 'unsatisfiable',
                'format': 'cnf',
                'is_unit_only': True,
                'has_equality': True,
            },
            'count': 50,
        },
        {
            'name': 'CNF Without Equality',
            'filters': {
                'status': 'unsatisfiable',
                'format': 'cnf',
                'has_equality': False,
            },
            'count': 50,
        },
        {
            'name': 'CNF With Equality',
            'filters': {
                'status': 'unsatisfiable',
                'format': 'cnf',
                'has_equality': True,
                'is_unit_only': False,
            },
            'count': 50,
        },
        {
            'name': 'FOF Without Equality',
            'filters': {
                'status': 'unsatisfiable',
                'format': 'fof',
                'has_equality': False,
            },
            'count': 50,
        },
        {
            'name': 'FOF With Equality',
            'filters': {
                'status': 'unsatisfiable',
                'format': 'fof',
                'has_equality': True,
            },
            'count': 50,
        },
    ]

    benchmark_problems = []

    for cat in categories:
        print(f"\nProcessing: {cat['name']}")
        filtered = filter_problems(all_problems, **cat['filters'])
        print(f"  Found {len(filtered)} matching problems")

        sampled = sample_problems(filtered, cat['count'])
        print(f"  Sampled {len(sampled)} problems")

        formatted = format_for_web(sampled, cat['name'])
        benchmark_problems.extend(formatted)

    # Output JSON
    output = {
        'problems': benchmark_problems,
        'summary': {
            'total': len(benchmark_problems),
            'categories': [cat['name'] for cat in categories],
            'source': 'problem_metadata.json',
        }
    }

    output_file = Path(__file__).parent.parent / "wasm" / "benchmark_problems.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nGenerated {len(benchmark_problems)} benchmark problems")
    print(f"Output: {output_file}")

    # Print category breakdown
    print("\nCategory breakdown:")
    for cat in categories:
        count = sum(1 for p in benchmark_problems if p['category'] == cat['name'])
        print(f"  {cat['name']}: {count}")


if __name__ == "__main__":
    main()
