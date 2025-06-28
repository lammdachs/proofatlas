#!/usr/bin/env python3
"""
Extract TPTP problems with bounded literals using pre-scanning to avoid parsing large files.

This script:
1. Pre-scans files to estimate literal count
2. Only fully parses files likely to be under the limit
3. Saves successfully parsed problems as compact JSON
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import argparse
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from proofatlas.fileformats.tptp import TPTPFormat
from proofatlas.core.logic import Problem


def quick_literal_estimate(file_path: Path, max_depth: int = 3, visited: Optional[set] = None) -> Tuple[int, bool]:
    """
    Quickly estimate the number of literals in a TPTP file without full parsing.
    
    Returns:
        Tuple of (estimated_literal_count, is_exact)
        is_exact is True if we counted all literals, False if we hit limits
    """
    if visited is None:
        visited = set()
    
    # Avoid circular includes
    file_str = str(file_path.resolve())
    if file_str in visited or max_depth <= 0:
        return 0, False
    
    visited.add(file_str)
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except:
        return 0, False
    
    # Count CNF clauses (simple pattern matching)
    cnf_literals = 0
    cnf_pattern = r'cnf\s*\([^,]+,[^,]+,\s*([^)]+)\)\s*\.'
    for match in re.finditer(cnf_pattern, content, re.DOTALL):
        clause_content = match.group(1)
        # Count | as disjunction separators (approximate literal count)
        # This is an estimate - actual count might be slightly different
        cnf_literals += clause_content.count('|') + 1
    
    # Count FOF formulas (these will need CNF conversion)
    fof_count = len(re.findall(r'fof\s*\(', content))
    
    # Handle includes
    include_literals = 0
    include_pattern = r'include\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)'
    tptp_root = os.environ.get('TPTP_PATH', '')
    
    for match in re.finditer(include_pattern, content):
        include_file = match.group(1)
        
        # Try to resolve the include path
        include_paths = []
        if tptp_root:
            include_paths.append(Path(tptp_root) / include_file)
        include_paths.append(file_path.parent / include_file)
        
        for inc_path in include_paths:
            if inc_path.exists():
                inc_literals, _ = quick_literal_estimate(inc_path, max_depth - 1, visited)
                include_literals += inc_literals
                break
    
    # Estimate total literals
    # For FOF, we make a rough estimate (each FOF might generate multiple clauses)
    estimated_total = cnf_literals + include_literals + (fof_count * 10)  # 10 is a heuristic
    
    # If file has too many FOF formulas, it's likely to be large
    is_exact = (fof_count == 0 and max_depth > 0)
    
    return estimated_total, is_exact


def count_literals(problem: Problem) -> int:
    """Count total number of literals in a problem."""
    return sum(len(clause.literals) for clause in problem.clauses)


def clause_to_dict(clause) -> Dict[str, Any]:
    """Convert a Clause object to a dictionary."""
    return {
        'literals': [
            {
                'polarity': lit.polarity,
                'predicate': {
                    'symbol': lit.predicate.symbol.name,
                    'arity': lit.predicate.symbol.arity
                },
                'terms': [term_to_dict(t) for t in lit.predicate.args]
            }
            for lit in clause.literals
        ]
    }


def term_to_dict(term) -> Dict[str, Any]:
    """Convert a Term object to a dictionary."""
    type_name = type(term).__name__
    
    if type_name == 'Variable':
        return {
            'type': 'variable',
            'name': term.name
        }
    elif type_name == 'Constant':
        return {
            'type': 'constant',
            'symbol': term.name
        }
    else:  # Function term
        return {
            'type': 'function',
            'symbol': term.symbol.name,
            'arity': term.symbol.arity,
            'arguments': [term_to_dict(arg) for arg in term.args]
        }


def problem_to_dict(problem: Problem, source_file: str) -> Dict[str, Any]:
    """Convert a Problem object to a dictionary."""
    return {
        'source_file': source_file,
        'num_clauses': len(problem.clauses),
        'num_literals': count_literals(problem),
        'clauses': [clause_to_dict(c) for c in problem.clauses],
        'conjecture_indices': list(problem.conjecture_indices)
    }


def process_single_file(file_info: Tuple[Path, Path, int, float]) -> Tuple[str, str, Any, int]:
    """
    Process a single TPTP file with pre-scanning.
    
    Args:
        file_info: Tuple of (file_path, tptp_root, max_literals, safety_factor)
    
    Returns:
        Tuple of (relative_path, status, result, actual_literals)
        status can be: 'success', 'skipped', 'filtered', 'error'
    """
    file_path, tptp_root, max_literals, safety_factor = file_info
    from proofatlas.fileformats.tptp_parser.parser import read_file
    relative_path = str(file_path.relative_to(file_path.parent.parent.parent))
    
    # First, do a quick pre-scan
    estimated_literals, is_exact = quick_literal_estimate(file_path)
    
    # Skip if estimate is too high (with safety factor)
    if estimated_literals > max_literals * safety_factor:
        return (relative_path, 'skipped', {
            'reason': 'pre-scan',
            'estimated_literals': estimated_literals,
            'is_exact': is_exact
        }, estimated_literals)
    
    # Try to parse the file
    try:
        problem = read_file(str(file_path), include_path=str(tptp_root) + '/')
        actual_literals = count_literals(problem)
        
        # Check if problem meets the literal bound
        if actual_literals > max_literals:
            return (relative_path, 'filtered', {
                'reason': 'actual_count',
                'estimated_literals': estimated_literals,
                'actual_literals': actual_literals
            }, actual_literals)
        
        problem_dict = problem_to_dict(problem, relative_path)
        return (relative_path, 'success', problem_dict, actual_literals)
        
    except Exception as e:
        return (relative_path, 'error', str(e), -1)


def process_files_parallel(tptp_files: List[Path], tptp_root: Path, max_literals: int, 
                          safety_factor: float, num_workers: int = None) -> Dict[str, List]:
    """Process files in parallel with pre-scanning."""
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    results = {
        'parsed_problems': [],
        'skipped_files': [],
        'filtered_files': [],
        'failed_files': []
    }
    
    # Create list of file info tuples
    file_infos = [(f, tptp_root, max_literals, safety_factor) for f in tptp_files]
    
    # Process files in parallel with progress bar
    with mp.Pool(num_workers) as pool:
        processed = list(tqdm(
            pool.imap(process_single_file, file_infos),
            total=len(tptp_files),
            desc="Processing TPTP files"
        ))
    
    # Categorize results
    for relative_path, status, result, literals in processed:
        if status == 'success':
            results['parsed_problems'].append(result)
        elif status == 'skipped':
            results['skipped_files'].append({
                'file': relative_path,
                **result
            })
        elif status == 'filtered':
            results['filtered_files'].append({
                'file': relative_path,
                **result
            })
        elif status == 'error':
            results['failed_files'].append({
                'file': relative_path,
                'error': result
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Extract TPTP problems with bounded literals using pre-scanning')
    parser.add_argument('--tptp-dir', type=str, 
                       default=os.getenv('TPTP_PATH', './.data/problems/tptp'),
                       help='Path to TPTP directory (default: from TPTP_PATH env var)')
    parser.add_argument('--tptp-root', type=str,
                       help='Path to TPTP root directory for resolving includes (auto-detected if not specified)')
    parser.add_argument('--output-dir', type=str, 
                       default=os.path.join(os.getenv('DATASETS_DIR', './.data/datasets'), 'tptp_bounded_json'),
                       help='Output directory for JSON files (default: $DATASETS_DIR/tptp_bounded_json)')
    parser.add_argument('--max-literals', type=int, default=100,
                       help='Maximum number of literals allowed (default: 100)')
    parser.add_argument('--safety-factor', type=float, default=1.5,
                       help='Safety factor for pre-scan (skip if estimate > max * factor, default: 1.5)')
    parser.add_argument('--max-files', type=int, help='Maximum number of files to process')
    parser.add_argument('--domain', type=str, help='Only process files from specific domain (e.g., PUZ, ALG)')
    parser.add_argument('--workers', type=int, help='Number of parallel workers')
    parser.add_argument('--batch-size', type=int, default=1000, 
                       help='Number of problems per JSON file')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only compute statistics without saving problems')
    args = parser.parse_args()
    
    tptp_path = Path(args.tptp_dir)
    output_path = Path(args.output_dir)
    
    # Validate TPTP directory
    if not tptp_path.exists():
        print(f"Error: TPTP directory not found: {tptp_path}")
        sys.exit(1)
    
    problems_dir = tptp_path / "Problems"
    if not problems_dir.exists():
        # Try if we're already in the TPTP directory
        problems_dir = tptp_path
        if not any(tptp_path.glob("*/*.p")):
            print(f"Error: No Problems directory found in {tptp_path}")
            sys.exit(1)
    
    # Auto-detect or validate TPTP root directory
    if args.tptp_root:
        tptp_root = Path(args.tptp_root)
        if not tptp_root.exists():
            print(f"Error: TPTP root directory not found: {tptp_root}")
            sys.exit(1)
        if not (tptp_root / 'Axioms').exists():
            print(f"Error: Axioms directory not found in: {tptp_root}")
            sys.exit(1)
    else:
        # Try to auto-detect the TPTP root directory
        possible_roots = [
            tptp_path,
            tptp_path.parent / 'TPTP-v9.0.0',
            tptp_path / 'TPTP-v9.0.0',
            tptp_path.parent,
        ]
        tptp_root = None
        for path in possible_roots:
            if (path / 'Axioms').exists():
                tptp_root = path
                print(f"Auto-detected TPTP root directory: {tptp_root}")
                break
        
        if not tptp_root:
            print("Warning: Could not auto-detect TPTP root directory. Include statements may fail.")
            tptp_root = tptp_path  # Fallback
    
    # Create output directory
    if not args.stats_only:
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .p files
    print("Finding TPTP problem files...")
    if args.domain:
        pattern = f"{args.domain}/**/*.p"
        tptp_files = list(problems_dir.glob(pattern))
        print(f"Found {len(tptp_files)} .p files in domain {args.domain}")
    else:
        tptp_files = list(problems_dir.glob("**/*.p"))
        print(f"Found {len(tptp_files)} .p files")
    
    if args.max_files:
        tptp_files = tptp_files[:args.max_files]
        print(f"Processing first {args.max_files} files")
    
    print(f"\nProcessing with literal bound: {args.max_literals}")
    print(f"Pre-scan safety factor: {args.safety_factor}")
    
    # Process files in parallel
    results = process_files_parallel(
        tptp_files, tptp_root, args.max_literals, args.safety_factor, args.workers
    )
    
    # Print statistics
    print(f"\nProcessing results:")
    print(f"  Total files: {len(tptp_files)}")
    print(f"  Successfully parsed: {len(results['parsed_problems'])}")
    print(f"  Skipped (pre-scan): {len(results['skipped_files'])}")
    print(f"  Filtered (actual > limit): {len(results['filtered_files'])}")
    print(f"  Failed to parse: {len(results['failed_files'])}")
    
    if results['parsed_problems']:
        # Compute literal statistics
        literal_counts = [p['num_literals'] for p in results['parsed_problems']]
        print(f"\nLiteral statistics for parsed problems:")
        print(f"  Min: {min(literal_counts)}")
        print(f"  Max: {max(literal_counts)}")
        print(f"  Avg: {sum(literal_counts) / len(literal_counts):.1f}")
    
    # Show examples of skipped files
    if results['skipped_files']:
        print(f"\nExample skipped files (pre-scan estimate too high):")
        sorted_skipped = sorted(results['skipped_files'], 
                               key=lambda x: x['estimated_literals'], reverse=True)
        for f in sorted_skipped[:5]:
            exact_str = " (exact)" if f.get('is_exact') else " (estimate)"
            print(f"  {f['file']}: {f['estimated_literals']} literals{exact_str}")
        if len(results['skipped_files']) > 5:
            print(f"  ... and {len(results['skipped_files']) - 5} more")
    
    # Save results if not stats-only
    if not args.stats_only and results['parsed_problems']:
        # Save problems in batches
        problems = results['parsed_problems']
        num_batches = (len(problems) + args.batch_size - 1) // args.batch_size
        
        for i in range(num_batches):
            start_idx = i * args.batch_size
            end_idx = min((i + 1) * args.batch_size, len(problems))
            batch = problems[start_idx:end_idx]
            
            output_file = output_path / f"tptp_problems_batch_{i:04d}.json"
            with open(output_file, 'w') as f:
                json.dump(batch, f, separators=(',', ':'))
            print(f"Saved batch {i+1}/{num_batches}: {len(batch)} problems")
        
        # Save summary
        summary = {
            'max_literals_filter': args.max_literals,
            'safety_factor': args.safety_factor,
            'total_files': len(tptp_files),
            'parsed_successfully': len(results['parsed_problems']),
            'skipped_prescan': len(results['skipped_files']),
            'filtered_actual': len(results['filtered_files']),
            'failed_to_parse': len(results['failed_files']),
            'problems_by_domain': {},
            'batch_size': args.batch_size,
            'num_batches': num_batches
        }
        
        # Count by domain
        for problem in results['parsed_problems']:
            domain = problem['source_file'].split('/')[0]
            summary['problems_by_domain'][domain] = summary['problems_by_domain'].get(domain, 0) + 1
        
        # Save all output files
        with open(output_path / "parsing_summary.json", 'w') as f:
            json.dump(summary, f, separators=(',', ':'))
        
        if results['skipped_files']:
            with open(output_path / "skipped_files.json", 'w') as f:
                json.dump(results['skipped_files'], f, separators=(',', ':'))
        
        if results['filtered_files']:
            with open(output_path / "filtered_files.json", 'w') as f:
                json.dump(results['filtered_files'], f, separators=(',', ':'))
        
        if results['failed_files']:
            with open(output_path / "failed_files.json", 'w') as f:
                json.dump(results['failed_files'], f, separators=(',', ':'))
        
        print(f"\nSaved all results to {output_path}")


if __name__ == "__main__":
    main()