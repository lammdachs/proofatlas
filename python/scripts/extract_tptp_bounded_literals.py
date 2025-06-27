#!/usr/bin/env python3
"""
Extract TPTP problems with a bounded number of literals and save them as JSON.

This script:
1. Finds all .p files in the TPTP directory
2. Attempts to parse each one
3. Filters problems by total literal count
4. Saves filtered problems as compact JSON
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from proofatlas.fileformats.tptp import TPTPFormat
from proofatlas.core.logic import Problem


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


def parse_single_file(file_info: Tuple[Path, Path, int]) -> Tuple[str, bool, Any, int]:
    """
    Parse a single TPTP file and check literal count.
    
    Args:
        file_info: Tuple of (file_path, tptp_root, max_literals)
    
    Returns:
        Tuple of (relative_path, success, result, num_literals)
        If success is True, result is the parsed problem dict
        If success is False, result is the error message or None if filtered
    """
    file_path, tptp_root, max_literals = file_info
    from proofatlas.fileformats.tptp_parser.parser import read_file
    relative_path = str(file_path.relative_to(file_path.parent.parent.parent))
    
    try:
        problem = read_file(str(file_path), include_path=str(tptp_root) + '/')
        num_literals = count_literals(problem)
        
        # Check if problem meets the literal bound
        if num_literals > max_literals:
            return (relative_path, False, None, num_literals)
        
        problem_dict = problem_to_dict(problem, relative_path)
        return (relative_path, True, problem_dict, num_literals)
    except Exception as e:
        return (relative_path, False, str(e), -1)


def parse_files_parallel(tptp_files: List[Path], tptp_root: Path, max_literals: int, num_workers: int = None) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
    """Parse files in parallel using multiprocessing."""
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    parsed_problems = []
    failed_files = []
    filtered_files = []
    literal_counts = {}
    
    # Create list of (file_path, tptp_root, max_literals) tuples
    file_infos = [(f, tptp_root, max_literals) for f in tptp_files]
    
    # Create a pool of workers
    with mp.Pool(num_workers) as pool:
        # Process files in parallel with progress bar
        results = list(tqdm(
            pool.imap(parse_single_file, file_infos),
            total=len(tptp_files),
            desc="Parsing TPTP files"
        ))
    
    # Separate successful, failed, and filtered parses
    for relative_path, success, result, num_literals in results:
        if success:
            parsed_problems.append(result)
            literal_counts[relative_path] = num_literals
        elif result is None:  # Filtered due to literal count
            filtered_files.append({
                'file': relative_path,
                'num_literals': num_literals
            })
            literal_counts[relative_path] = num_literals
        else:  # Parse error
            failed_files.append({
                'file': relative_path,
                'error': result
            })
    
    return parsed_problems, failed_files, filtered_files, literal_counts


def main():
    parser = argparse.ArgumentParser(description='Extract TPTP problems with bounded literals to JSON')
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
        # Try to auto-detect the TPTP root directory (contains both Problems and Axioms)
        possible_roots = [
            tptp_path,  # Already points to TPTP-v9.0.0
            tptp_path.parent / 'TPTP-v9.0.0',
            tptp_path / 'TPTP-v9.0.0',
            tptp_path.parent,
        ]
        tptp_root = None
        for path in possible_roots:
            if (path / 'Axioms').exists():
                tptp_root = path
                print(f"Auto-detected TPTP root directory: {tptp_root}")
                print(f"Axioms directory: {tptp_root / 'Axioms'}")
                break
        
        if not tptp_root:
            print("Warning: Could not auto-detect TPTP root directory. Include statements may fail.")
            print("Use --tptp-root to specify the TPTP root directory containing the Axioms folder.")
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
    
    print(f"\nFiltering problems with <= {args.max_literals} literals...")
    
    # Parse files in parallel
    parsed_problems, failed_files, filtered_files, literal_counts = parse_files_parallel(
        tptp_files, tptp_root, args.max_literals, args.workers
    )
    
    # Print statistics
    print(f"\nParsing results:")
    print(f"  Total files processed: {len(tptp_files)}")
    print(f"  Successfully parsed (within bound): {len(parsed_problems)}")
    print(f"  Filtered (exceeded literal bound): {len(filtered_files)}")
    print(f"  Failed to parse: {len(failed_files)}")
    
    if parsed_problems:
        # Compute literal statistics for accepted problems
        accepted_literal_counts = [p['num_literals'] for p in parsed_problems]
        print(f"\nLiteral statistics for accepted problems:")
        print(f"  Min literals: {min(accepted_literal_counts)}")
        print(f"  Max literals: {max(accepted_literal_counts)}")
        print(f"  Avg literals: {sum(accepted_literal_counts) / len(accepted_literal_counts):.1f}")
    
    if filtered_files:
        # Show some examples of filtered files
        print(f"\nExample filtered files (> {args.max_literals} literals):")
        sorted_filtered = sorted(filtered_files, key=lambda x: x['num_literals'], reverse=True)
        for f in sorted_filtered[:5]:
            print(f"  {f['file']}: {f['num_literals']} literals")
        if len(filtered_files) > 5:
            print(f"  ... and {len(filtered_files) - 5} more")
    
    # Save if not stats-only mode
    if not args.stats_only and parsed_problems:
        # Save problems in batches
        num_batches = (len(parsed_problems) + args.batch_size - 1) // args.batch_size
        for i in range(num_batches):
            start_idx = i * args.batch_size
            end_idx = min((i + 1) * args.batch_size, len(parsed_problems))
            batch = parsed_problems[start_idx:end_idx]
            
            output_file = output_path / f"tptp_problems_batch_{i:04d}.json"
            with open(output_file, 'w') as f:
                json.dump(batch, f, separators=(',', ':'))
            print(f"Saved batch {i+1}/{num_batches}: {len(batch)} problems to {output_file}")
        
        # Save summary
        summary = {
            'max_literals_filter': args.max_literals,
            'total_files': len(tptp_files),
            'parsed_successfully': len(parsed_problems),
            'filtered_by_literal_count': len(filtered_files),
            'failed_to_parse': len(failed_files),
            'problems_by_domain': {},
            'batch_size': args.batch_size,
            'num_batches': num_batches,
            'literal_statistics': {
                'min': min(accepted_literal_counts) if accepted_literal_counts else 0,
                'max': max(accepted_literal_counts) if accepted_literal_counts else 0,
                'avg': sum(accepted_literal_counts) / len(accepted_literal_counts) if accepted_literal_counts else 0
            }
        }
        
        # Count problems by domain
        for problem in parsed_problems:
            domain = problem['source_file'].split('/')[0]
            summary['problems_by_domain'][domain] = summary['problems_by_domain'].get(domain, 0) + 1
        
        summary_file = output_path / "parsing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, separators=(',', ':'))
        print(f"\nSaved summary to {summary_file}")
        
        # Save failed files list
        if failed_files:
            failed_file = output_path / "failed_files.json"
            with open(failed_file, 'w') as f:
                json.dump(failed_files, f, separators=(',', ':'))
            print(f"Saved failed files list to {failed_file}")
        
        # Save filtered files list
        if filtered_files:
            filtered_file = output_path / "filtered_files.json"
            with open(filtered_file, 'w') as f:
                json.dump(filtered_files, f, separators=(',', ':'))
            print(f"Saved filtered files list to {filtered_file}")


if __name__ == "__main__":
    main()