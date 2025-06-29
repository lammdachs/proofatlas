#!/usr/bin/env python3
"""
Unified TPTP extraction script with flexible filtering options.

This script:
1. Finds all .p files in the TPTP directory
2. Optionally pre-scans files to estimate literal count
3. Attempts to parse each file using the Rust parser
4. Filters problems based on criteria (literal count, etc.)
5. Saves successfully parsed problems as JSON
6. Creates a summary of parsed/failed problems
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import proofatlas_rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("Warning: Rust module not available. Using Python parser fallback.")

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
        
        # Count CNF literals (simplified)
        cnf_pattern = r'cnf\([^,]+,[^,]+,\s*\((.*?)\)\s*\)\.'
        cnf_matches = re.findall(cnf_pattern, content, re.DOTALL)
        
        literal_count = 0
        for clause_content in cnf_matches:
            # Count literals by splitting on | and removing whitespace
            literals = [l.strip() for l in clause_content.split('|') if l.strip()]
            literal_count += len(literals)
        
        # Check for includes
        include_pattern = r'include\(\s*[\'"]([^\'\"]+)[\'\"]'
        includes = re.findall(include_pattern, content)
        
        # Process includes
        for include_file in includes:
            # Handle relative paths
            if include_file.startswith('Problems/'):
                include_path = file_path.parent.parent.parent / include_file
            else:
                include_path = file_path.parent / include_file
            
            if include_path.exists():
                sub_count, sub_exact = quick_literal_estimate(include_path, max_depth - 1, visited)
                literal_count += sub_count
                if not sub_exact:
                    return literal_count, False
        
        # Check if file has FOF formulas (harder to estimate)
        if 'fof(' in content:
            # Can't accurately estimate FOF literal count
            return literal_count, False
        
        return literal_count, True
        
    except Exception:
        return 0, False


def parse_problem(args: Tuple[Path, str, Optional[int], bool]) -> Tuple[Path, Optional[Dict], Optional[str], Optional[int]]:
    """Parse a single TPTP file."""
    file_path, include_path, max_literals, use_prescan = args
    
    # Pre-scan if requested
    if use_prescan and max_literals is not None:
        estimated_literals, is_exact = quick_literal_estimate(file_path)
        if is_exact and estimated_literals > max_literals:
            return file_path, None, "Exceeded literal limit (prescan)", estimated_literals
    
    try:
        # Try Rust parser first if available
        if RUST_AVAILABLE:
            problem = proofatlas_rust.parser.parse_file(str(file_path), include_path)
            
            # Convert to JSON
            json_str = problem.to_json()
            data = json.loads(json_str)
            
            # Add metadata
            data['metadata'] = {
                'source_file': file_path.name,
                'num_clauses': len(problem),
                'num_literals': sum(len(clause['literals']) for clause in data['clauses'])
            }
            
            # Check literal count
            if max_literals is not None and data['metadata']['num_literals'] > max_literals:
                return file_path, None, f"Exceeded literal limit ({data['metadata']['num_literals']} > {max_literals})", data['metadata']['num_literals']
            
            return file_path, data, None, data['metadata']['num_literals']
            
        else:
            # Fallback to Python parser
            parser = TPTPFormat()
            problem = parser.parse_file(str(file_path))
            
            # Count literals
            num_literals = sum(len(clause.literals) for clause in problem.clauses)
            
            # Check literal count
            if max_literals is not None and num_literals > max_literals:
                return file_path, None, f"Exceeded literal limit ({num_literals} > {max_literals})", num_literals
            
            # Convert to dict (simplified)
            data = {
                'clauses': [
                    {
                        'literals': [
                            {
                                'polarity': lit.polarity,
                                'predicate': str(lit.predicate)
                            }
                            for lit in clause.literals
                        ]
                    }
                    for clause in problem.clauses
                ],
                'metadata': {
                    'source_file': file_path.name,
                    'num_clauses': len(problem.clauses),
                    'num_literals': num_literals
                }
            }
            
            return file_path, data, None, num_literals
            
    except Exception as e:
        return file_path, None, str(e), None


def main():
    parser = argparse.ArgumentParser(description="Extract TPTP problems with flexible filtering")
    parser.add_argument("tptp_dir", help="Directory containing TPTP files")
    parser.add_argument("output_dir", help="Directory to save extracted problems")
    parser.add_argument("--max-literals", type=int, help="Maximum number of literals per problem")
    parser.add_argument("--prescan", action="store_true", help="Pre-scan files to estimate literal count")
    parser.add_argument("--include-path", help="Path to TPTP library for includes")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--limit", type=int, help="Maximum number of problems to extract")
    parser.add_argument("--format", choices=['compact', 'pretty'], default='pretty', 
                       help="JSON output format")
    
    args = parser.parse_args()
    
    # Find all .p files
    tptp_path = Path(args.tptp_dir)
    if not tptp_path.exists():
        print(f"Error: TPTP directory not found: {tptp_path}")
        sys.exit(1)
    
    problem_files = list(tptp_path.rglob("*.p"))
    print(f"Found {len(problem_files)} TPTP files")
    
    # Apply limit if specified
    if args.limit:
        problem_files = problem_files[:args.limit]
        print(f"Limited to {len(problem_files)} files")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set up include path
    include_path = args.include_path or str(tptp_path)
    
    # Parse problems in parallel
    parse_args = [(f, include_path, args.max_literals, args.prescan) for f in problem_files]
    
    results = []
    with mp.Pool(args.workers) as pool:
        with tqdm(total=len(problem_files), desc="Parsing") as pbar:
            for result in pool.imap_unordered(parse_problem, parse_args):
                results.append(result)
                pbar.update(1)
    
    # Process results
    successful = []
    failed = []
    filtered = []
    
    for file_path, data, error, literal_count in results:
        if data is not None:
            successful.append((file_path, data))
        elif error and "Exceeded literal limit" in error:
            filtered.append((file_path, error, literal_count))
        else:
            failed.append((file_path, error))
    
    print(f"\nSuccessfully parsed: {len(successful)}")
    print(f"Filtered by literal count: {len(filtered)}")
    print(f"Failed to parse: {len(failed)}")
    
    # Save successful problems
    for file_path, data in tqdm(successful, desc="Saving"):
        # Create relative path structure
        rel_path = file_path.relative_to(tptp_path)
        output_file = output_path / rel_path.with_suffix('.json')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            if args.format == 'compact':
                json.dump(data, f, separators=(',', ':'))
            else:
                json.dump(data, f, indent=2)
    
    # Save summary
    summary = {
        'total_files': len(problem_files),
        'successful': len(successful),
        'filtered': len(filtered),
        'failed': len(failed),
        'max_literals': args.max_literals,
        'used_prescan': args.prescan,
        'filtered_problems': [
            {
                'file': str(f.relative_to(tptp_path)),
                'reason': reason,
                'literal_count': lit_count
            }
            for f, reason, lit_count in filtered
        ],
        'failed_problems': [
            {
                'file': str(f.relative_to(tptp_path)),
                'error': error
            }
            for f, error in failed[:100]  # Limit to first 100 failures
        ]
    }
    
    with open(output_path / 'extraction_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nExtraction complete. Summary saved to {output_path / 'extraction_summary.json'}")


if __name__ == "__main__":
    main()