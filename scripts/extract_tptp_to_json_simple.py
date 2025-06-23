#!/usr/bin/env python3
"""
Simple single-threaded version to extract TPTP problems to JSON.

This script:
1. Finds all .p files in the TPTP directory
2. Attempts to parse each one
3. Saves successfully parsed problems as JSON
4. Creates a summary of parsed/failed problems
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from proofatlas.fileformats.tptp import TPTPFormat
from proofatlas.core.logic import Problem, Clause, Literal, Term


def serialize_term(term) -> Dict[str, Any]:
    """Convert a Term object to a JSON-serializable dictionary."""
    # Check type by class name to properly distinguish Variables
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
            'functor': term.symbol.name,
            'arity': term.symbol.arity,
            'arguments': [serialize_term(arg) for arg in term.args]
        }


def serialize_literal(literal: Literal) -> Dict[str, Any]:
    """Convert a Literal object to a JSON-serializable dictionary."""
    return {
        'polarity': literal.polarity,
        'predicate': literal.predicate.symbol.name,
        'arity': literal.predicate.symbol.arity,
        'terms': [serialize_term(t) for t in literal.predicate.args]
    }


def serialize_clause(clause: Clause) -> Dict[str, Any]:
    """Convert a Clause object to a JSON-serializable dictionary."""
    return {
        'literals': [serialize_literal(lit) for lit in clause.literals],
        'is_empty': len(clause.literals) == 0
    }


def serialize_problem(problem: Problem, source_file: str) -> Dict[str, Any]:
    """Convert a Problem object to a JSON-serializable dictionary."""
    # Collect statistics
    all_predicates = set()
    all_functions = set()
    all_constants = set()
    max_clause_size = 0
    
    for clause in problem.clauses:
        max_clause_size = max(max_clause_size, len(clause.literals))
        for lit in clause.literals:
            all_predicates.add((lit.predicate.symbol.name, lit.predicate.symbol.arity))
            for term in lit.predicate.args:
                collect_symbols_from_term(term, all_functions, all_constants)
    
    return {
        'source_file': source_file,
        'statistics': {
            'num_clauses': len(problem.clauses),
            'num_predicates': len(all_predicates),
            'num_functions': len(all_functions),
            'num_constants': len(all_constants),
            'max_clause_size': max_clause_size,
            'predicates': sorted(list(all_predicates)),
            'functions': sorted(list(all_functions)),
            'constants': sorted(list(all_constants))
        },
        'clauses': [serialize_clause(c) for c in problem.clauses]
    }


def collect_symbols_from_term(term, functions: set, constants: set):
    """Recursively collect function and constant symbols from a term."""
    type_name = type(term).__name__
    
    if type_name == 'Variable':
        pass
    elif type_name == 'Constant':
        constants.add(term.name)
    else:  # Function term
        functions.add((term.symbol.name, term.symbol.arity))
        for arg in term.args:
            collect_symbols_from_term(arg, functions, constants)


def main():
    parser = argparse.ArgumentParser(description='Extract TPTP problems to JSON (simple version)')
    parser.add_argument('--tptp-dir', type=str, 
                       default=os.getenv('TPTP_PATH', './.data/problems/tptp'),
                       help='Path to TPTP directory (default: from TPTP_PATH env var)')
    parser.add_argument('--tptp-root', type=str,
                       help='Path to TPTP root directory for resolving includes (auto-detected if not specified)')
    parser.add_argument('--output-dir', type=str, 
                       default=os.path.join(os.getenv('DATASETS_DIR', './.data/datasets'), 'tptp_json'),
                       help='Output directory for JSON files (default: $DATASETS_DIR/tptp_json)')
    parser.add_argument('--max-files', type=int, help='Maximum number of files to process')
    parser.add_argument('--domain', type=str, help='Only process files from specific domain (e.g., PUZ, ALG)')
    parser.add_argument('--save-individual', action='store_true', 
                       help='Save each problem as individual JSON file')
    args = parser.parse_args()
    
    tptp_path = Path(args.tptp_dir)
    output_path = Path(args.output_dir)
    
    # Validate TPTP directory
    if not tptp_path.exists():
        print(f"Error: TPTP directory not found: {tptp_path}")
        sys.exit(1)
    
    # Find Problems directory
    if (tptp_path / "Problems").exists():
        problems_dir = tptp_path / "Problems"
    else:
        problems_dir = tptp_path
    
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
    
    # Parse files
    tptp_parser = TPTPFormat()
    parsed_problems = []
    failed_files = []
    
    print("\nParsing files...")
    for file_path in tqdm(tptp_files):
        relative_path = str(file_path.relative_to(problems_dir))
        
        try:
            # We need to create a custom parser that uses the correct include path
            # Pass the TPTP root directory for resolving includes
            from proofatlas.fileformats.tptp_parser.parser import read_file
            problem = read_file(str(file_path), include_path=str(tptp_root) + '/')
            problem_dict = serialize_problem(problem, relative_path)
            parsed_problems.append(problem_dict)
            
            # Save individual file if requested
            if args.save_individual:
                individual_output = output_path / "individual" / relative_path.replace('.p', '.json')
                individual_output.parent.mkdir(parents=True, exist_ok=True)
                with open(individual_output, 'w') as f:
                    json.dump(problem_dict, f, indent=2)
                    
        except Exception as e:
            failed_files.append({
                'file': relative_path,
                'error': str(e),
                'error_type': type(e).__name__
            })
    
    # Save all problems together
    print(f"\nSuccessfully parsed {len(parsed_problems)} problems")
    print(f"Failed to parse {len(failed_files)} files")
    
    if parsed_problems:
        all_problems_file = output_path / "all_problems.json"
        print(f"\nSaving all problems to {all_problems_file}...")
        with open(all_problems_file, 'w') as f:
            json.dump(parsed_problems, f, indent=2)
    
    # Save summary
    summary = {
        'total_files': len(tptp_files),
        'parsed_successfully': len(parsed_problems),
        'failed_to_parse': len(failed_files),
        'problems_by_domain': {},
        'error_types': {}
    }
    
    # Count problems by domain
    for problem in parsed_problems:
        domain = problem['source_file'].split('/')[0]
        summary['problems_by_domain'][domain] = summary['problems_by_domain'].get(domain, 0) + 1
    
    # Count error types
    for failure in failed_files:
        error_type = failure['error_type']
        summary['error_types'][error_type] = summary['error_types'].get(error_type, 0) + 1
    
    summary_file = output_path / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_file}")
    
    # Save failed files list
    if failed_files:
        failed_file = output_path / "failed_files.json"
        with open(failed_file, 'w') as f:
            json.dump(failed_files, f, indent=2)
        print(f"Saved {len(failed_files)} failed files to {failed_file}")
        
        # Print example failures by error type
        print("\nExample failures by error type:")
        errors_by_type = {}
        for failure in failed_files:
            error_type = failure['error_type']
            if error_type not in errors_by_type:
                errors_by_type[error_type] = []
            errors_by_type[error_type].append(failure)
        
        for error_type, failures in sorted(errors_by_type.items()):
            print(f"\n{error_type} ({len(failures)} files):")
            for failure in failures[:2]:  # Show 2 examples per type
                print(f"  {failure['file']}: {failure['error']}")


if __name__ == "__main__":
    main()