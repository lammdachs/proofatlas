#!/usr/bin/env python3
"""
Parse and display a TPTP file.

Usage:
    python scripts/parse_tptp.py <tptp_file>
    
Example:
    python scripts/parse_tptp.py ALG001-1.p
    python scripts/parse_tptp.py problems/ALG/ALG001-1.p
"""

import sys
import os
from pathlib import Path
from collections import defaultdict

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from proofatlas.fileformats.tptp_parser.parser import read_file
from proofatlas.core.logic import Predicate, Function, Variable, Constant


def analyze_problem(problem):
    """Analyze a parsed problem and return statistics."""
    stats = {
        'num_clauses': len(problem.clauses),
        'num_literals': 0,
        'predicates': defaultdict(int),
        'functions': defaultdict(int),
        'variables': set(),
        'constants': set(),
        'max_clause_size': 0,
        'min_clause_size': float('inf'),
        'avg_clause_size': 0,
        'positive_literals': 0,
        'negative_literals': 0,
        'num_conjecture_clauses': len(problem.conjecture_indices),
        'conjecture_indices': sorted(problem.conjecture_indices)
    }
    
    for clause in problem.clauses:
        clause_size = len(clause.literals)
        stats['num_literals'] += clause_size
        stats['max_clause_size'] = max(stats['max_clause_size'], clause_size)
        stats['min_clause_size'] = min(stats['min_clause_size'], clause_size)
        
        for literal in clause.literals:
            # Count positive/negative literals
            if literal.polarity:
                stats['positive_literals'] += 1
            else:
                stats['negative_literals'] += 1
            
            # Count predicates
            pred_key = f"{literal.predicate.symbol.name}/{literal.predicate.symbol.arity}"
            stats['predicates'][pred_key] += 1
            
            # Collect terms
            def collect_from_term(term):
                if isinstance(term, Variable):
                    stats['variables'].add(term.name)
                elif isinstance(term, Constant):
                    stats['constants'].add(term.name)
                elif isinstance(term, Function) and not isinstance(term, (Variable, Constant)):
                    func_key = f"{term.symbol.name}/{term.symbol.arity}"
                    stats['functions'][func_key] += 1
                    for arg in term.args:
                        collect_from_term(arg)
            
            for term in literal.predicate.args:
                collect_from_term(term)
    
    if stats['num_clauses'] > 0:
        stats['avg_clause_size'] = stats['num_literals'] / stats['num_clauses']
    
    return stats


def display_problem(problem, stats, show_clauses=True):
    """Display the parsed problem and its statistics."""
    print("=" * 60)
    print("PROBLEM STATISTICS")
    print("=" * 60)
    print(f"Clauses:           {stats['num_clauses']}")
    print(f"  Axioms:          {stats['num_clauses'] - stats['num_conjecture_clauses']}")
    print(f"  Conjectures:     {stats['num_conjecture_clauses']}")
    if stats['conjecture_indices']:
        print(f"    Indices:       {stats['conjecture_indices']}")
    print(f"Literals:          {stats['num_literals']}")
    print(f"  Positive:        {stats['positive_literals']}")
    print(f"  Negative:        {stats['negative_literals']}")
    print(f"Clause sizes:      min={stats['min_clause_size']}, "
          f"max={stats['max_clause_size']}, "
          f"avg={stats['avg_clause_size']:.1f}")
    print(f"Variables:         {len(stats['variables'])} "
          f"({', '.join(sorted(stats['variables'])) if stats['variables'] else 'none'})")
    print(f"Constants:         {len(stats['constants'])} "
          f"({', '.join(sorted(stats['constants'])) if stats['constants'] else 'none'})")
    
    print(f"\nPredicates ({len(stats['predicates'])}):")
    for pred, count in sorted(stats['predicates'].items()):
        print(f"  {pred}: {count} occurrences")
    
    if stats['functions']:
        print(f"\nFunctions ({len(stats['functions'])}):")
        for func, count in sorted(stats['functions'].items()):
            print(f"  {func}: {count} occurrences")
    
    if show_clauses:
        print("\n" + "=" * 60)
        print("CLAUSES")
        print("=" * 60)
        for i, clause in enumerate(problem.clauses):
            marker = "[CONJ]" if problem.is_conjecture_clause(i) else "      "
            print(f"{i+1:3d}. {marker} {clause}")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    
    tptp_file = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(tptp_file):
        # Try common TPTP problem directories
        search_paths = [tptp_file]
        
        # Check TPTP_PATH environment variable
        tptp_path = os.environ.get('TPTP_PATH')
        if tptp_path:
            # Extract problem domain from filename (first 3 letters)
            if len(tptp_file) >= 3:
                domain = tptp_file[:3].upper()
                search_paths.extend([
                    os.path.join(tptp_path, 'Problems', domain, tptp_file),
                    os.path.join(tptp_path, 'Problems', domain, tptp_file.upper()),
                ])
            search_paths.append(os.path.join(tptp_path, 'Problems', tptp_file))
        
        # Try other common locations
        search_paths.extend([
            f"problems/{tptp_file}",
            f"data/problems/{tptp_file}",
            f".data/problems/{tptp_file}",
            f"TPTP/Problems/{tptp_file[:3]}/{tptp_file}" if len(tptp_file) >= 3 else None,
            f"data/TPTP/Problems/{tptp_file[:3]}/{tptp_file}" if len(tptp_file) >= 3 else None
        ])
        
        # Remove None values
        search_paths = [p for p in search_paths if p]
        
        found = False
        for path in search_paths:
            if os.path.exists(path):
                tptp_file = path
                found = True
                break
        
        if not found:
            print(f"Error: File '{tptp_file}' not found.")
            print("Searched in:")
            for path in search_paths:
                print(f"  - {path}")
            if not tptp_path:
                print("\nHint: Set TPTP_PATH environment variable to your TPTP library location.")
            sys.exit(1)
    
    print(f"Parsing: {tptp_file}")
    
    try:
        # Parse the file
        # Determine include path for resolving axiom includes
        tptp_root = os.environ.get('TPTP_PATH', '')
        if tptp_root and os.path.exists(tptp_root):
            include_path = str(tptp_root) + '/'
        else:
            # Use file's parent directory as fallback
            include_path = str(Path(tptp_file).parent) + '/'
        
        import time
        parse_start = time.time()
        problem = read_file(tptp_file, include_path=include_path)
        parse_time = time.time() - parse_start
        
        # Analyze the problem
        stats = analyze_problem(problem)
        
        # Display results
        display_problem(problem, stats, show_clauses=(stats['num_clauses'] <= 50))
        
        if stats['num_clauses'] > 50:
            print(f"\n(Showing first 10 and last 10 clauses of {stats['num_clauses']} total)")
            print("-" * 60)
            for i in range(min(10, len(problem.clauses))):
                marker = "[CONJ]" if problem.is_conjecture_clause(i) else "      "
                print(f"{i+1:3d}. {marker} {problem.clauses[i]}")
            if len(problem.clauses) > 20:
                print("     ...")
            for i in range(max(10, len(problem.clauses) - 10), len(problem.clauses)):
                marker = "[CONJ]" if problem.is_conjecture_clause(i) else "      "
                print(f"{i+1:3d}. {marker} {problem.clauses[i]}")
        
        # Print parsing time
        print(f"\nParsing completed in {parse_time:.3f} seconds")
        
    except Exception as e:
        print(f"Error parsing file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()