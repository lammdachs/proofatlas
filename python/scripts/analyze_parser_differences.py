#!/usr/bin/env python3
"""
Analyze differences between Rust and Python TPTP parsers.

This script provides detailed analysis of parsing differences.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

# Add the python directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from proofatlas.core.logic import Problem, Clause, Literal
from proofatlas.fileformats.tptp import TPTPFormat

# Import the Rust parser
try:
    import proofatlas_rust
except ImportError:
    print("Error: proofatlas_rust module not found. Please build it with 'cd rust && maturin develop'")
    sys.exit(1)


def clause_to_string(clause: Clause) -> str:
    """Convert a clause to a normalized string representation."""
    literals = []
    for lit in clause.literals:
        pred_name = lit.predicate.name
        neg = "~" if lit.negated else ""
        args = ",".join(str(arg) for arg in lit.arguments)
        literals.append(f"{neg}{pred_name}({args})")
    return " | ".join(sorted(literals))


def analyze_file(file_path: Path) -> Dict:
    """Analyze parsing differences for a single file."""
    result = {
        "file": str(file_path),
        "rust_success": False,
        "python_success": False,
        "differences": []
    }
    
    # Parse with Rust
    try:
        rust_data = proofatlas_rust.parser.parse_file_to_dict(str(file_path))
        result["rust_success"] = True
        result["rust_clause_count"] = len(rust_data.get("clauses", []))
        rust_clauses = rust_data.get("clauses", [])
    except Exception as e:
        result["rust_error"] = str(e)
        rust_clauses = []
    
    # Parse with Python
    try:
        parser = TPTPFormat()
        python_problem = parser.parse_file(file_path)
        result["python_success"] = True
        result["python_clause_count"] = len(python_problem.clauses)
        python_clauses = [clause_to_string(clause) for clause in python_problem.clauses]
    except Exception as e:
        result["python_error"] = str(e)
        python_clauses = []
    
    # Compare if both successful
    if result["rust_success"] and result["python_success"]:
        # Convert Rust clauses to string format for comparison
        rust_clause_strs = []
        for clause in rust_clauses:
            literals = []
            for lit in clause.get("literals", []):
                neg = "~" if lit.get("polarity", True) == False else ""
                pred = lit.get("predicate", {})
                pred_name = pred.get("symbol", pred.get("name", "?"))
                args = lit.get("terms", lit.get("arguments", []))
                arg_strs = []
                for arg in args:
                    if isinstance(arg, dict):
                        if arg.get("type") == "variable":
                            arg_strs.append(arg["name"])
                        elif arg.get("type") == "constant":
                            arg_strs.append(arg["name"])
                        elif arg.get("type") == "function":
                            func_args = ",".join(str(a) for a in arg.get("args", []))
                            arg_strs.append(f"{arg['name']}({func_args})")
                        else:
                            arg_strs.append(str(arg))
                    else:
                        arg_strs.append(str(arg))
                literals.append(f"{neg}{pred_name}({','.join(arg_strs)})")
            rust_clause_strs.append(" | ".join(sorted(literals)))
        
        # Compare clause sets
        rust_set = set(rust_clause_strs)
        python_set = set(python_clauses)
        
        only_rust = rust_set - python_set
        only_python = python_set - rust_set
        
        if only_rust:
            result["differences"].append(f"Clauses only in Rust ({len(only_rust)}): {list(only_rust)[:3]}")
        if only_python:
            result["differences"].append(f"Clauses only in Python ({len(only_python)}): {list(only_python)[:3]}")
            
        result["clause_count_match"] = len(rust_clauses) == len(python_clauses)
        result["clause_content_match"] = rust_set == python_set
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Analyze parser differences in detail")
    parser.add_argument("file", type=str, help="TPTP file to analyze")
    
    args = parser.parse_args()
    
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist")
        sys.exit(1)
        
    print(f"Analyzing {file_path}...\n")
    
    result = analyze_file(file_path)
    
    print(f"Rust parser: {'SUCCESS' if result['rust_success'] else 'FAILED'}")
    if result['rust_success']:
        print(f"  Clauses: {result['rust_clause_count']}")
    else:
        print(f"  Error: {result.get('rust_error', 'Unknown')}")
        
    print(f"\nPython parser: {'SUCCESS' if result['python_success'] else 'FAILED'}")
    if result['python_success']:
        print(f"  Clauses: {result['python_clause_count']}")
    else:
        print(f"  Error: {result.get('python_error', 'Unknown')}")
        
    if result['differences']:
        print(f"\nDifferences found:")
        for diff in result['differences']:
            print(f"  - {diff}")
    elif result['rust_success'] and result['python_success']:
        print(f"\nNo differences found - parsers agree!")


if __name__ == "__main__":
    main()