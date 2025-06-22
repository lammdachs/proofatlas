#!/usr/bin/env python3
"""
TPTP file parsing example.

This example demonstrates how to parse TPTP problem files
and work with the resulting Problem objects.
"""

from pathlib import Path
from proofatlas.fileformats import get_format_handler


def parse_tptp_string():
    """Parse a TPTP problem from a string."""
    print("=== Parsing TPTP from String ===\n")
    
    # Example TPTP content
    tptp_content = """
% Simple propositional problem
% Axioms: P, P->Q, Q->R
% Goal: R

fof(axiom1, axiom, p).
fof(axiom2, axiom, (p => q)).
fof(axiom3, axiom, (q => r)).
fof(goal, conjecture, r).
"""
    
    # Get TPTP format handler
    tptp_handler = get_format_handler(format_name="tptp")
    
    # Parse the content
    problem = tptp_handler.parse_string(tptp_content)
    
    print(f"Parsed problem with {len(problem.clauses)} clauses:")
    for i, clause in enumerate(problem.clauses):
        print(f"  {i}: {clause}")
    
    return problem


def parse_tptp_file():
    """Parse a TPTP problem from a file."""
    print("\n=== Parsing TPTP from File ===\n")
    
    # Check if we have test data
    test_file = Path(__file__).parent.parent / "tests" / ".data" / "problems" / "simple_contradiction.json"
    
    # For this example, let's create a temporary TPTP file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.p', delete=False) as f:
        f.write("""
% Contradiction example
fof(pos, axiom, p(a)).
fof(neg, axiom, ~p(a)).
""")
        temp_path = f.name
    
    try:
        # Get handler by file extension
        handler = get_format_handler(file_path=temp_path)
        
        # Parse the file
        problem = handler.parse_file(temp_path)
        
        print(f"Parsed problem from {Path(temp_path).name}:")
        print(f"  Clauses: {len(problem.clauses)}")
        for i, clause in enumerate(problem.clauses):
            print(f"    {i}: {clause}")
            
    finally:
        # Clean up
        Path(temp_path).unlink()
    
    return problem


def format_problem_to_tptp():
    """Convert a Problem object back to TPTP format."""
    print("\n=== Formatting Problem to TPTP ===\n")
    
    from proofatlas.core import (
        Constant, Predicate, Literal, Clause, Problem
    )
    
    # Create a problem programmatically
    a = Constant("a")
    b = Constant("b")
    P = Predicate("p", 1)
    Q = Predicate("q", 2)
    
    clauses = [
        Clause(Literal(P(a), True)),                    # p(a)
        Clause(Literal(P(b), False), Literal(Q(a, b), True)),  # ~p(b) | q(a,b)
        Clause(Literal(Q(a, b), False))                 # ~q(a,b)
    ]
    
    problem = Problem(*clauses)
    
    # Get TPTP handler and format the problem
    handler = get_format_handler(format_name="tptp")
    tptp_output = handler.format_problem(problem)
    
    print("Generated TPTP:")
    print(tptp_output)
    
    return tptp_output


def demonstrate_tptp_features():
    """Demonstrate various TPTP parsing features."""
    print("\n=== TPTP Format Features ===\n")
    
    handler = get_format_handler(format_name="tptp")
    
    # Different formula types
    examples = [
        ("Atomic formula", "fof(f1, axiom, p)."),
        ("Negation", "fof(f2, axiom, ~q)."),
        ("Conjunction", "fof(f3, axiom, (p & q))."),
        ("Disjunction", "fof(f4, axiom, (p | q))."),
        ("Implication", "fof(f5, axiom, (p => q))."),
        ("Equivalence", "fof(f6, axiom, (p <=> q))."),
        ("Quantified", "fof(f7, axiom, ![X]: p(X))."),
        ("Function terms", "fof(f8, axiom, p(f(a,b)))."),
    ]
    
    for name, tptp in examples:
        try:
            problem = handler.parse_string(tptp)
            print(f"{name}:")
            print(f"  TPTP: {tptp}")
            print(f"  Parsed: {problem.clauses[0] if problem.clauses else 'No clauses'}")
        except Exception as e:
            print(f"{name}: Parse error - {e}")
        print()


def main():
    """Main example function."""
    # Parse from string
    problem1 = parse_tptp_string()
    
    # Parse from file
    problem2 = parse_tptp_file()
    
    # Format to TPTP
    tptp_output = format_problem_to_tptp()
    
    # Show various features
    demonstrate_tptp_features()


if __name__ == "__main__":
    main()