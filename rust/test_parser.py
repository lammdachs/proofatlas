#!/usr/bin/env python3
"""Test script for the Rust TPTP parser"""

import sys
import os
import time

# For development - add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python', 'src'))

try:
    import proofatlas_rust
    print("✓ Successfully imported proofatlas_rust")
except ImportError as e:
    print(f"✗ Failed to import proofatlas_rust: {e}")
    print("\nPlease build the module first:")
    print("  cd rust")
    print("  maturin develop")
    sys.exit(1)

from proofatlas.core.logic import Problem

def test_basic_parsing():
    """Test basic TPTP parsing"""
    print("\n=== Testing Basic Parsing ===")
    
    # Simple CNF example
    tptp_content = """
    cnf(clause1, axiom, (p(X) | ~q(X))).
    cnf(clause2, axiom, (q(a))).
    cnf(clause3, negated_conjecture, (~p(a))).
    """
    
    problem = proofatlas_rust.parser.parse_string(tptp_content)
    print(f"Parsed problem type: {type(problem)}")
    print(f"Number of clauses: {len(problem.clauses)}")
    print(f"Is Python Problem instance: {isinstance(problem, Problem)}")
    print(f"Conjecture indices: {problem.conjecture_indices}")
    
    for i, clause in enumerate(problem.clauses):
        marker = "[CONJ]" if problem.is_conjecture_clause(i) else "      "
        print(f"{i+1}. {marker} {clause}")

def test_file_parsing():
    """Test file parsing with example files"""
    print("\n=== Testing File Parsing ===")
    
    # Try to find an example TPTP file
    test_files = [
        "ALG001-1.p",
        "PUZ001-1.p",
        "../examples/simple.p",
    ]
    
    tptp_path = os.environ.get('TPTP_PATH', '')
    if tptp_path:
        parser = proofatlas_rust.parser.RustTPTPParser(include_path=tptp_path)
        
        for test_file in test_files:
            full_path = os.path.join(tptp_path, "Problems", test_file[:3], test_file)
            if os.path.exists(full_path):
                print(f"\nParsing {test_file}...")
                try:
                    start = time.time()
                    problem = parser.parse_file(full_path)
                    elapsed = time.time() - start
                    
                    print(f"✓ Success! Parsed in {elapsed:.3f}s")
                    print(f"  Clauses: {len(problem.clauses)}")
                    print(f"  Literals: {sum(len(c.literals) for c in problem.clauses)}")
                    
                except Exception as e:
                    print(f"✗ Failed: {e}")
                break
    else:
        print("TPTP_PATH not set, skipping file tests")

def test_prescan():
    """Test pre-scanning functionality"""
    print("\n=== Testing Pre-scan ===")
    
    # Create a test file with known content
    test_content = """
    cnf(c1, axiom, (p(X) | q(X) | r(X))).
    cnf(c2, axiom, (s(Y) | t(Y))).
    cnf(c3, axiom, (u(Z))).
    """
    
    # Save to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.p', delete=False) as f:
        f.write(test_content)
        temp_file = f.name
    
    try:
        count, is_exact = proofatlas_rust.parser.prescan_file(temp_file)
        print(f"Pre-scan result: {count} literals (exact: {is_exact})")
        
        # Verify with actual parse
        problem = proofatlas_rust.parser.parse_file(temp_file)
        actual_count = sum(len(c.literals) for c in problem.clauses)
        print(f"Actual count: {actual_count} literals")
        
    finally:
        os.unlink(temp_file)

def test_dict_conversion():
    """Test dictionary conversion for JSON"""
    print("\n=== Testing Dictionary Conversion ===")
    
    tptp_content = """
    cnf(test, axiom, (f(X,a) = g(b,Y))).
    """
    
    parser = proofatlas_rust.parser.RustTPTPParser()
    
    # This would work if we had a test file
    # problem_dict = parser.parse_file_to_dict("test.p")
    
    # For now, test with parse_string and manual dict conversion
    problem = proofatlas_rust.parser.parse_string(tptp_content)
    print(f"Parsed clause: {problem.clauses[0]}")
    print(f"First literal: {problem.clauses[0].literals[0]}")

def benchmark_comparison():
    """Compare Rust vs Python parser performance"""
    print("\n=== Performance Comparison ===")
    
    # Only run if we have both parsers available
    try:
        from proofatlas.fileformats.tptp_parser.parser import read_string as python_parse
        
        # Test content with multiple clauses
        test_content = "\n".join([
            f"cnf(c{i}, axiom, (p{i}(X) | ~q{i}(Y,Z)))."
            for i in range(100)
        ])
        
        # Time Python parser
        start = time.time()
        python_problem = python_parse(test_content)
        python_time = time.time() - start
        
        # Time Rust parser
        start = time.time()
        rust_problem = proofatlas_rust.parser.parse_string(test_content)
        rust_time = time.time() - start
        
        print(f"Python parser: {python_time:.4f}s")
        print(f"Rust parser:   {rust_time:.4f}s")
        print(f"Speedup:       {python_time/rust_time:.1f}x")
        
        # Verify same results
        print(f"Same clause count: {len(python_problem.clauses) == len(rust_problem.clauses)}")
        
    except ImportError:
        print("Python parser not available for comparison")

if __name__ == "__main__":
    print("Testing ProofAtlas Rust Parser")
    print("=" * 40)
    
    test_basic_parsing()
    test_file_parsing()
    test_prescan()
    test_dict_conversion()
    benchmark_comparison()
    
    print("\n✓ All tests completed!")