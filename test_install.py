#!/usr/bin/env python3
"""Test that the ProofAtlas installation works correctly."""

import sys

def test_import():
    """Test basic import"""
    try:
        from proofatlas import ProofState, ClauseInfo, saturate_step
        print("✓ Successfully imported ProofAtlas modules")
        return True
    except ImportError as e:
        print(f"✗ Failed to import ProofAtlas: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    try:
        from proofatlas import ProofState
        
        # Create a proof state
        state = ProofState()
        print("✓ Created ProofState instance")
        
        # Add some clauses
        tptp = """
        cnf(c1, axiom, p(a)).
        cnf(c2, axiom, ~p(X) | q(X)).
        cnf(c3, axiom, ~q(a)).
        """
        
        clause_ids = state.add_clauses_from_tptp(tptp)
        print(f"✓ Added {len(clause_ids)} clauses from TPTP")
        
        # Check state
        print(f"✓ State has {state.num_clauses()} clauses")
        
        return True
    except Exception as e:
        print(f"✗ Error testing functionality: {e}")
        return False

def main():
    print("Testing ProofAtlas installation...")
    print("-" * 40)
    
    success = True
    
    # Test import
    if not test_import():
        success = False
        print("\nPlease install ProofAtlas with: pip install -e .")
        return 1
    
    # Test functionality
    if not test_basic_functionality():
        success = False
    
    print("-" * 40)
    if success:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())