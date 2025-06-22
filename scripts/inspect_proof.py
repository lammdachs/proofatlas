#!/usr/bin/env python3
"""Script to inspect proof files from the command line.

Usage:
    python inspect_proof.py <proof_file> [problem_file]
    
If problem_file is not provided, it will try to extract the problem from the proof.
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from proofatlas.navigator import navigate_proof
from proofatlas.proofs.serialization import load_proof
from proofatlas.core.logic import Problem


def extract_problem_from_proof(proof_path):
    """Extract the initial problem from a proof file."""
    proof = load_proof(proof_path)
    
    # Get initial clauses from the first state
    initial_state = proof.initial_state
    all_clauses = initial_state.processed + initial_state.unprocessed
    
    if not all_clauses:
        print("Warning: No clauses found in initial state")
        return None
    
    # Create a problem from the initial clauses
    problem = Problem(*all_clauses)
    return problem


def load_proof_metadata(proof_path):
    """Load and display proof metadata if available."""
    with open(proof_path, 'r') as f:
        data = json.load(f)
    
    # Check if it's wrapped with metadata
    if isinstance(data, dict) and 'proof' in data:
        print("=" * 60)
        print("PROOF METADATA")
        print("=" * 60)
        if 'description' in data:
            print(f"Description: {data['description']}")
        if 'generator' in data:
            print(f"Generator: {data['generator']}")
        print("=" * 60)
        print()
        return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Inspect theorem proving proofs interactively.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect a proof with its original problem
  python inspect_proof.py proof.json problem.json
  
  # Inspect a proof without the problem file (extracts from proof)
  python inspect_proof.py proof.json
  
  # Inspect a generated proof from tests
  python inspect_proof.py ../.data/proofs/test_examples/basic_loop/modus_ponens.json

Navigation Controls:
  n/→     - Next step
  p/←     - Previous step  
  g       - Go to step number
  f       - First step
  l       - Last step
  h/?     - Show help
  q       - Quit
        """
    )
    
    parser.add_argument(
        'proof_file',
        help='Path to the proof JSON file'
    )
    parser.add_argument(
        'problem_file',
        nargs='?',
        help='Path to the problem JSON file (optional)'
    )
    parser.add_argument(
        '--no-metadata',
        action='store_true',
        help='Skip displaying proof metadata'
    )
    
    args = parser.parse_args()
    
    # Validate proof file exists
    proof_path = Path(args.proof_file)
    if not proof_path.exists():
        print(f"Error: Proof file not found: {proof_path}")
        sys.exit(1)
    
    # Display metadata unless skipped
    if not args.no_metadata:
        has_metadata = load_proof_metadata(proof_path)
        if has_metadata:
            print("Press Enter to start navigation...")
            input()
    
    # Handle problem file
    problem_path = None
    if args.problem_file:
        problem_path = Path(args.problem_file)
        if not problem_path.exists():
            print(f"Error: Problem file not found: {problem_path}")
            sys.exit(1)
    else:
        print("No problem file provided, extracting problem from proof...")
        
        # For wrapped proofs, we need to handle them specially
        with open(proof_path, 'r') as f:
            data = json.load(f)
        
        # If it's wrapped, extract the proof part
        if isinstance(data, dict) and 'proof' in data:
            # Create a temporary file with just the proof
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                json.dump(data['proof'], tmp)
                temp_proof_path = tmp.name
            
            # Use the temporary file for navigation
            proof_path = temp_proof_path
    
    print(f"\nStarting proof navigator...")
    print(f"Proof: {args.proof_file}")
    if problem_path:
        print(f"Problem: {problem_path}")
    print()
    
    try:
        # Navigate the proof
        navigate_proof(str(proof_path), str(problem_path) if problem_path else None)
        print("\nNavigation complete!")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError during navigation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up temporary file if created
        if 'temp_proof_path' in locals():
            import os
            try:
                os.unlink(temp_proof_path)
            except:
                pass


if __name__ == "__main__":
    main()