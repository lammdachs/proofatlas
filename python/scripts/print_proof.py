#!/usr/bin/env python3
"""Script to print proof files in a readable format.

Usage:
    python print_proof.py <proof_file> [--step N]
    
Options:
    --step N    Show only step N (0-indexed)
    --summary   Show only a summary of the proof
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from proofatlas.proofs.serialization import load_proof


def print_separator(char="-", width=80):
    """Print a separator line."""
    print(char * width)


def print_clause(clause, prefix=""):
    """Print a clause with optional prefix."""
    print(f"{prefix}{clause}")


def print_proof_summary(proof):
    """Print a summary of the proof."""
    print_separator("=")
    print("PROOF SUMMARY")
    print_separator("=")
    
    print(f"Total steps: {len(proof.steps)}")
    print(f"Inference steps: {proof.length}")
    
    # Count rule applications
    rule_counts = {}
    total_generated = 0
    for step in proof.steps:
        for rule_app in step.applied_rules:
            rule_counts[rule_app.rule_name] = rule_counts.get(rule_app.rule_name, 0) + 1
            total_generated += len(rule_app.generated_clauses)
    
    print(f"\nRule applications:")
    for rule, count in sorted(rule_counts.items()):
        print(f"  {rule}: {count}")
    
    print(f"\nTotal clauses generated: {total_generated}")
    
    # Check for contradiction
    final_state = proof.final_state
    all_clauses = final_state.processed + final_state.unprocessed
    has_empty = any(len(c.literals) == 0 for c in all_clauses)
    
    print(f"\nContradiction found: {'Yes' if has_empty else 'No'}")
    print(f"Final processed clauses: {len(final_state.processed)}")
    print(f"Final unprocessed clauses: {len(final_state.unprocessed)}")


def print_proof_step(proof, step_num):
    """Print a single proof step in detail."""
    if step_num < 0 or step_num >= len(proof.steps):
        print(f"Error: Step {step_num} out of range (0-{len(proof.steps)-1})")
        return
    
    step = proof.steps[step_num]
    
    print_separator("=")
    print(f"STEP {step_num}")
    print_separator("=")
    
    # Selected clause
    if step.selected_clause is not None:
        print(f"Selected clause index: {step.selected_clause}")
        if 'selected_clause_obj' in step.metadata:
            print(f"Selected clause: {step.metadata['selected_clause_obj']}")
    else:
        print("No clause selected (final state)")
    
    # Applied rules
    if step.applied_rules:
        print(f"\nApplied rules ({len(step.applied_rules)}):")
        for i, rule_app in enumerate(step.applied_rules):
            print(f"\n  {i+1}. {rule_app.rule_name}")
            # Check metadata for given clause usage
            if rule_app.metadata.get('with_given_clause'):
                print(f"     Parents: {rule_app.parents} + given clause")
            elif rule_app.metadata.get('on_given_clause'):
                print(f"     Applied to: given clause")
            else:
                print(f"     Parents: {rule_app.parents}")
            if rule_app.generated_clauses:
                print(f"     Generated clauses:")
                for clause in rule_app.generated_clauses:
                    print_clause(clause, "       ")
    
    # State
    print("\nPROCESSED CLAUSES:")
    print_separator("-", 40)
    if step.state.processed:
        for i, clause in enumerate(step.state.processed):
            print(f"  {i}: {clause}")
    else:
        print("  (empty)")
    
    print("\nUNPROCESSED CLAUSES:")
    print_separator("-", 40)
    if step.state.unprocessed:
        for i, clause in enumerate(step.state.unprocessed):
            print(f"  {i}: {clause}")
    else:
        print("  (empty)")
    
    # Metadata
    if step.metadata:
        relevant_metadata = {k: v for k, v in step.metadata.items() 
                           if k not in ['selected_clause_obj', 'new_clauses']}
        if relevant_metadata:
            print("\nMetadata:")
            for key, value in relevant_metadata.items():
                print(f"  {key}: {value}")


def print_full_proof(proof):
    """Print the entire proof step by step."""
    print_proof_summary(proof)
    
    for i in range(len(proof.steps)):
        print()
        print_proof_step(proof, i)
        if i < len(proof.steps) - 1:
            print("\n" + "=" * 80 + "\n")


def load_proof_with_metadata(proof_path):
    """Load proof, handling wrapped format with metadata."""
    with open(proof_path, 'r') as f:
        data = json.load(f)
    
    # Check if it's wrapped with metadata
    if isinstance(data, dict) and 'proof' in data:
        # Display metadata
        print_separator("=")
        print("FILE METADATA")
        print_separator("=")
        if 'description' in data:
            print(f"Description: {data['description']}")
        if 'generator' in data:
            print(f"Generator: {data['generator']}")
        print()
        
        # Save proof data to temporary file to load with decoder
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(data['proof'], tmp)
            temp_path = tmp.name
        
        proof = load_proof(temp_path)
        
        # Clean up
        import os
        os.unlink(temp_path)
        
        return proof
    else:
        return load_proof(proof_path)


def main():
    parser = argparse.ArgumentParser(
        description="Print theorem proving proofs in readable format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Print entire proof
  python print_proof.py proof.json
  
  # Print only step 2
  python print_proof.py proof.json --step 2
  
  # Print only summary
  python print_proof.py proof.json --summary
        """
    )
    
    parser.add_argument(
        'proof_file',
        help='Path to the proof JSON file'
    )
    parser.add_argument(
        '--step',
        type=int,
        metavar='N',
        help='Show only step N (0-indexed)'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Show only proof summary'
    )
    
    args = parser.parse_args()
    
    # Validate proof file exists
    proof_path = Path(args.proof_file)
    if not proof_path.exists():
        print(f"Error: Proof file not found: {proof_path}")
        sys.exit(1)
    
    try:
        # Load proof
        proof = load_proof_with_metadata(proof_path)
        
        # Display based on options
        if args.summary:
            print_proof_summary(proof)
        elif args.step is not None:
            print_proof_step(proof, args.step)
        else:
            print_full_proof(proof)
            
    except Exception as e:
        print(f"Error loading proof: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()