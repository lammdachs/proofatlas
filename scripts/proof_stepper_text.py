#!/usr/bin/env python3
"""
Text-based proof stepper for ProofAtlas JSON output.
Allows stepping through the saturation process with simple commands.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict

class ProofStepperText:
    def __init__(self, proof_data: Dict):
        self.data = proof_data
        self.steps = []
        self.current_step = 0
        self.clauses = {}
        self.processed = set()
        self.unprocessed = set()
        self._build_steps()
    
    def _build_steps(self):
        """Build the sequence of saturation steps from proof data."""
        # Initialize with input clauses
        for i, clause in enumerate(self.data['initial_clauses']):
            self.clauses[i] = clause
            self.unprocessed.add(i)
            self.steps.append({
                'type': 'input',
                'clause_idx': i,
                'clause': clause,
                'inference': {'rule': 'Input', 'premises': []},
                'processed': set(),
                'unprocessed': self.unprocessed.copy(),
                'new_clauses': {i},
                'current_given': None
            })
        
        # Process proof steps
        if 'result' in self.data:
            proof_steps = []
            if self.data['result']['result'] == 'Proof':
                proof_steps = self.data['result']['proof']['steps']
            elif 'proof_steps' in self.data['result']:
                proof_steps = self.data['result']['proof_steps']
            
            current_given = None
            
            for i, step in enumerate(proof_steps):
                if step['inference']['rule'] == 'Input':
                    continue  # Already handled
                
                idx = step['clause_idx']
                
                # Handle GivenClauseSelection steps
                if step['inference']['rule'] == 'GivenClauseSelection':
                    # Move the previous given clause to processed if there was one
                    if current_given is not None:
                        self.processed.add(current_given)
                    
                    # This is a given clause selection - remove from unprocessed
                    if idx in self.unprocessed:
                        self.unprocessed.discard(idx)
                    
                    # For GivenClauseSelection, the clause was already stored
                    clause = self.clauses[idx]
                    
                    # Track this as the current given clause
                    current_given = idx
                    
                    self.steps.append({
                        'type': 'given_selection',
                        'clause_idx': idx,
                        'clause': clause,
                        'inference': step['inference'],
                        'processed': self.processed.copy(),
                        'unprocessed': self.unprocessed.copy(),
                        'new_clauses': set(),
                        'current_given': current_given
                    })
                else:
                    # Regular inference step - store the new clause
                    if 'conclusion' in step['inference'] and step['inference']['conclusion']:
                        self.clauses[idx] = step['inference']['conclusion']
                    
                    # The new clause goes to unprocessed
                    self.unprocessed.add(idx)
                    
                    # Check if this is the last inference with the current given clause
                    # by looking ahead to see if the next step is a new given selection
                    is_last_with_given = False
                    if i + 1 < len(proof_steps):
                        next_step = proof_steps[i + 1]
                        if next_step['inference']['rule'] == 'GivenClauseSelection':
                            is_last_with_given = True
                    else:
                        is_last_with_given = True  # Last step in proof
                    
                    self.steps.append({
                        'type': 'inference',
                        'clause_idx': idx,
                        'clause': self.clauses[idx],
                        'inference': step['inference'],
                        'processed': self.processed.copy(),
                        'unprocessed': self.unprocessed.copy(),
                        'new_clauses': {idx},
                        'current_given': current_given
                    })
                    
                    # Move the given clause to processed after its last inference
                    if is_last_with_given and current_given is not None:
                        self.processed.add(current_given)
                        current_given = None
    
    def format_term(self, term: Dict) -> str:
        """Format a term for display."""
        if term['type'] == 'Variable':
            return term['name']
        elif term['type'] == 'Constant':
            return term['name']
        elif term['type'] == 'Function':
            args = ', '.join(self.format_term(arg) for arg in term['args'])
            return f"{term['name']}({args})"
        return str(term)
    
    def format_atom(self, atom: Dict) -> str:
        """Format an atom for display."""
        if atom['args']:
            args = ', '.join(self.format_term(arg) for arg in atom['args'])
            return f"{atom['predicate']}({args})"
        return atom['predicate']
    
    def format_literal(self, literal: Dict) -> str:
        """Format a literal for display."""
        atom_str = self.format_atom(literal['atom'])
        if literal['polarity']:
            return atom_str
        else:
            return f"~{atom_str}"
    
    def format_clause(self, clause: Dict) -> str:
        """Format a clause for display."""
        if not clause['literals']:
            return "⊥"  # Empty clause
        literals = [self.format_literal(lit) for lit in clause['literals']]
        return " ∨ ".join(literals)
    
    def display_step(self):
        """Display the current step."""
        print("\n" + "=" * 80)
        print(f"Step {self.current_step + 1}/{len(self.steps)}")
        print("=" * 80)
        
        if self.current_step >= len(self.steps):
            print("No more steps")
            return
        
        step = self.steps[self.current_step]
        
        # Step information
        if step['type'] == 'input':
            print(f"\n📥 INPUT CLAUSE {step['clause_idx']}:")
            print(f"   {self.format_clause(step['clause'])}")
        elif step['type'] == 'given_selection':
            print(f"\n🎯 GIVEN CLAUSE SELECTED: [{step['clause_idx']}]")
            print(f"   {self.format_clause(step['clause'])}")
        else:
            rule = step['inference']['rule']
            premises = step['inference']['premises']
            print(f"\n⚡ INFERENCE: {rule}")
            print(f"   From clauses: {premises}")
            print(f"   New clause [{step['clause_idx']}]: {self.format_clause(step['clause'])}")
        
        # Show current given clause if any
        current_given = step.get('current_given')
        if current_given is not None:
            print(f"\n💡 CURRENT GIVEN CLAUSE: [{current_given}]")
            if current_given in self.clauses:
                print(f"   {self.format_clause(self.clauses[current_given])}")
        
        # Check for empty clause
        if not step['clause']['literals']:
            print("\n🏆 EMPTY CLAUSE DERIVED - PROOF FOUND!")
        
        # Statistics
        processed = step.get('processed', set())
        unprocessed = step.get('unprocessed', set())
        new_clauses = step.get('new_clauses', set())
        
        print(f"\n📊 Statistics:")
        print(f"   Processed: {len(processed)} clauses")
        print(f"   Unprocessed: {len(unprocessed)} clauses")
        print(f"   Total: {len(processed) + len(unprocessed)} clauses")
        
        # Show some processed clauses
        if processed:
            print(f"\n✅ Processed clauses (showing last 5):")
            for idx in sorted(processed)[-5:]:
                if idx in self.clauses:
                    clause_str = self.format_clause(self.clauses[idx])
                    if len(clause_str) > 70:
                        clause_str = clause_str[:67] + "..."
                    print(f"   [{idx:3}] {clause_str}")
        
        # Show some unprocessed clauses  
        if unprocessed:
            print(f"\n⏳ Unprocessed clauses (showing first 5):")
            for idx in sorted(unprocessed)[:5]:
                if idx in self.clauses:
                    clause_str = self.format_clause(self.clauses[idx])
                    if len(clause_str) > 70:
                        clause_str = clause_str[:67] + "..."
                    marker = "🆕" if idx in new_clauses else "  "
                    print(f"   {marker} [{idx:3}] {clause_str}")
            
            if len(unprocessed) > 5:
                print(f"   ... and {len(unprocessed) - 5} more")
    
    def run_interactive(self):
        """Run the interactive stepper."""
        print("\n" + "=" * 80)
        print("PROOFATLAS SATURATION STEPPER")
        print("=" * 80)
        print(f"Problem: {self.data['problem_file']}")
        print(f"Total steps: {len(self.steps)}")
        print("\nCommands:")
        print("  n/↵  - Next step")
        print("  p    - Previous step")
        print("  g N  - Go to step N")
        print("  f    - First step")
        print("  l    - Last step")
        print("  s    - Search for clause")
        print("  h    - Show this help")
        print("  q    - Quit")
        
        self.display_step()
        
        while True:
            try:
                cmd = input("\nstepper> ").strip().lower()
                
                if cmd == 'q':
                    break
                elif cmd == 'n' or cmd == '':
                    if self.current_step < len(self.steps) - 1:
                        self.current_step += 1
                        self.display_step()
                    else:
                        print("Already at last step")
                elif cmd == 'p':
                    if self.current_step > 0:
                        self.current_step -= 1
                        self.display_step()
                    else:
                        print("Already at first step")
                elif cmd == 'f':
                    self.current_step = 0
                    self.display_step()
                elif cmd == 'l':
                    self.current_step = len(self.steps) - 1
                    self.display_step()
                elif cmd.startswith('g '):
                    try:
                        step_num = int(cmd[2:]) - 1
                        if 0 <= step_num < len(self.steps):
                            self.current_step = step_num
                            self.display_step()
                        else:
                            print(f"Step must be between 1 and {len(self.steps)}")
                    except ValueError:
                        print("Invalid step number")
                elif cmd == 's':
                    search = input("Search for: ").strip().lower()
                    found = False
                    for i in range(self.current_step + 1, len(self.steps)):
                        clause_str = self.format_clause(self.steps[i]['clause']).lower()
                        if search in clause_str:
                            self.current_step = i
                            self.display_step()
                            found = True
                            break
                    if not found:
                        print("Not found in remaining steps")
                elif cmd == 'h':
                    print("\nCommands:")
                    print("  n/↵  - Next step")
                    print("  p    - Previous step")
                    print("  g N  - Go to step N")
                    print("  f    - First step")
                    print("  l    - Last step")
                    print("  s    - Search for clause")
                    print("  h    - Show this help")
                    print("  q    - Quit")
                else:
                    print("Unknown command. Type 'h' for help.")
                    
            except KeyboardInterrupt:
                print("\nUse 'q' to quit")
            except EOFError:
                break

def main():
    parser = argparse.ArgumentParser(description='Step through ProofAtlas saturation process')
    parser.add_argument('json_file', help='Path to proof JSON file')
    parser.add_argument('-s', '--step', type=int, help='Start at specific step')
    
    args = parser.parse_args()
    
    # Load JSON file
    try:
        with open(args.json_file, 'r') as f:
            proof_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{args.json_file}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{args.json_file}': {e}")
        sys.exit(1)
    
    stepper = ProofStepperText(proof_data)
    
    if not stepper.steps:
        print("No proof steps found in the JSON file")
        sys.exit(1)
    
    if args.step:
        stepper.current_step = max(0, min(args.step - 1, len(stepper.steps) - 1))
    
    stepper.run_interactive()

if __name__ == '__main__':
    main()