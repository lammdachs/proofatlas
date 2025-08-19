#!/usr/bin/env python3
"""
Interactive proof viewer for ProofAtlas JSON output.
Provides visualization and analysis of proof attempts.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

class ProofViewer:
    def __init__(self, proof_data: Dict):
        self.data = proof_data
        self.clauses = {}  # Map clause index to clause data
        self._build_clause_index()
    
    def _build_clause_index(self):
        """Build index of all clauses from initial and proof steps."""
        # Add initial clauses
        for i, clause in enumerate(self.data['initial_clauses']):
            self.clauses[i] = clause
        
        # Add clauses from proof steps if available
        if 'result' in self.data:
            if self.data['result']['result'] == 'Proof':
                for step in self.data['result']['proof']['steps']:
                    idx = step['clause_idx']
                    self.clauses[idx] = step['inference']['conclusion']
            elif 'proof_steps' in self.data['result']:
                for step in self.data['result']['proof_steps']:
                    idx = step['clause_idx']
                    self.clauses[idx] = step['inference']['conclusion']
    
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
    
    def print_header(self):
        """Print header information."""
        print("=" * 80)
        print("PROOFATLAS PROOF VIEWER")
        print("=" * 80)
        print(f"Problem: {self.data['problem_file']}")
        print(f"Initial clauses: {len(self.data['initial_clauses'])}")
        
        config = self.data['config']
        print(f"\nConfiguration:")
        print(f"  Max clauses: {config['max_clauses']}")
        print(f"  Timeout: {config['timeout_seconds']}s")
        print(f"  Superposition: {config['use_superposition']}")
        print(f"  Literal selection: {config['literal_selection']}")
        
        stats = self.data.get('statistics', {})
        print(f"\nStatistics:")
        print(f"  Time elapsed: {stats.get('time_elapsed_seconds', 0):.3f}s")
        print(f"  Clauses generated: {stats.get('clauses_generated', 'N/A')}")
        print(f"  Clauses processed: {stats.get('clauses_processed', 'N/A')}")
        
        result = self.data['result']
        print(f"\nResult: {result['result']}")
        if result['result'] == 'Proof':
            print(f"  Proof found with {len(result['proof']['steps'])} steps")
            print(f"  Empty clause at index: {result['proof']['empty_clause_idx']}")
        elif result['result'] == 'Timeout':
            print(f"  Timeout after {result['time_seconds']:.3f}s")
        elif result['result'] == 'ResourceLimit':
            print(f"  Resource limit: {result.get('reason', 'Unknown')}")
    
    def show_initial_clauses(self):
        """Display initial clauses."""
        print("\nInitial Clauses:")
        print("-" * 40)
        for i, clause in enumerate(self.data['initial_clauses']):
            clause_str = self.format_clause(clause)
            print(f"[{i:3}] {clause_str}")
    
    def show_clause_generation_summary(self):
        """Show summary of clause generation for non-proof results."""
        result = self.data['result']
        if 'final_clauses' in result:
            print(f"\nFinal clause count: {len(result['final_clauses'])}")
            
            # Analyze clause sizes
            sizes = defaultdict(int)
            for clause in result['final_clauses']:
                size = len(clause['literals'])
                sizes[size] += 1
            
            print("\nClause size distribution:")
            for size in sorted(sizes.keys()):
                print(f"  Size {size}: {sizes[size]} clauses")
            
            # Show some of the final clauses
            print("\nSample of final clauses (last 10):")
            for clause in result['final_clauses'][-10:]:
                if clause.get('id') is not None:
                    print(f"  [{clause['id']:4}] {self.format_clause(clause)}")
                else:
                    print(f"  [    ] {self.format_clause(clause)}")
        
        if 'proof_steps' in result:
            print(f"\nProof steps generated: {len(result['proof_steps'])}")
            
            # Analyze inference rules used
            rule_counts = defaultdict(int)
            for step in result['proof_steps']:
                rule_counts[step['inference']['rule']] += 1
            
            print("\nInference rules used:")
            for rule, count in sorted(rule_counts.items()):
                print(f"  {rule}: {count}")
    
    def show_proof_tree(self):
        """Display proof in tree format."""
        if self.data['result']['result'] != 'Proof':
            print("\nNo proof found - showing clause generation instead.")
            self.show_clause_generation_summary()
            return
        
        proof = self.data['result']['proof']
        print("\nProof Tree:")
        print("-" * 40)
        
        # Build parent-child relationships
        children = defaultdict(list)
        for step in proof['steps']:
            for parent in step['inference']['premises']:
                children[parent].append(step['clause_idx'])
        
        # Find the empty clause and work backwards
        empty_idx = proof['empty_clause_idx']
        
        def print_tree(idx: int, depth: int = 0, visited: set = None):
            if visited is None:
                visited = set()
            if idx in visited:
                print("  " * depth + f"[{idx}] <cycle detected>")
                return
            visited.add(idx)
            
            # Find the step for this clause
            step = None
            for s in proof['steps']:
                if s['clause_idx'] == idx:
                    step = s
                    break
            
            if idx in self.clauses:
                clause_str = self.format_clause(self.clauses[idx])
            else:
                clause_str = "???"
            
            if step:
                rule = step['inference']['rule']
                parents = step['inference']['premises']
                print("  " * depth + f"[{idx}] {rule} from {parents}: {clause_str}")
                
                # Print parents
                for parent in parents:
                    print_tree(parent, depth + 1, visited.copy())
            else:
                # Input clause
                print("  " * depth + f"[{idx}] Input: {clause_str}")
        
        print_tree(empty_idx)
    
    def show_proof_linear(self):
        """Display proof in linear format."""
        if self.data['result']['result'] != 'Proof':
            print("\nNo proof found.")
            return
        
        proof = self.data['result']['proof']
        print("\nLinear Proof:")
        print("-" * 40)
        
        for step in proof['steps']:
            idx = step['clause_idx']
            rule = step['inference']['rule']
            premises = step['inference']['premises']
            
            if idx in self.clauses:
                clause_str = self.format_clause(self.clauses[idx])
            else:
                clause_str = self.format_clause(step['inference']['conclusion'])
            
            if rule == 'Input':
                print(f"[{idx:3}] Input: {clause_str}")
            else:
                print(f"[{idx:3}] {rule} from {premises}: {clause_str}")
    
    def analyze_proof(self):
        """Analyze proof characteristics."""
        if self.data['result']['result'] != 'Proof':
            print("\nNo proof to analyze.")
            return
        
        proof = self.data['result']['proof']
        print("\nProof Analysis:")
        print("-" * 40)
        
        # Count inference rules used
        rule_counts = defaultdict(int)
        for step in proof['steps']:
            rule_counts[step['inference']['rule']] += 1
        
        print("Inference rules used:")
        for rule, count in sorted(rule_counts.items()):
            print(f"  {rule}: {count}")
        
        # Analyze clause sizes
        clause_sizes = []
        for step in proof['steps']:
            if 'conclusion' in step['inference']:
                size = len(step['inference']['conclusion']['literals'])
                clause_sizes.append(size)
        
        if clause_sizes:
            print(f"\nClause sizes:")
            print(f"  Average: {sum(clause_sizes) / len(clause_sizes):.1f}")
            print(f"  Max: {max(clause_sizes)}")
            print(f"  Min: {min(clause_sizes)}")
        
        # Find most used clauses
        clause_usage = defaultdict(int)
        for step in proof['steps']:
            for premise in step['inference']['premises']:
                clause_usage[premise] += 1
        
        if clause_usage:
            print(f"\nMost used clauses:")
            for idx, count in sorted(clause_usage.items(), key=lambda x: x[1], reverse=True)[:5]:
                if idx in self.clauses:
                    clause_str = self.format_clause(self.clauses[idx])
                    print(f"  [{idx}] used {count} times: {clause_str}")
    
    def interactive_mode(self):
        """Interactive exploration of the proof."""
        print("\nInteractive Mode Commands:")
        print("  i - Show initial clauses")
        print("  p - Show linear proof")
        print("  t - Show proof tree")
        print("  a - Analyze proof")
        print("  g - Show clause generation summary (for non-proofs)")
        print("  c <n> - Show clause n with details")
        print("  s <n> - Show step n")
        print("  f - Show final clauses (for non-proofs)")
        print("  q - Quit")
        
        while True:
            try:
                cmd = input("\nproof> ").strip().lower()
                
                if cmd == 'q':
                    break
                elif cmd == 'i':
                    self.show_initial_clauses()
                elif cmd == 'p':
                    self.show_proof_linear()
                elif cmd == 't':
                    self.show_proof_tree()
                elif cmd == 'a':
                    self.analyze_proof()
                elif cmd == 'g':
                    self.show_clause_generation_summary()
                elif cmd == 'f':
                    if 'final_clauses' in self.data['result']:
                        print(f"\nFinal clauses ({len(self.data['result']['final_clauses'])} total):")
                        for i, clause in enumerate(self.data['result']['final_clauses'][:50]):
                            print(f"  [{clause.get('id', i):4}] {self.format_clause(clause)}")
                        if len(self.data['result']['final_clauses']) > 50:
                            print(f"  ... and {len(self.data['result']['final_clauses']) - 50} more")
                    else:
                        print("No final clauses available")
                elif cmd.startswith('c '):
                    try:
                        idx = int(cmd.split()[1])
                        if idx in self.clauses:
                            print(f"\nClause {idx}:")
                            print(f"  {self.format_clause(self.clauses[idx])}")
                            print(f"  Literals: {len(self.clauses[idx]['literals'])}")
                            print(f"  Tautology: {self.clauses[idx].get('is_tautology', False)}")
                        else:
                            print(f"Clause {idx} not found")
                    except (ValueError, IndexError):
                        print("Usage: c <clause_number>")
                elif cmd.startswith('s '):
                    try:
                        n = int(cmd.split()[1])
                        if self.data['result']['result'] == 'Proof':
                            steps = self.data['result']['proof']['steps']
                            if 0 <= n < len(steps):
                                step = steps[n]
                                print(f"\nStep {n}:")
                                print(f"  Clause index: {step['clause_idx']}")
                                print(f"  Rule: {step['inference']['rule']}")
                                print(f"  Premises: {step['inference']['premises']}")
                                print(f"  Conclusion: {self.format_clause(step['inference']['conclusion'])}")
                            else:
                                print(f"Step {n} out of range (0-{len(steps)-1})")
                        else:
                            print("No proof available")
                    except (ValueError, IndexError):
                        print("Usage: s <step_number>")
                else:
                    print("Unknown command. Type 'q' to quit.")
                    
            except KeyboardInterrupt:
                print("\nUse 'q' to quit")
            except EOFError:
                break

def main():
    parser = argparse.ArgumentParser(description='View ProofAtlas proof JSON files')
    parser.add_argument('json_file', help='Path to proof JSON file')
    parser.add_argument('-i', '--interactive', action='store_true', 
                       help='Enter interactive mode')
    parser.add_argument('-p', '--proof', action='store_true',
                       help='Show linear proof')
    parser.add_argument('-t', '--tree', action='store_true',
                       help='Show proof tree')
    parser.add_argument('-a', '--analyze', action='store_true',
                       help='Analyze proof')
    parser.add_argument('-c', '--clauses', action='store_true',
                       help='Show initial clauses')
    
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
    
    viewer = ProofViewer(proof_data)
    viewer.print_header()
    
    if args.clauses:
        viewer.show_initial_clauses()
    
    if args.proof:
        viewer.show_proof_linear()
    
    if args.tree:
        viewer.show_proof_tree()
    
    if args.analyze:
        viewer.analyze_proof()
    
    if args.interactive or not any([args.proof, args.tree, args.analyze, args.clauses]):
        viewer.interactive_mode()

if __name__ == '__main__':
    main()