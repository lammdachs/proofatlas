#!/usr/bin/env python3
"""
Interactive proof stepper for ProofAtlas JSON output.
Allows stepping through the saturation process with arrow keys.
"""

import json
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict
import curses

class ProofStepper:
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
        # First, pre-populate clauses from final_clauses if available
        # This ensures we have clause data even when conclusions are not included
        if 'result' in self.data and 'final_clauses' in self.data['result']:
            for clause in self.data['result']['final_clauses']:
                if 'id' in clause:
                    idx = clause['id']
                    # Store clause without the 'id' field
                    clause_data = {k: v for k, v in clause.items() if k != 'id'}
                    self.clauses[idx] = clause_data
        
        # Initialize with input clauses (overwrite if they exist)
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
                'new_clauses': {i}
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
                    
                    # For GivenClauseSelection, the clause should already exist
                    if idx not in self.clauses:
                        print(f"Warning: GivenClauseSelection for missing clause {idx}, skipping")
                        continue
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
                    elif idx not in self.clauses:
                        # If no conclusion and clause doesn't exist, create a placeholder
                        self.clauses[idx] = {
                            'literals': [{'polarity': True, 'atom': {'predicate': f'[Clause {idx} - conclusion not included in JSON]', 'args': []}}]
                        }
                    
                    # The new clause goes to unprocessed
                    self.unprocessed.add(idx)
                    
                    # Check if this is the last inference with the current given clause
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
                        'clause': self.clauses.get(idx, {'literals': []}),
                        'inference': step['inference'],
                        'processed': self.processed.copy(),
                        'unprocessed': self.unprocessed.copy(),
                        'new_clauses': {idx},
                        'current_given': current_given
                    })
                    
                    # Move the given clause to processed after its last inference
                    if is_last_with_given and current_given is not None:
                        self.processed.add(current_given)
    
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
    
    def run_interactive(self, stdscr):
        """Run the interactive stepper with curses."""
        curses.curs_set(0)  # Hide cursor
        stdscr.clear()
        
        # Color pairs
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Given clause
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # New clause
        curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)    # Processed
        curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLACK)   # Unprocessed
        curses.init_pair(5, curses.COLOR_RED, curses.COLOR_BLACK)     # Empty clause
        
        while True:
            stdscr.clear()
            height, width = stdscr.getmaxyx()
            
            # Header
            header = f"ProofAtlas Saturation Stepper - Step {self.current_step + 1}/{len(self.steps)}"
            stdscr.addstr(0, 0, header, curses.A_BOLD)
            stdscr.addstr(1, 0, "=" * min(len(header), width - 1))
            
            if self.current_step < len(self.steps):
                step = self.steps[self.current_step]
                
                # Step info
                row = 3
                if step['type'] == 'input':
                    stdscr.addstr(row, 0, f"Input clause {step['clause_idx']}:")
                    row += 1
                elif step['type'] == 'given_selection':
                    stdscr.addstr(row, 0, f"GIVEN CLAUSE SELECTED: [{step['clause_idx']}]", curses.color_pair(1) | curses.A_BOLD)
                    row += 1
                else:
                    rule = step['inference']['rule']
                    premises = step['inference']['premises']
                    stdscr.addstr(row, 0, f"Inference: {rule} from {premises} → clause {step['clause_idx']}")
                    row += 1
                
                row += 1
                
                # Show clause sets
                processed = step.get('processed', set())
                unprocessed = step.get('unprocessed', set())
                new_clauses = step.get('new_clauses', set())
                current_given = step.get('current_given')
                
                # Show current given clause first if it exists
                if current_given is not None:
                    stdscr.addstr(row, 0, f"CURRENT GIVEN CLAUSE:", curses.color_pair(1) | curses.A_BOLD)
                    row += 1
                    if current_given in self.clauses:
                        clause_str = self.format_clause(self.clauses[current_given])
                        if len(clause_str) > width - 10:
                            clause_str = clause_str[:width-13] + "..."
                        stdscr.addstr(row, 2, f"[{current_given:3}] ", curses.color_pair(1) | curses.A_BOLD)
                        stdscr.addstr(row, 8, clause_str[:width-10], curses.color_pair(1) | curses.A_BOLD)
                        row += 1
                    row += 1
                
                # Processed clauses
                stdscr.addstr(row, 0, f"Processed ({len(processed)}):", curses.A_BOLD)
                row += 1
                
                display_count = 0
                for idx in sorted(processed)[:10]:  # Show first 10
                    if row >= height - 12:
                        break
                    if idx in self.clauses:
                        clause_str = self.format_clause(self.clauses[idx])
                        if len(clause_str) > width - 10:
                            clause_str = clause_str[:width-13] + "..."
                        
                        color = curses.color_pair(3)
                        
                        stdscr.addstr(row, 2, f"[{idx:3}] ", color)
                        stdscr.addstr(row, 8, clause_str[:width-10], color)
                        row += 1
                        display_count += 1
                
                if len(processed) > display_count:
                    stdscr.addstr(row, 2, f"... and {len(processed) - display_count} more")
                    row += 1
                
                row += 1
                
                # Unprocessed clauses
                stdscr.addstr(row, 0, f"Unprocessed ({len(unprocessed)}):", curses.A_BOLD)
                row += 1
                
                display_count = 0
                for idx in sorted(unprocessed)[:10]:  # Show first 10
                    if row >= height - 8:
                        break
                    if idx in self.clauses:
                        clause_str = self.format_clause(self.clauses[idx])
                        if len(clause_str) > width - 10:
                            clause_str = clause_str[:width-13] + "..."
                        
                        color = curses.color_pair(4)
                        if idx in new_clauses:
                            color = curses.color_pair(2) | curses.A_BOLD
                        
                        # Check for empty clause
                        if not self.clauses[idx]['literals']:
                            color = curses.color_pair(5) | curses.A_BOLD
                        
                        stdscr.addstr(row, 2, f"[{idx:3}] ", color)
                        stdscr.addstr(row, 8, clause_str[:width-10], color)
                        row += 1
                        display_count += 1
                
                if len(unprocessed) > display_count:
                    stdscr.addstr(row, 2, f"... and {len(unprocessed) - display_count} more")
                    row += 1
            
            # Controls at bottom
            controls = "←/→: Navigate | Home: First | End: Last | q: Quit"
            stdscr.addstr(height - 3, 0, "-" * min(len(controls), width - 1))
            stdscr.addstr(height - 2, 0, controls)
            
            # Legend
            legend_row = height - 6
            stdscr.addstr(legend_row, 0, "Legend: ", curses.A_BOLD)
            stdscr.addstr(legend_row, 8, "Given", curses.color_pair(1) | curses.A_BOLD)
            stdscr.addstr(legend_row, 15, "New", curses.color_pair(2) | curses.A_BOLD)
            stdscr.addstr(legend_row, 20, "Processed", curses.color_pair(3))
            stdscr.addstr(legend_row, 31, "Unprocessed", curses.color_pair(4))
            stdscr.addstr(legend_row, 44, "Empty", curses.color_pair(5) | curses.A_BOLD)
            
            stdscr.refresh()
            
            # Handle input
            key = stdscr.getch()
            
            if key == ord('q') or key == ord('Q'):
                break
            elif key == curses.KEY_LEFT:
                if self.current_step > 0:
                    self.current_step -= 1
            elif key == curses.KEY_RIGHT:
                if self.current_step < len(self.steps) - 1:
                    self.current_step += 1
            elif key == curses.KEY_HOME:
                self.current_step = 0
            elif key == curses.KEY_END:
                self.current_step = len(self.steps) - 1
            elif key == ord('j'):  # vi-style down
                if self.current_step < len(self.steps) - 1:
                    self.current_step += 1
            elif key == ord('k'):  # vi-style up
                if self.current_step > 0:
                    self.current_step -= 1
            elif key == ord('g'):  # Jump to step
                stdscr.addstr(height - 1, 0, "Go to step: ")
                curses.echo()
                curses.curs_set(1)
                try:
                    step_str = stdscr.getstr(height - 1, 12, 10).decode('utf-8')
                    step_num = int(step_str) - 1
                    if 0 <= step_num < len(self.steps):
                        self.current_step = step_num
                except:
                    pass
                curses.noecho()
                curses.curs_set(0)
            elif key == ord('/'):  # Search for clause
                stdscr.addstr(height - 1, 0, "Search clause: ")
                curses.echo()
                curses.curs_set(1)
                try:
                    search_str = stdscr.getstr(height - 1, 15, 30).decode('utf-8').lower()
                    # Find next step containing this string
                    for i in range(self.current_step + 1, len(self.steps)):
                        step = self.steps[i]
                        clause_str = self.format_clause(step['clause']).lower()
                        if search_str in clause_str:
                            self.current_step = i
                            break
                except:
                    pass
                curses.noecho()
                curses.curs_set(0)

def main():
    parser = argparse.ArgumentParser(description='Step through ProofAtlas saturation process')
    parser.add_argument('json_file', help='Path to proof JSON file')
    
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
    
    stepper = ProofStepper(proof_data)
    
    if not stepper.steps:
        print("No proof steps found in the JSON file")
        sys.exit(1)
    
    # Run with curses
    try:
        curses.wrapper(stepper.run_interactive)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()