"""Interactive command-line proof navigator."""

import sys
from typing import Optional
import termios
import tty

from proofatlas.core import Problem, load_problem
from proofatlas.proofs import Proof, load_proof
from proofatlas.rules import RuleApplication
from proofatlas.core.logic import Clause, Literal


class ProofNavigator:
    """Interactive proof step navigator."""
    
    def __init__(self, proof: Proof, problem: Optional[Problem] = None):
        """
        Initialize the navigator.
        
        Args:
            proof: The proof to navigate
            problem: Optional original problem for context
        """
        self.proof = proof
        self.problem = problem
        self.current_step = 0
        self.total_steps = len(proof.steps)
        
        # Terminal setup (will be initialized in __enter__)
        self.original_settings = None
    
    def __enter__(self):
        """Enter raw mode for key capture."""
        self.original_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore terminal settings."""
        if self.original_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original_settings)
    
    def clear_screen(self):
        """Clear the terminal screen."""
        # Use ANSI escape codes for better cross-platform support
        sys.stdout.write('\033[2J\033[H')
        sys.stdout.flush()
    
    def get_key(self) -> str:
        """Get a single keypress."""
        key = sys.stdin.read(1)
        
        # Handle escape sequences for arrow keys
        if key == '\x1b':
            seq = sys.stdin.read(2)
            if seq == '[A':
                return 'UP'
            elif seq == '[B':
                return 'DOWN'
            elif seq == '[C':
                return 'RIGHT'
            elif seq == '[D':
                return 'LEFT'
        
        return key
    
    def format_clause(self, clause: Clause, index: Optional[int] = None) -> str:
        """Format a clause for display."""
        if len(clause.literals) == 0:
            clause_str = "⊥ (empty clause)"
        else:
            clause_str = " ∨ ".join(self.format_literal(lit) for lit in clause.literals)
        
        if index is not None:
            return f"[{index:2d}] {clause_str}"
        return clause_str
    
    def format_literal(self, literal: Literal) -> str:
        """Format a literal for display."""
        atom_str = str(literal.predicate)
        if not literal.polarity:
            return f"¬{atom_str}"
        return atom_str
    
    def format_rule_application(self, rule_app: RuleApplication) -> str:
        """Format a RuleApplication for display.
        
        Args:
            rule_app: RuleApplication object
            
        Returns:
            Formatted string representation
        """
        parts = [f"{rule_app.rule_name}:"]
        
        # Format parents
        if len(rule_app.parents) == 1:
            parts.append(f"[{rule_app.parents[0]}]")
        elif len(rule_app.parents) == 2:
            parts.append(f"[{rule_app.parents[0]}] + [{rule_app.parents[1]}]")
        elif rule_app.parents:
            parents_str = " + ".join(f"[{p}]" for p in rule_app.parents)
            parts.append(parents_str)
        
        # Show results
        if rule_app.generated_clauses:
            if len(rule_app.generated_clauses) == 1:
                clause_str = self.format_clause(rule_app.generated_clauses[0])
                parts.append(f"→ {clause_str}")
            else:
                parts.append(f"→ {len(rule_app.generated_clauses)} new clauses")
        
        if rule_app.deleted_clause_indices:
            parts.append(f"(deleted: {rule_app.deleted_clause_indices})")
        
        return " ".join(parts)
    
    def format_inference(self, inference: dict) -> str:
        """Format an inference for display.
        
        Args:
            inference: Dictionary containing inference information
            
        Returns:
            Formatted string representation of the inference
        """
        rule = inference.get("rule", "unknown")
        
        if rule == "resolution":
            parents = inference.get("parents", [])
            result = inference.get("result", "?")
            if len(parents) == 2:
                return f"Resolution: [{parents[0]}] + [{parents[1]}] → {result}"
            else:
                return f"Resolution: {parents} → {result}"
        
        elif rule == "factoring":
            parent = inference.get("parent", "?")
            result = inference.get("result", "?")
            return f"Factoring: [{parent}] → {result}"
        
        elif rule == "subsumption":
            subsumed = inference.get("subsumed", "?")
            subsuming = inference.get("subsuming", "?")
            return f"Subsumption: [{subsuming}] subsumes [{subsumed}]"
        
        elif rule == "equality_resolution":
            parent = inference.get("parent", "?")
            result = inference.get("result", "?")
            return f"Equality Resolution: [{parent}] → {result}"
        
        elif rule == "superposition":
            parents = inference.get("parents", [])
            result = inference.get("result", "?")
            if len(parents) == 2:
                return f"Superposition: [{parents[0]}] + [{parents[1]}] → {result}"
            else:
                return f"Superposition: {parents} → {result}"
        
        else:
            # Generic format for unknown rules
            parents = inference.get("parents", inference.get("parent", []))
            result = inference.get("result", None)
            if isinstance(parents, list) and len(parents) > 0:
                parent_str = " + ".join(f"[{p}]" for p in parents)
            elif parents:
                parent_str = f"[{parents}]"
            else:
                parent_str = "?"
            
            if result:
                return f"{rule.title()}: {parent_str} → {result}"
            else:
                return f"{rule.title()}: {parent_str}"
    
    def display_step(self):
        """Display the current proof step."""
        self.clear_screen()
        
        step = self.proof.steps[self.current_step]
        
        # Build output as a list of lines to ensure proper alignment
        lines = []
        
        # Header
        lines.append("╔" + "═" * 78 + "╗")
        header_text = f" PROOF NAVIGATOR - Step {self.current_step + 1}/{self.total_steps}"
        lines.append("║" + header_text.ljust(78) + "║")
        lines.append("╠" + "═" * 38 + "╦" + "═" * 39 + "╣")
        
        # Column headers
        lines.append("║" + " PROCESSED".ljust(38) + "║" + " UNPROCESSED".ljust(39) + "║")
        lines.append("╠" + "═" * 38 + "╬" + "═" * 39 + "╣")
        
        # Prepare clause lists
        processed = step.state.processed
        unprocessed = step.state.unprocessed
        
        # Get selected clause info - for displaying which clause will be selected next
        # (not which clause was selected to get to this state)
        selected_idx = step.selected_clause
        selected_clause_str = None
        if selected_idx is not None:
            if selected_idx < len(step.state.unprocessed):
                selected_clause_str = self.format_clause(step.state.unprocessed[selected_idx])
        
        # Calculate max rows needed
        max_rows = max(len(processed), len(unprocessed), 1)
        
        # Display clauses in columns
        for i in range(max_rows):
            left_content = ""
            right_content = ""
            
            # Processed column
            if i < len(processed):
                clause_str = self.format_clause(processed[i])
                left_content = f" [{i:2d}] {clause_str}"
                if len(left_content) > 38:
                    left_content = left_content[:35] + "..."
            
            # Unprocessed column
            if i < len(unprocessed):
                clause_str = self.format_clause(unprocessed[i])
                if i == selected_idx:
                    right_content = f"→[{i:2d}] {clause_str}"
                else:
                    right_content = f" [{i:2d}] {clause_str}"
                if len(right_content) > 38:
                    right_content = right_content[:35] + "..."
            
            lines.append(f"║{left_content:<38}║{right_content:<39}║")
        
        # Footer for columns
        lines.append("╠" + "═" * 38 + "╩" + "═" * 39 + "╣")
        
        # Status information
        lines.append("║" + " STATUS:".ljust(78) + "║")
        
        # Show selected clause info
        if selected_idx is not None:
            if selected_clause_str:
                status = f"Given clause [{selected_idx}]: {selected_clause_str}"
                if len(status) > 76:
                    status = status[:73] + "..."
                lines.append("║" + f"   {status}".ljust(78) + "║")
            else:
                lines.append("║" + f"   Given clause: {selected_idx} (out of range)".ljust(78) + "║")
        else:
            lines.append("║" + "   No clause selected (final state)".ljust(78) + "║")
        
        # Show applied rules
        if step.applied_rules:
            lines.append("║" + "   Rules applied:".ljust(78) + "║")
            for rule_app in step.applied_rules:
                rule_line = self.format_rule_application(rule_app)
                if len(rule_line) > 74:
                    rule_line = rule_line[:71] + "..."
                lines.append("║" + f"     {rule_line}".ljust(78) + "║")
        
        # Show other metadata
        if step.metadata:
            for key, value in step.metadata.items():
                if key == "given_clause":
                    # Skip given_clause in metadata since it's already shown in status
                    continue
                elif key == "inferences" and isinstance(value, list):
                    # Handle legacy format
                    lines.append("║" + "   Inferences performed:".ljust(78) + "║")
                    for inference in value:
                        inf_line = self.format_inference(inference)
                        if len(inf_line) > 74:
                            inf_line = inf_line[:71] + "..."
                        lines.append("║" + f"     {inf_line}".ljust(78) + "║")
                else:
                    meta_line = f"  {key}: {value}"
                    if len(meta_line) > 76:
                        meta_line = meta_line[:73] + "..."
                    lines.append("║" + f" {meta_line}".ljust(78) + "║")
        
        # Check for contradiction
        if any(len(c.literals) == 0 for c in step.state.processed + step.state.unprocessed):
            lines.append("║" + " ".ljust(78) + "║")
            lines.append("║" + "   *** CONTRADICTION FOUND! ***".ljust(78) + "║")
        
        # Navigation help
        lines.append("╠" + "═" * 78 + "╣")
        nav_text = " Navigation: ← Previous │ → Next │ ↑ First │ ↓ Last │ q Quit │ h Help"
        lines.append("║" + nav_text.ljust(78) + "║")
        lines.append("╚" + "═" * 78 + "╝")
        
        # Print all lines at once
        # When in raw mode, we need to handle line endings differently
        for line in lines:
            sys.stdout.write(line + '\r\n')
        sys.stdout.flush()
    
    def display_help(self):
        """Display help screen."""
        self.clear_screen()
        
        lines = []
        lines.append("╔" + "═" * 78 + "╗")
        lines.append("║" + " PROOF NAVIGATOR HELP".ljust(78) + "║")
        lines.append("╠" + "═" * 78 + "╣")
        lines.append("║" + " ".ljust(78) + "║")
        lines.append("║" + " NAVIGATION KEYS:".ljust(78) + "║")
        lines.append("║" + "   → or Right Arrow : Next step".ljust(78) + "║")
        lines.append("║" + "   ← or Left Arrow  : Previous step".ljust(78) + "║") 
        lines.append("║" + "   ↑ or Up Arrow    : First step".ljust(78) + "║")
        lines.append("║" + "   ↓ or Down Arrow  : Last step".ljust(78) + "║")
        lines.append("║" + "   q                : Quit".ljust(78) + "║")
        lines.append("║" + "   h                : This help screen".ljust(78) + "║")
        lines.append("║" + " ".ljust(78) + "║")
        lines.append("║" + " DISPLAY ELEMENTS:".ljust(78) + "║")
        lines.append("║" + "   [n]              : Clause index".ljust(78) + "║")
        lines.append("║" + "   ¬                : Negation".ljust(78) + "║")
        lines.append("║" + "   ∨                : Disjunction (OR)".ljust(78) + "║")
        lines.append("║" + "   ⊥                : Empty clause (contradiction)".ljust(78) + "║")
        lines.append("║" + "   →                : Currently selected clause".ljust(78) + "║")
        lines.append("║" + " ".ljust(78) + "║")
        lines.append("║" + " PROOF STRUCTURE:".ljust(78) + "║")
        lines.append("║" + "   - The display shows two columns: PROCESSED and UNPROCESSED clauses".ljust(78) + "║")
        lines.append("║" + "   - Selected clause (marked with →) will be processed in the next step".ljust(78) + "║")
        lines.append("║" + "   - Metadata shows additional information (rules, parent clauses, etc.)".ljust(78) + "║")
        lines.append("║" + "   - Empty clause (⊥) indicates a contradiction has been found".ljust(78) + "║")
        lines.append("║" + " ".ljust(78) + "║")
        lines.append("╠" + "═" * 78 + "╣")
        lines.append("║" + " Press any key to continue...".ljust(78) + "║")
        lines.append("╚" + "═" * 78 + "╝")
        
        # When in raw mode, we need to handle line endings differently
        for line in lines:
            sys.stdout.write(line + '\r\n')
        sys.stdout.flush()
        self.get_key()
    
    def run(self):
        """Run the interactive navigation loop."""
        self.display_step()
        
        while True:
            key = self.get_key()
            
            if key == 'q':
                break
            elif key == 'h':
                self.display_help()
                self.display_step()
            elif key in ['RIGHT', 'd']:
                if self.current_step < self.total_steps - 1:
                    self.current_step += 1
                    self.display_step()
            elif key in ['LEFT', 'a']:
                if self.current_step > 0:
                    self.current_step -= 1
                    self.display_step()
            elif key in ['UP', 'w']:
                self.current_step = 0
                self.display_step()
            elif key in ['DOWN', 's']:
                self.current_step = self.total_steps - 1
                self.display_step()
        
        self.clear_screen()


def navigate_proof(proof_path: str, problem_path: Optional[str] = None):
    """
    Navigate a proof from a file.
    
    Args:
        proof_path: Path to the proof JSON file
        problem_path: Optional path to the problem JSON file
    """
    proof = load_proof(proof_path)
    problem = load_problem(problem_path) if problem_path else None
    
    with ProofNavigator(proof, problem) as navigator:
        navigator.run()


def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Interactive proof navigator for ProofAtlas"
    )
    parser.add_argument(
        "proof",
        help="Path to proof JSON file"
    )
    parser.add_argument(
        "-p", "--problem",
        help="Path to problem JSON file (optional, for context)"
    )
    
    args = parser.parse_args()
    
    try:
        navigate_proof(args.proof, args.problem)
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()