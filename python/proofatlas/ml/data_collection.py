"""Collect training data from TPTP problems for clause selection learning"""

import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
from tqdm import tqdm

from proofatlas import ProofState
from .graph_utils import to_torch_tensors


class TrainingDataCollector:
    """Collect training data from successful proofs"""

    def __init__(
        self,
        prove_binary: str = "target/release/prove",
        timeout: int = 60,
    ):
        """
        Args:
            prove_binary: Path to prove binary
            timeout: Timeout in seconds per problem
        """
        self.prove_binary = prove_binary
        self.timeout = timeout

    def collect_from_problem(
        self, problem_file: Path
    ) -> Optional[Dict[str, Any]]:
        """
        Run prover on a problem and extract training data if proof found.

        Args:
            problem_file: Path to TPTP problem file

        Returns:
            Dictionary with training examples and metadata, or None if no proof
        """
        # Run prover with JSON output
        try:
            result = subprocess.run(
                [self.prove_binary, str(problem_file), "--timeout", str(self.timeout)],
                capture_output=True,
                text=True,
                timeout=self.timeout + 10,  # Extra time for overhead
            )
        except subprocess.TimeoutExpired:
            return None

        if result.returncode != 0:
            return None  # No proof found

        # Parse proof from output
        # The prove binary outputs proof information to stdout
        output_lines = result.stdout.strip().split("\n")

        # Look for "Proof found" indicator
        if not any("Proof found" in line for line in output_lines):
            return None

        # Extract training data using Python interface
        # Read the problem and generate proof
        state = ProofState()
        state.add_clauses_from_file(str(problem_file))

        # Get proof trace (would need to be added to ProofState)
        # For now, return placeholder
        # TODO: Implement proper proof extraction from ProofState

        return {
            "problem": str(problem_file),
            "status": "proof_found",
            "training_examples": [],  # Placeholder
        }

    def collect_from_directory(
        self,
        problems_dir: Path,
        output_file: Path,
        pattern: str = "**/*.p",
        max_problems: Optional[int] = None,
    ):
        """
        Collect training data from all problems in a directory.

        Args:
            problems_dir: Directory containing TPTP problems
            output_file: Where to save collected training data
            pattern: Glob pattern for problem files
            max_problems: Maximum number of problems to process
        """
        problem_files = list(problems_dir.glob(pattern))

        if max_problems:
            problem_files = problem_files[:max_problems]

        training_dataset = []
        successful_problems = 0

        for problem_file in tqdm(problem_files, desc="Collecting data"):
            data = self.collect_from_problem(problem_file)

            if data is not None:
                training_dataset.append(data)
                successful_problems += 1

        print(f"\nCollected data from {successful_problems}/{len(problem_files)} problems")
        print(f"Total training examples: {sum(len(d['training_examples']) for d in training_dataset)}")

        # Save dataset
        torch.save(training_dataset, output_file)
        print(f"Saved training data to {output_file}")


def extract_training_examples_from_proof_steps(
    proof_steps: List[Dict[str, Any]],
    empty_clause_idx: int,
) -> List[Dict[str, Any]]:
    """
    Extract training examples from proof steps.

    This implements the proof DAG extraction in Python.

    Args:
        proof_steps: List of proof steps from ProofState
        empty_clause_idx: Index of the empty clause (proof goal)

    Returns:
        List of training examples with clause indices and labels
    """
    # Build proof DAG backwards from empty clause
    proof_clauses = set()
    to_visit = [empty_clause_idx]

    # Map clause index to proof step
    step_map = {step["clause_idx"]: step for step in proof_steps}

    while to_visit:
        clause_idx = to_visit.pop()

        if clause_idx in proof_clauses:
            continue

        proof_clauses.add(clause_idx)

        # Add parent clauses
        if clause_idx in step_map:
            step = step_map[clause_idx]
            to_visit.extend(step.get("parent_ids", []))

    # Create training examples
    training_examples = []
    all_clause_indices = set(step["clause_idx"] for step in proof_steps)

    for clause_idx in all_clause_indices:
        label = 1 if clause_idx in proof_clauses else 0
        training_examples.append({
            "clause_idx": clause_idx,
            "label": label,
        })

    return training_examples


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect ML training data from TPTP problems")
    parser.add_argument("problems_dir", type=Path, help="Directory containing TPTP problems")
    parser.add_argument("--output", "-o", type=Path, default=Path("data/training_data.pt"),
                        help="Output file for training data")
    parser.add_argument("--prove-binary", default="target/release/prove",
                        help="Path to prove binary")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Timeout per problem in seconds")
    parser.add_argument("--max-problems", type=int, default=None,
                        help="Maximum number of problems to process")

    args = parser.parse_args()

    collector = TrainingDataCollector(
        prove_binary=args.prove_binary,
        timeout=args.timeout,
    )

    collector.collect_from_directory(
        problems_dir=args.problems_dir,
        output_file=args.output,
        max_problems=args.max_problems,
    )
