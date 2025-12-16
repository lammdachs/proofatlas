"""Collect training data from TPTP problems for clause selection learning"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import torch

from proofatlas import ProofState
from .graph_utils import to_torch_tensors


@dataclass
class TrainingDataset:
    """Dataset of training examples for clause selection"""

    # List of graph tensors (one per clause)
    graphs: List[Dict[str, torch.Tensor]]

    # Labels (1 = in proof, 0 = not in proof)
    labels: torch.Tensor

    # Metadata
    problem_names: List[str]  # Problem name for each example
    clause_ids: List[int]     # Original clause ID


def run_saturation_loop(
    state: ProofState,
    max_iterations: int = 10000,
    timeout_secs: float = 60.0,
) -> bool:
    """
    Run saturation loop on a ProofState until proof found or limit reached.

    This uses the full Rust saturation engine which includes demodulation
    and other optimizations for better proof finding.

    Args:
        state: ProofState with initial clauses
        max_iterations: Maximum number of iterations
        timeout_secs: Timeout in seconds

    Returns:
        True if proof found, False otherwise
    """
    proof_found, _status = state.run_saturation(max_iterations, timeout_secs)
    return proof_found


def collect_from_problem(
    problem_file: Path,
    max_iterations: int = 10000,
    timeout_secs: float = 60.0,
) -> Optional[Dict[str, Any]]:
    """
    Run prover on a problem and extract training data if proof found.

    Args:
        problem_file: Path to TPTP problem file
        max_iterations: Maximum saturation iterations
        timeout_secs: Timeout in seconds

    Returns:
        Dictionary with training data, or None if no proof found
    """
    # Read and parse problem
    with open(problem_file) as f:
        content = f.read()

    # Create proof state and add clauses
    state = ProofState()
    try:
        state.add_clauses_from_tptp(content)
    except Exception as e:
        print(f"Parse error for {problem_file}: {e}")
        return None

    # Run saturation
    proof_found = run_saturation_loop(state, max_iterations, timeout_secs)

    if not proof_found:
        return None

    # Extract training examples
    examples = state.extract_training_examples()

    if not examples:
        return None

    # Get clause graphs for all clauses
    clause_ids = [e.clause_idx for e in examples]
    graphs = state.clauses_to_graphs(clause_ids)

    # Convert graphs to tensors
    graph_tensors = []
    for graph in graphs:
        tensors = to_torch_tensors(graph)
        graph_tensors.append(tensors)

    # Get labels
    labels = [e.label for e in examples]

    # Get statistics
    stats = state.get_proof_statistics()

    return {
        "problem": str(problem_file),
        "graphs": graph_tensors,
        "labels": labels,
        "clause_ids": clause_ids,
        "statistics": stats,
    }


def collect_from_directory(
    problems_dir: Path,
    output_file: Path,
    pattern: str = "**/*.p",
    max_problems: Optional[int] = None,
    max_iterations: int = 10000,
    timeout_secs: float = 60.0,
    verbose: bool = True,
) -> TrainingDataset:
    """
    Collect training data from all problems in a directory.

    Args:
        problems_dir: Directory containing TPTP problems
        output_file: Where to save collected training data
        pattern: Glob pattern for problem files
        max_problems: Maximum number of problems to process
        max_iterations: Max saturation iterations per problem
        timeout_secs: Timeout in seconds per problem
        verbose: Print progress

    Returns:
        TrainingDataset with all collected examples
    """
    problem_files = sorted(problems_dir.glob(pattern))

    if max_problems:
        problem_files = problem_files[:max_problems]

    all_graphs = []
    all_labels = []
    all_problem_names = []
    all_clause_ids = []

    successful = 0
    failed = 0

    for i, problem_file in enumerate(problem_files):
        if verbose and (i + 1) % 10 == 0:
            print(f"Processing {i + 1}/{len(problem_files)}: {problem_file.name}")

        try:
            data = collect_from_problem(problem_file, max_iterations, timeout_secs)
        except Exception as e:
            if verbose:
                print(f"Error processing {problem_file}: {e}")
            failed += 1
            continue

        if data is None:
            failed += 1
            continue

        successful += 1

        # Accumulate data
        all_graphs.extend(data["graphs"])
        all_labels.extend(data["labels"])
        all_problem_names.extend([str(problem_file)] * len(data["labels"]))
        all_clause_ids.extend(data["clause_ids"])

    if verbose:
        print(f"\nCollected data from {successful}/{len(problem_files)} problems")
        print(f"Total training examples: {len(all_labels)}")
        if all_labels:
            pos = sum(all_labels)
            neg = len(all_labels) - pos
            print(f"Positive (in proof): {pos} ({100*pos/len(all_labels):.1f}%)")
            print(f"Negative (not in proof): {neg} ({100*neg/len(all_labels):.1f}%)")

    # Create dataset
    dataset = TrainingDataset(
        graphs=all_graphs,
        labels=torch.tensor(all_labels, dtype=torch.float32),
        problem_names=all_problem_names,
        clause_ids=all_clause_ids,
    )

    # Save dataset
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "graphs": all_graphs,
        "labels": all_labels,
        "problem_names": all_problem_names,
        "clause_ids": all_clause_ids,
    }, output_file)

    if verbose:
        print(f"Saved training data to {output_file}")

    return dataset


def load_training_dataset(path: Path) -> TrainingDataset:
    """Load a saved training dataset"""
    data = torch.load(path)
    return TrainingDataset(
        graphs=data["graphs"],
        labels=torch.tensor(data["labels"], dtype=torch.float32),
        problem_names=data["problem_names"],
        clause_ids=data["clause_ids"],
    )


# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect ML training data from TPTP problems"
    )
    parser.add_argument(
        "problems_dir",
        type=Path,
        help="Directory containing TPTP problems"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/training_data.pt"),
        help="Output file for training data"
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Maximum number of problems to process"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10000,
        help="Maximum saturation iterations per problem"
    )
    parser.add_argument(
        "--pattern",
        default="**/*.p",
        help="Glob pattern for problem files"
    )

    args = parser.parse_args()

    collect_from_directory(
        problems_dir=args.problems_dir,
        output_file=args.output,
        pattern=args.pattern,
        max_problems=args.max_problems,
        max_iterations=args.max_iterations,
    )
