"""Example demonstrating the proof navigator."""

from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from proofatlas.navigator import navigate_proof


def main():
    """Navigate the simple contradiction proof."""
    test_data_dir = Path(__file__).parent.parent / "tests" / ".data"
    
    proof_path = test_data_dir / "proofs" / "simple_contradiction_proof.json"
    problem_path = test_data_dir / "problems" / "simple_contradiction.json"
    
    print("Starting proof navigator...")
    print(f"Proof: {proof_path}")
    print(f"Problem: {problem_path}")
    print()
    print("Press Enter to start navigation...")
    input()
    
    # Navigate the proof
    navigate_proof(str(proof_path), str(problem_path))
    
    print("Navigation complete!")


if __name__ == "__main__":
    main()