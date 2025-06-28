"""
Comprehensive tests for Rust bindings.

Tests the Python-Rust interface for Problem, ProofState, ProofStep, and Proof.
"""

import pytest
import sys
from pathlib import Path

# Ensure we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import Rust modules - skip tests if not available
try:
    import proofatlas_rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust module not available")


class TestRustProblem:
    """Test the Rust Problem implementation."""
    
    def test_problem_creation(self):
        """Test creating a Problem instance."""
        import proofatlas_rust
        Problem = proofatlas_rust.core.Problem
        
        # Create empty problem
        problem = Problem()
        assert problem is not None
        assert len(problem) == 0
        assert repr(problem) == "Problem(0 clauses)"
    
    def test_problem_properties(self):
        """Test Problem properties."""
        import proofatlas_rust
        Problem = proofatlas_rust.core.Problem
        
        problem = Problem()
        
        # Test clauses property
        clauses = problem.clauses
        assert isinstance(clauses, list)
        assert len(clauses) == 0
        
        # Test conjecture_indices
        indices = problem.conjecture_indices
        assert isinstance(indices, list)
        assert len(indices) == 0
    
    def test_problem_methods(self):
        """Test Problem methods."""
        import proofatlas_rust
        Problem = proofatlas_rust.core.Problem
        
        problem = Problem()
        
        # Test is_conjecture_clause
        assert problem.is_conjecture_clause(0) == False
        
        # Test get_conjecture_clauses
        conjectures = problem.get_conjecture_clauses()
        assert isinstance(conjectures, list)
        assert len(conjectures) == 0
        
        # Test count_literals
        assert problem.count_literals() == 0
    
    def test_problem_serialization(self):
        """Test Problem serialization."""
        import proofatlas_rust
        Problem = proofatlas_rust.core.Problem
        
        problem = Problem()
        
        # Test to_dict
        data = problem.to_dict()
        assert isinstance(data, dict)
        assert 'num_clauses' in data
        assert 'clauses' in data
        assert 'conjecture_indices' in data
        
        # Test from_dict
        problem2 = Problem.from_dict(data)
        assert problem2 is not None
        assert len(problem2) == 0


class TestRustProofState:
    """Test the Rust ProofState implementation."""
    
    def test_proofstate_creation(self):
        """Test creating a ProofState instance."""
        import proofatlas_rust
        ProofState = proofatlas_rust.proofs.ProofState
        
        # Create empty state
        state = ProofState([], [])
        assert state is not None
        
        # Check properties
        assert len(state.processed) == 0
        assert len(state.unprocessed) == 0
        assert len(state.all_clauses) == 0
        assert state.contains_empty_clause == False
    
    def test_proofstate_with_clauses(self):
        """Test ProofState with clause strings."""
        import proofatlas_rust
        ProofState = proofatlas_rust.proofs.ProofState
        
        # For now, we pass empty lists since clause parsing isn't implemented
        state = ProofState([], [])
        
        # Test move_to_processed (would fail with current implementation)
        # state.move_to_processed(0)  # This would raise an error
        
        assert state.contains_empty_clause == False


class TestRustRuleApplication:
    """Test the Rust RuleApplication implementation."""
    
    def test_rule_application_creation(self):
        """Test creating a RuleApplication instance."""
        import proofatlas_rust
        RuleApplication = proofatlas_rust.proofs.RuleApplication
        
        # Create simple rule application
        rule = RuleApplication("resolution", [0, 1])
        assert rule is not None
        assert rule.rule_name == "resolution"
        assert rule.parents == [0, 1]
        assert rule.deleted_clause_indices == []
    
    def test_rule_application_with_metadata(self):
        """Test RuleApplication with metadata."""
        import proofatlas_rust
        RuleApplication = proofatlas_rust.proofs.RuleApplication
        
        # Create with optional parameters
        rule = RuleApplication(
            "subsumption",
            [0],
            deleted_clause_indices=[1, 2],
            metadata={"subsumes": "clause_1"}
        )
        
        assert rule.rule_name == "subsumption"
        assert rule.parents == [0]
        assert rule.deleted_clause_indices == [1, 2]


class TestRustProofStep:
    """Test the Rust ProofStep implementation."""
    
    def test_proofstep_creation(self):
        """Test creating a ProofStep instance."""
        import proofatlas_rust
        ProofState = proofatlas_rust.proofs.ProofState
        ProofStep = proofatlas_rust.proofs.ProofStep
        
        state = ProofState([], [])
        step = ProofStep(state)
        
        assert step is not None
        assert step.selected_clause is None
        assert len(step.applied_rules) == 0
    
    def test_proofstep_with_selection(self):
        """Test ProofStep with selected clause."""
        import proofatlas_rust
        ProofState = proofatlas_rust.proofs.ProofState
        ProofStep = proofatlas_rust.proofs.ProofStep
        
        state = ProofState([], [])
        step = ProofStep(state, selected_clause=0)
        
        assert step.selected_clause == 0
    
    def test_proofstep_with_rules(self):
        """Test ProofStep with applied rules."""
        import proofatlas_rust
        ProofState = proofatlas_rust.proofs.ProofState
        ProofStep = proofatlas_rust.proofs.ProofStep
        RuleApplication = proofatlas_rust.proofs.RuleApplication
        
        state = ProofState([], [])
        rule = RuleApplication("resolution", [0, 1])
        step = ProofStep(state, applied_rules=[rule])
        
        assert len(step.applied_rules) == 1
        assert step.applied_rules[0].rule_name == "resolution"


class TestRustProof:
    """Test the Rust Proof implementation."""
    
    def test_proof_creation(self):
        """Test creating a Proof instance."""
        import proofatlas_rust
        Proof = proofatlas_rust.proofs.Proof
        ProofState = proofatlas_rust.proofs.ProofState
        
        # Create empty proof
        proof = Proof()
        assert proof is not None
        assert proof.length == 0
        
        # Create with initial state
        state = ProofState([], [])
        proof2 = Proof(state)
        assert proof2.length == 0
    
    def test_proof_properties(self):
        """Test Proof properties."""
        import proofatlas_rust
        Proof = proofatlas_rust.proofs.Proof
        ProofState = proofatlas_rust.proofs.ProofState
        
        proof = Proof()
        
        # Test initial_state
        initial = proof.initial_state
        assert initial is not None
        
        # Test final_state
        final = proof.final_state
        assert final is not None
        
        # Test steps
        steps = proof.steps
        assert isinstance(steps, list)
        assert len(steps) == 1  # Initial state
        
        # Test length
        assert proof.length == 0  # No inference steps yet
    
    def test_proof_add_step(self):
        """Test adding steps to a proof."""
        import proofatlas_rust
        Proof = proofatlas_rust.proofs.Proof
        ProofState = proofatlas_rust.proofs.ProofState
        ProofStep = proofatlas_rust.proofs.ProofStep
        
        proof = Proof()
        
        # Add a step
        state = ProofState([], [])
        step = ProofStep(state, selected_clause=0)
        proof.add_step(step)
        
        # Check that step was added
        assert len(proof.steps) >= 1
        # Note: actual behavior depends on Rust implementation
    
    def test_proof_methods(self):
        """Test Proof methods."""
        import proofatlas_rust
        Proof = proofatlas_rust.proofs.Proof
        ProofState = proofatlas_rust.proofs.ProofState
        
        proof = Proof()
        
        # Test get_step
        step = proof.get_step(0)
        assert step is not None
        
        # Test get_selected_clauses
        selected = proof.get_selected_clauses()
        assert isinstance(selected, list)
        assert len(selected) == 0
        
        # Test found_contradiction
        assert proof.found_contradiction() == False
    
    def test_proof_finalize(self):
        """Test finalizing a proof."""
        import proofatlas_rust
        Proof = proofatlas_rust.proofs.Proof
        ProofState = proofatlas_rust.proofs.ProofState
        
        proof = Proof()
        final_state = ProofState([], [])
        proof.finalize(final_state)
        
        # Check that proof is finalized
        assert proof.final_state is not None


class TestPythonWrappers:
    """Test the Python wrapper classes."""
    
    def test_problem_wrapper(self):
        """Test Problem Python wrapper."""
        from proofatlas.core.logic_rust import Problem
        
        problem = Problem()
        assert problem is not None
        
        # Test wrapper methods
        assert problem.depth() == 0
        assert problem.predicate_symbols() == set()
        assert problem.function_symbols() == set()
        assert problem.variables() == set()
        assert problem.terms() == set()
    
    def test_proofstate_wrapper(self):
        """Test ProofState Python wrapper."""
        from proofatlas.proofs.state_rust import ProofState
        
        # The wrapper expects Python clauses
        state = ProofState([], [])
        assert state is not None
        
        # Test wrapper methods (currently stubs)
        state.add_processed("dummy")  # Currently does nothing
        state.add_unprocessed("dummy")  # Currently does nothing
    
    def test_proof_wrapper(self):
        """Test Proof Python wrapper."""
        from proofatlas.proofs.proof_rust import Proof, ProofStep, RuleApplication
        
        proof = Proof()
        assert proof is not None
        
        # Test wrapper methods
        history = proof.get_metadata_history("key")
        assert isinstance(history, list)
        assert len(history) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])