"""Tests for loading serialized test data."""

from pathlib import Path

from proofatlas.core import (
    load_problem,
    Problem, Clause, Literal, Constant, Variable, Predicate
)
from proofatlas.proofs import (
    Proof, load_proof
)


TEST_DATA_DIR = Path(__file__).parent / "test_data"
PROBLEMS_DIR = TEST_DATA_DIR / "problems"
PROOFS_DIR = TEST_DATA_DIR / "proofs"


class TestSerializedProblems:
    """Test loading serialized problems."""
    
    def test_load_simple_contradiction(self):
        """Test loading simple contradiction problem."""
        problem = load_problem(PROBLEMS_DIR / "simple_contradiction.json")
        
        assert isinstance(problem, Problem)
        assert len(problem.clauses) == 2
        
        # Check first clause: P(a)
        c1 = problem.clauses[0]
        assert len(c1.literals) == 1
        assert c1.literals[0].polarity == True
        
        # Check second clause: ~P(a)
        c2 = problem.clauses[1]
        assert len(c2.literals) == 1
        assert c2.literals[0].polarity == False
    
    def test_load_modus_ponens(self):
        """Test loading modus ponens problem."""
        problem = load_problem(PROBLEMS_DIR / "modus_ponens.json")
        
        assert isinstance(problem, Problem)
        assert len(problem.clauses) == 3
        
        # P
        assert len(problem.clauses[0].literals) == 1
        assert problem.clauses[0].literals[0].polarity == True
        
        # ~P | Q
        assert len(problem.clauses[1].literals) == 2
        assert problem.clauses[1].literals[0].polarity == False
        assert problem.clauses[1].literals[1].polarity == True
        
        # ~Q
        assert len(problem.clauses[2].literals) == 1
        assert problem.clauses[2].literals[0].polarity == False
    
    def test_load_syllogism(self):
        """Test loading syllogism problem."""
        problem = load_problem(PROBLEMS_DIR / "syllogism.json")
        
        assert isinstance(problem, Problem)
        assert len(problem.clauses) == 3
        
        # ~Human(X) | Mortal(X)
        c1 = problem.clauses[0]
        assert len(c1.literals) == 2
        assert c1.literals[0].polarity == False
        assert c1.literals[1].polarity == True
        
        # Check that it contains a variable
        has_variable = False
        for lit in c1.literals:
            for arg in lit.predicate.args:
                if isinstance(arg, Variable):
                    has_variable = True
                    break
        assert has_variable
        
        # Human(socrates)
        c2 = problem.clauses[1]
        assert len(c2.literals) == 1
        assert c2.literals[0].polarity == True
        
        # ~Mortal(socrates)
        c3 = problem.clauses[2]
        assert len(c3.literals) == 1
        assert c3.literals[0].polarity == False


class TestSerializedProofs:
    """Test loading serialized proofs."""
    
    def test_load_simple_contradiction_proof(self):
        """Test loading proof for simple contradiction."""
        proof = load_proof(PROOFS_DIR / "simple_contradiction_proof.json")
        
        assert isinstance(proof, Proof)
        assert len(proof.steps) == 3
        
        # Initial state
        step0 = proof.steps[0]
        assert len(step0.state.processed) == 0
        assert len(step0.state.unprocessed) == 2
        assert step0.selected_clause == 0
        assert step0.metadata.get("rule") == "given_clause"
        
        # After selecting second clause (given clause algorithm)
        step1 = proof.steps[1]
        assert len(step1.state.processed) == 1
        assert len(step1.state.unprocessed) == 1
        assert step1.selected_clause == 0
        assert step1.metadata.get("rule") == "given_clause"
        # The resolution happens as part of processing this clause
        assert step1.metadata.get("inferences") is not None
        
        # Final state (no selection)
        step2 = proof.steps[2]
        assert len(step2.state.processed) == 2
        assert len(step2.state.unprocessed) == 1
        assert step2.selected_clause is None
        
        # Check that empty clause was found
        assert proof.is_complete
        empty_clause = step2.state.unprocessed[0]
        assert len(empty_clause.literals) == 0
    
    def test_proof_problem_correspondence(self):
        """Test that proof corresponds to its problem."""
        problem = load_problem(PROBLEMS_DIR / "simple_contradiction.json")
        proof = load_proof(PROOFS_DIR / "simple_contradiction_proof.json")
        
        # Initial state should contain all problem clauses
        initial_clauses = proof.initial_state.unprocessed
        assert len(initial_clauses) == len(problem.clauses)
        
        # Check that the clauses match (comparing by string repr)
        problem_strs = {str(c) for c in problem.clauses}
        initial_strs = {str(c) for c in initial_clauses}
        assert problem_strs == initial_strs


class TestRoundTrip:
    """Test that serialized data can be loaded and used."""
    
    def test_create_proof_from_loaded_problem(self):
        """Test creating a new proof from a loaded problem."""
        problem = load_problem(PROBLEMS_DIR / "modus_ponens.json")
        
        # Create a proof
        from proofatlas.proofs import ProofState, Proof
        
        initial_state = ProofState([], list(problem.clauses))
        proof = Proof(initial_state)
        
        # Should be able to manipulate the proof
        new_state = ProofState([problem.clauses[0]], problem.clauses[1:])
        proof.add_step(new_state, selected_clause=0, rule="given_clause")
        
        assert len(proof.steps) == 2
        assert proof.steps[0].selected_clause == 0
        assert proof.steps[1].selected_clause is None  # Invariant maintained