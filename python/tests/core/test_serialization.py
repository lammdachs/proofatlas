"""Tests for core JSON serialization."""

import json
import tempfile
from pathlib import Path

import pytest

from proofatlas.core import (
    Variable, Constant, Function, Predicate,
    Literal, Clause, Problem,
    problem_to_json, problem_from_json,
    save_problem, load_problem
)
from proofatlas.proofs import (
    ProofState, Proof, ProofStep,
    proof_to_json, proof_from_json,
    save_proof, load_proof
)


class TestJSONSerialization:
    """Test JSON serialization of core objects."""
    
    def test_variable_serialization(self):
        """Test Variable serialization."""
        x = Variable("X")
        json_str = json.dumps(x, cls=problem_to_json.__globals__['CoreJSONEncoder'])
        data = json.loads(json_str)
        
        assert data["_type"] == "Variable"
        assert data["name"] == "X"
        
        # Round trip
        x2 = json.loads(json_str, object_hook=problem_from_json.__globals__['decode_core_object'])
        assert isinstance(x2, Variable)
        assert x2.name == x.name
    
    def test_constant_serialization(self):
        """Test Constant serialization."""
        a = Constant("a")
        json_str = json.dumps(a, cls=problem_to_json.__globals__['CoreJSONEncoder'])
        data = json.loads(json_str)
        
        assert data["_type"] == "Constant"
        assert data["name"] == "a"
        
        # Round trip
        a2 = json.loads(json_str, object_hook=problem_from_json.__globals__['decode_core_object'])
        assert isinstance(a2, Constant)
        assert a2.name == a.name
    
    def test_function_serialization(self):
        """Test Function serialization."""
        f = Function("f", 2)
        json_str = json.dumps(f, cls=problem_to_json.__globals__['CoreJSONEncoder'])
        data = json.loads(json_str)
        
        assert data["_type"] == "Function"
        assert data["name"] == "f"
        assert data["arity"] == 2
    
    def test_predicate_serialization(self):
        """Test Predicate serialization."""
        P = Predicate("P", 1)
        json_str = json.dumps(P, cls=problem_to_json.__globals__['CoreJSONEncoder'])
        data = json.loads(json_str)
        
        assert data["_type"] == "Predicate"
        assert data["name"] == "P"
        assert data["arity"] == 1
    
    def test_term_serialization(self):
        """Test compound Term serialization."""
        f = Function("f", 2)
        a = Constant("a")
        x = Variable("X")
        term = f(a, x)
        
        json_str = json.dumps(term, cls=problem_to_json.__globals__['CoreJSONEncoder'])
        data = json.loads(json_str)
        
        assert data["_type"] == "Term"
        assert data["symbol"]["_type"] == "Function"
        assert len(data["args"]) == 2
    
    def test_literal_serialization(self):
        """Test Literal serialization."""
        P = Predicate("P", 1)
        a = Constant("a")
        lit = Literal(P(a), True)
        
        json_str = json.dumps(lit, cls=problem_to_json.__globals__['CoreJSONEncoder'])
        lit2 = json.loads(json_str, object_hook=problem_from_json.__globals__['decode_core_object'])
        
        assert isinstance(lit2, Literal)
        assert lit2.polarity == True
    
    def test_clause_serialization(self):
        """Test Clause serialization."""
        P = Predicate("P", 1)
        Q = Predicate("Q", 1)
        a = Constant("a")
        
        clause = Clause(Literal(P(a), False), Literal(Q(a), True))
        json_str = json.dumps(clause, cls=problem_to_json.__globals__['CoreJSONEncoder'])
        clause2 = json.loads(json_str, object_hook=problem_from_json.__globals__['decode_core_object'])
        
        assert isinstance(clause2, Clause)
        assert len(clause2.literals) == 2
    
    def test_problem_serialization(self):
        """Test Problem serialization."""
        P = Predicate("P", 1)
        Q = Predicate("Q", 1)
        a = Constant("a")
        x = Variable("X")
        
        problem = Problem(
            Clause(Literal(P(a), True)),
            Clause(Literal(P(x), False), Literal(Q(x), True)),
            Clause(Literal(Q(a), False))
        )
        
        # To JSON and back
        json_str = problem_to_json(problem)
        problem2 = problem_from_json(json_str)
        
        assert isinstance(problem2, Problem)
        assert len(problem2.clauses) == 3
        assert len(problem2.clauses[0].literals) == 1
        assert len(problem2.clauses[1].literals) == 2
        assert len(problem2.clauses[2].literals) == 1
    
    def test_proof_serialization(self):
        """Test Proof serialization."""
        P = Predicate("P", 1)
        a = Constant("a")
        
        c1 = Clause(Literal(P(a), True))
        c2 = Clause(Literal(P(a), False))
        empty = Clause()
        
        # Create a simple proof
        state1 = ProofState([], [c1, c2])
        proof = Proof(state1)
        
        state2 = ProofState([c1], [c2])
        proof.add_step(state2, selected_clause=0, rule="given_clause")
        
        state3 = ProofState([c1, c2], [empty])
        proof.add_step(state3, selected_clause=1, rule="resolution")
        
        final_state = ProofState([c1, c2, empty], [])
        proof.finalize(final_state)
        
        # Check original proof structure
        print(f"Original proof has {len(proof.steps)} steps")
        for i, step in enumerate(proof.steps):
            print(f"  Step {i}: selected={step.selected_clause}")
        
        # Serialize and deserialize
        json_str = proof_to_json(proof)
        proof2 = proof_from_json(json_str)
        
        assert isinstance(proof2, Proof)
        # The proof maintains the invariant that last step has no selection
        assert proof2.steps[-1].selected_clause is None
        assert proof2.is_complete  # Found empty clause
        
        # Check that the essential information is preserved
        assert len(proof.steps) == len(proof2.steps)
        for i in range(len(proof.steps)):
            assert proof.steps[i].selected_clause == proof2.steps[i].selected_clause
            assert len(proof.steps[i].state.processed) == len(proof2.steps[i].state.processed)
            assert len(proof.steps[i].state.unprocessed) == len(proof2.steps[i].state.unprocessed)
    
    def test_save_load_problem(self):
        """Test saving and loading Problem to/from file."""
        P = Predicate("P", 1)
        a = Constant("a")
        problem = Problem(Clause(Literal(P(a), True)))
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            save_problem(problem, f.name)
            problem2 = load_problem(f.name)
            
            assert isinstance(problem2, Problem)
            assert len(problem2.clauses) == 1
            
            # Cleanup
            Path(f.name).unlink()
    
    def test_save_load_proof(self):
        """Test saving and loading Proof to/from file."""
        P = Predicate("P", 1)
        a = Constant("a")
        
        state = ProofState([], [Clause(Literal(P(a), True))])
        proof = Proof(state)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            save_proof(proof, f.name)
            proof2 = load_proof(f.name)
            
            assert isinstance(proof2, Proof)
            assert len(proof2.steps) == 1
            
            # Cleanup
            Path(f.name).unlink()
    
    def test_complex_problem_serialization(self):
        """Test serialization of a more complex problem."""
        # Predicates and functions
        P = Predicate("P", 2)
        Q = Predicate("Q", 1)
        R = Predicate("R", 3)
        f = Function("f", 1)
        g = Function("g", 2)
        
        # Constants and variables
        a = Constant("a")
        b = Constant("b")
        x = Variable("X")
        y = Variable("Y")
        
        # Create complex terms
        fa = f(a)
        gxy = g(x, y)
        fgxy = f(gxy)
        
        # Create clauses
        c1 = Clause(
            Literal(P(x, fa), True),
            Literal(Q(x), False)
        )
        c2 = Clause(
            Literal(R(a, b, fgxy), True),
            Literal(P(gxy, y), False)
        )
        c3 = Clause(Literal(Q(b), True))
        
        problem = Problem(c1, c2, c3)
        
        # Serialize and deserialize
        json_str = problem_to_json(problem)
        problem2 = problem_from_json(json_str)
        
        assert len(problem2.clauses) == 3
        
        # Verify structure is preserved
        assert len(problem2.clauses[0].literals) == 2
        assert len(problem2.clauses[1].literals) == 2
        assert len(problem2.clauses[2].literals) == 1
    
    def test_metadata_preservation(self):
        """Test that proof metadata is preserved."""
        state1 = ProofState([], [])
        proof = Proof(state1)
        
        state2 = ProofState([], [])
        proof.add_step(
            state2, 
            selected_clause=0,
            rule="resolution",
            parent_clauses=[0, 1],
            custom_data={"score": 0.95}
        )
        
        json_str = proof_to_json(proof)
        proof2 = proof_from_json(json_str)
        
        # The step with metadata should be preserved
        assert len(proof2.steps) == len(proof.steps)
        
        # Find the step with our metadata
        step_with_metadata = None
        for step in proof2.steps:
            if "rule" in step.metadata and step.metadata["rule"] == "resolution":
                step_with_metadata = step
                break
        
        assert step_with_metadata is not None
        assert step_with_metadata.selected_clause == 0
        assert step_with_metadata.metadata["rule"] == "resolution"
        assert step_with_metadata.metadata["parent_clauses"] == [0, 1]
        assert step_with_metadata.metadata["custom_data"]["score"] == 0.95