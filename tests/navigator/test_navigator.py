"""Tests for the proof navigator."""

import pytest
from unittest.mock import Mock, patch

from proofatlas.core import (
    Problem, Clause, Literal, Predicate, Constant
)
from proofatlas.proofs import (
    ProofState, Proof, ProofStep
)
from proofatlas.navigator import ProofNavigator


class TestProofNavigator:
    """Test ProofNavigator functionality."""
    
    @pytest.fixture
    def simple_proof(self):
        """Create a simple proof for testing."""
        P = Predicate("P", 1)
        a = Constant("a")
        
        c1 = Clause(Literal(P(a), True))
        c2 = Clause(Literal(P(a), False))
        
        # Initial state
        state1 = ProofState([], [c1, c2])
        proof = Proof(state1)
        
        # Process first clause
        state2 = ProofState([c1], [c2])
        proof.add_step(state2, selected_clause=0, rule="given_clause")
        
        # Process second clause and find contradiction
        empty = Clause()
        state3 = ProofState([c1, c2], [empty])
        proof.add_step(state3, selected_clause=0, rule="resolution")
        
        # Final state
        final_state = ProofState([c1, c2, empty], [])
        proof.finalize(final_state)
        
        return proof
    
    @pytest.fixture
    def simple_problem(self):
        """Create a simple problem for testing."""
        P = Predicate("P", 1)
        a = Constant("a")
        
        return Problem(
            Clause(Literal(P(a), True)),
            Clause(Literal(P(a), False))
        )
    
    def test_navigator_init(self, simple_proof, simple_problem):
        """Test navigator initialization."""
        nav = ProofNavigator(simple_proof, simple_problem)
        
        assert nav.proof == simple_proof
        assert nav.problem == simple_problem
        assert nav.current_step == 0
        assert nav.total_steps == len(simple_proof.steps)
    
    def test_format_clause(self, simple_proof):
        """Test clause formatting."""
        nav = ProofNavigator(simple_proof)
        
        # Regular clause
        P = Predicate("P", 1)
        a = Constant("a")
        clause = Clause(Literal(P(a), True))
        assert nav.format_clause(clause) == "P(a)"
        assert nav.format_clause(clause, 0) == "[ 0] P(a)"
        
        # Negated literal
        clause = Clause(Literal(P(a), False))
        assert nav.format_clause(clause) == "¬P(a)"
        
        # Empty clause
        empty = Clause()
        assert nav.format_clause(empty) == "⊥ (empty clause)"
        
        # Multiple literals
        Q = Predicate("Q", 1)
        clause = Clause(
            Literal(P(a), False),
            Literal(Q(a), True)
        )
        assert nav.format_clause(clause) == "¬P(a) ∨ Q(a)"
    
    def test_format_literal(self, simple_proof):
        """Test literal formatting."""
        nav = ProofNavigator(simple_proof)
        
        P = Predicate("P", 1)
        a = Constant("a")
        
        # Positive literal
        lit = Literal(P(a), True)
        assert nav.format_literal(lit) == "P(a)"
        
        # Negative literal
        lit = Literal(P(a), False)
        assert nav.format_literal(lit) == "¬P(a)"
    
    def test_navigation_keys(self, simple_proof):
        """Test navigation key handling."""
        nav = ProofNavigator(simple_proof)
        
        # Test moving forward
        assert nav.current_step == 0
        nav.current_step = 1  # Simulate RIGHT key
        assert nav.current_step == 1
        
        # Test moving backward
        nav.current_step = 0  # Simulate LEFT key
        assert nav.current_step == 0
        
        # Test boundaries
        nav.current_step = nav.total_steps - 1
        assert nav.current_step == nav.total_steps - 1
    
    @patch('sys.stdin')
    def test_get_key(self, mock_stdin, simple_proof):
        """Test key capture."""
        nav = ProofNavigator(simple_proof)
        
        # Regular key
        mock_stdin.read.return_value = 'q'
        assert nav.get_key() == 'q'
        
        # Arrow key (escape sequence)
        mock_stdin.read.side_effect = ['\x1b', '[A']
        assert nav.get_key() == 'UP'
        
        mock_stdin.read.side_effect = ['\x1b', '[B']
        assert nav.get_key() == 'DOWN'
        
        mock_stdin.read.side_effect = ['\x1b', '[C']
        assert nav.get_key() == 'RIGHT'
        
        mock_stdin.read.side_effect = ['\x1b', '[D']
        assert nav.get_key() == 'LEFT'