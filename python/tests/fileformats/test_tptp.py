"""Tests for TPTP file format handler."""

import pytest
from pathlib import Path
import tempfile

from proofatlas.fileformats.tptp import TPTPFormat
from proofatlas.core.logic import (
    Variable, Constant, Function, Predicate, 
    Literal, Clause, Problem
)


class TestTPTPFormat:
    """Test TPTPFormat functionality."""
    
    @pytest.fixture
    def tptp_format(self):
        """Create TPTPFormat instance."""
        return TPTPFormat()
    
    def test_properties(self, tptp_format):
        """Test format properties."""
        assert tptp_format.name == 'tptp'
        assert tptp_format.extensions == ['.p', '.tptp', '.ax']
    
    def test_parse_string_simple(self, tptp_format):
        """Test parsing a simple TPTP string."""
        content = """
        cnf(clause1, axiom, p(a)).
        cnf(clause2, axiom, ~p(X) | q(X)).
        cnf(clause3, axiom, ~q(a)).
        """
        
        problem = tptp_format.parse_string(content)
        assert isinstance(problem, Problem)
        assert len(problem.clauses) == 3
    
    def test_parse_string_fof(self, tptp_format):
        """Test parsing FOF formulas."""
        content = """
        fof(axiom1, axiom, p(a)).
        fof(axiom2, axiom, ![X] : (p(X) => q(X))).
        fof(goal, conjecture, q(a)).
        """
        
        problem = tptp_format.parse_string(content)
        assert isinstance(problem, Problem)
        # FOF gets converted to CNF, conjecture gets negated
        assert len(problem.clauses) >= 3
    
    def test_parse_file(self, tptp_format):
        """Test parsing from file."""
        content = "cnf(test, axiom, p(a))."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.p', delete=False) as f:
            f.write(content)
            f.flush()
            
            problem = tptp_format.parse_file(Path(f.name))
            assert isinstance(problem, Problem)
            assert len(problem.clauses) == 1
            
            # Cleanup
            Path(f.name).unlink()
    
    def test_format_problem(self, tptp_format):
        """Test formatting a problem as TPTP string."""
        # Create a simple problem
        a = Constant("a")
        P = Predicate("p", 1)
        Q = Predicate("q", 1)
        
        c1 = Clause(Literal(P(a), True))
        c2 = Clause(Literal(Q(a), False))
        problem = Problem(c1, c2)
        
        result = tptp_format.format_problem(problem)
        assert isinstance(result, str)
        assert "cnf(clause_0" in result
        assert "cnf(clause_1" in result
        assert "p(a)" in result
        assert "~q(a)" in result
    
    def test_write_file(self, tptp_format):
        """Test writing problem to file."""
        # Create a simple problem
        a = Constant("a")
        P = Predicate("p", 1)
        problem = Problem(Clause(Literal(P(a), True)))
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.p', delete=False) as f:
            tptp_format.write_file(problem, Path(f.name))
            
            # Read back and verify
            with open(f.name, 'r') as rf:
                content = rf.read()
                assert "cnf(clause_0" in content
                assert "p(a)" in content
            
            # Cleanup
            Path(f.name).unlink()
    
    def test_clause_to_tptp(self, tptp_format):
        """Test converting clauses to TPTP format."""
        # Empty clause
        empty = Clause()
        assert tptp_format._clause_to_tptp(empty) == "$false"
        
        # Single literal
        a = Constant("a")
        P = Predicate("p", 1)
        single = Clause(Literal(P(a), True))
        assert tptp_format._clause_to_tptp(single) == "p(a)"
        
        # Multiple literals
        Q = Predicate("q", 1)
        multi = Clause(Literal(P(a), False), Literal(Q(a), True))
        result = tptp_format._clause_to_tptp(multi)
        assert "~p(a)" in result
        assert "q(a)" in result
        assert "|" in result
    
    def test_literal_to_tptp(self, tptp_format):
        """Test converting literals to TPTP format."""
        a = Constant("a")
        P = Predicate("p", 1)
        
        # Positive literal
        pos = Literal(P(a), True)
        assert tptp_format._literal_to_tptp(pos) == "p(a)"
        
        # Negative literal
        neg = Literal(P(a), False)
        assert tptp_format._literal_to_tptp(neg) == "~p(a)"
    
    def test_atom_to_tptp(self, tptp_format):
        """Test converting atoms to TPTP format."""
        a = Constant("a")
        b = Constant("b")
        
        # Equality
        eq = Predicate("=", 2)
        eq_atom = eq(a, b)
        assert tptp_format._atom_to_tptp(eq_atom) == "a = b"
        
        # Propositional
        P = Predicate("p", 0)
        prop = P()
        assert tptp_format._atom_to_tptp(prop) == "p"
        
        # Unary predicate
        Q = Predicate("q", 1)
        unary = Q(a)
        assert tptp_format._atom_to_tptp(unary) == "q(a)"
        
        # Binary predicate
        R = Predicate("r", 2)
        binary = R(a, b)
        assert tptp_format._atom_to_tptp(binary) == "r(a, b)"
    
    def test_term_to_tptp(self, tptp_format):
        """Test converting terms to TPTP format."""
        # Variable
        x = Variable("X")
        assert tptp_format._term_to_tptp(x) == "X"
        
        # Constant
        a = Constant("a")
        assert tptp_format._term_to_tptp(a) == "a"
        
        # Function with no args (same as constant)
        c = Function("c", 0)()
        assert tptp_format._term_to_tptp(c) == "c"
        
        # Function with args
        f = Function("f", 2)
        term = f(a, x)
        assert tptp_format._term_to_tptp(term) == "f(a, X)"