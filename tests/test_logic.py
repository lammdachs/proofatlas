"""Tests for core.logic module."""

import unittest
from proofatlas.core.logic import (
    Variable, Constant, Function, Predicate,
    Term, Literal, Clause, Problem,
    reset_context, get_predicates, get_functions
)


class TestSymbols(unittest.TestCase):
    """Test basic symbol classes."""
    
    def test_variable_creation(self):
        """Test creating variables."""
        x = Variable("X")
        self.assertEqual(x.name, "X")
        self.assertEqual(x.arity, 0)
        self.assertIsInstance(x, Variable)
        self.assertIsInstance(x, Term)
    
    def test_constant_creation(self):
        """Test creating constants."""
        a = Constant("a")
        self.assertEqual(a.name, "a")
        self.assertEqual(a.arity, 0)
        self.assertIsInstance(a, Constant)
        self.assertIsInstance(a, Function)
        self.assertIsInstance(a, Term)
    
    def test_function_creation(self):
        """Test creating function symbols."""
        f = Function("f", 1)
        g = Function("g", 2)
        self.assertEqual(f.name, "f")
        self.assertEqual(f.arity, 1)
        self.assertEqual(g.arity, 2)
    
    def test_predicate_creation(self):
        """Test creating predicate symbols."""
        P = Predicate("P", 1)
        Q = Predicate("Q", 2)
        self.assertEqual(P.name, "P")
        self.assertEqual(P.arity, 1)
        self.assertEqual(Q.arity, 2)
    
    def test_symbol_hashing(self):
        """Test that symbols can be hashed and compared."""
        f1 = Function("f", 1)
        f2 = Function("f", 1)
        g = Function("g", 1)
        
        # Same name and arity should be equal
        self.assertEqual(hash(f1), hash(f2))
        self.assertNotEqual(hash(f1), hash(g))
        
        # Can be used in sets
        symbols = {f1, f2, g}
        self.assertEqual(len(symbols), 2)  # f1 and f2 are the same


class TestTerms(unittest.TestCase):
    """Test term construction and manipulation."""
    
    def setUp(self):
        """Set up common symbols."""
        self.x = Variable("X")
        self.y = Variable("Y")
        self.a = Constant("a")
        self.f = Function("f", 1)
        self.g = Function("g", 2)
    
    def test_simple_terms(self):
        """Test creating simple terms."""
        # f(a)
        fa = self.f(self.a)
        self.assertIsInstance(fa, Term)
        self.assertEqual(fa.symbol, self.f)
        self.assertEqual(len(fa.args), 1)
        self.assertEqual(fa.args[0], self.a)
        
        # g(x, y)
        gxy = self.g(self.x, self.y)
        self.assertEqual(gxy.symbol, self.g)
        self.assertEqual(len(gxy.args), 2)
        self.assertEqual(gxy.args[0], self.x)
        self.assertEqual(gxy.args[1], self.y)
    
    def test_nested_terms(self):
        """Test creating nested terms."""
        # f(g(x, a))
        gxa = self.g(self.x, self.a)
        fgxa = self.f(gxa)
        
        self.assertEqual(fgxa.symbol, self.f)
        self.assertEqual(fgxa.args[0].symbol, self.g)
        self.assertEqual(fgxa.args[0].args[0], self.x)
        self.assertEqual(fgxa.args[0].args[1], self.a)
    
    def test_term_methods(self):
        """Test term methods like variables() and symbols()."""
        # Create term f(g(X, a), Y)
        gxa = self.g(self.x, self.a)
        fgxay = self.f(gxa, self.y)
        
        # Check variables
        vars = fgxay.variables()
        self.assertEqual(len(vars), 2)
        self.assertIn(self.x, vars)
        self.assertIn(self.y, vars)
        
        # Check function symbols
        symbols = fgxay.function_symbols()
        self.assertIn(self.f, symbols)
        self.assertIn(self.g, symbols)
        self.assertIn(self.a, symbols)


class TestLiterals(unittest.TestCase):
    """Test literal creation and manipulation."""
    
    def setUp(self):
        """Set up common symbols."""
        self.P = Predicate("P", 1)
        self.Q = Predicate("Q", 2)
        self.x = Variable("X")
        self.a = Constant("a")
    
    def test_positive_literal(self):
        """Test creating positive literals."""
        # P(a)
        pa = Literal(self.P(self.a), True)
        self.assertTrue(pa.polarity)
        self.assertEqual(pa.predicate.symbol, self.P)
        self.assertEqual(pa.predicate.args[0], self.a)
    
    def test_negative_literal(self):
        """Test creating negative literals."""
        # ~P(X)
        not_px = Literal(self.P(self.x), False)
        self.assertFalse(not_px.polarity)
        self.assertEqual(not_px.predicate.symbol, self.P)
        self.assertEqual(not_px.predicate.args[0], self.x)
    
    def test_literal_string_representation(self):
        """Test literal string representation."""
        pa = Literal(self.P(self.a), True)
        not_px = Literal(self.P(self.x), False)
        
        self.assertEqual(str(pa), "P(a)")
        self.assertEqual(str(not_px), "~P(X)")
    
    def test_literal_comparison(self):
        """Test literal equality and hashing."""
        l1 = Literal(self.P(self.a), True)
        l2 = Literal(self.P(self.a), True)
        l3 = Literal(self.P(self.a), False)
        
        self.assertEqual(l1, l2)
        self.assertNotEqual(l1, l3)
        self.assertEqual(hash(l1), hash(l2))


class TestClauses(unittest.TestCase):
    """Test clause creation and manipulation."""
    
    def setUp(self):
        """Set up common symbols and literals."""
        self.P = Predicate("P", 1)
        self.Q = Predicate("Q", 2)
        self.x = Variable("X")
        self.y = Variable("Y")
        self.a = Constant("a")
        
        self.px = Literal(self.P(self.x), True)
        self.not_py = Literal(self.P(self.y), False)
        self.qxy = Literal(self.Q(self.x, self.y), True)
    
    def test_empty_clause(self):
        """Test creating empty clause (contradiction)."""
        empty = Clause()
        self.assertEqual(len(empty.literals), 0)
        self.assertTrue(isinstance(empty.literals, tuple))
    
    def test_unit_clause(self):
        """Test creating unit clauses."""
        unit = Clause(self.px)
        self.assertEqual(len(unit.literals), 1)
        self.assertEqual(unit.literals[0], self.px)
    
    def test_binary_clause(self):
        """Test creating binary clauses."""
        binary = Clause(self.px, self.not_py)
        self.assertEqual(len(binary.literals), 2)
        self.assertIn(self.px, binary.literals)
        self.assertIn(self.not_py, binary.literals)
    
    def test_clause_methods(self):
        """Test clause methods."""
        clause = Clause(self.px, self.qxy)
        
        # Test variables
        vars = clause.variables()
        self.assertEqual(len(vars), 2)
        self.assertIn(self.x, vars)
        self.assertIn(self.y, vars)
        
        # Test predicates
        preds = clause.predicate_symbols()
        self.assertEqual(len(preds), 2)
        self.assertIn(self.P, preds)
        self.assertIn(self.Q, preds)
    
    def test_clause_string_representation(self):
        """Test clause string representation."""
        empty = Clause()
        unit = Clause(self.px)
        binary = Clause(self.px, self.not_py)
        
        self.assertEqual(str(empty), "")
        self.assertEqual(str(unit), "P(X)")
        self.assertEqual(str(binary), "P(X) | ~P(Y)")


class TestProblem(unittest.TestCase):
    """Test Problem class."""
    
    def setUp(self):
        """Create a simple problem."""
        P = Predicate("P", 1)
        Q = Predicate("Q", 1)
        a = Constant("a")
        b = Constant("b")
        x = Variable("X")
        
        # Create clauses for: P(a), ~P(X) | Q(X), ~Q(b)
        self.c1 = Clause(Literal(P(a), True))
        self.c2 = Clause(Literal(P(x), False), Literal(Q(x), True))
        self.c3 = Clause(Literal(Q(b), False))
        
        self.problem = Problem(self.c1, self.c2, self.c3)
    
    def test_problem_creation(self):
        """Test creating problems."""
        self.assertEqual(len(self.problem.clauses), 3)
        self.assertEqual(self.problem.clauses[0], self.c1)
        self.assertEqual(self.problem.clauses[1], self.c2)
        self.assertEqual(self.problem.clauses[2], self.c3)
    
    def test_problem_symbols(self):
        """Test extracting symbols from problem."""
        # Predicates
        preds = self.problem.predicate_symbols()
        pred_names = {p.name for p in preds}
        self.assertEqual(pred_names, {"P", "Q"})
        
        # Functions (includes constants)
        funcs = self.problem.function_symbols()
        func_names = {f.name for f in funcs}
        self.assertEqual(func_names, {"a", "b"})
        
        # Variables
        vars = self.problem.variables()
        var_names = {v.name for v in vars}
        self.assertEqual(var_names, {"X"})
    
    def test_problem_string_representation(self):
        """Test problem string representation."""
        problem_str = str(self.problem)
        self.assertIn("P(a)", problem_str)
        self.assertIn("~P(X) | Q(X)", problem_str)
        self.assertIn("~Q(b)", problem_str)


class TestContext(unittest.TestCase):
    """Test global context functionality."""
    
    def test_context_reset(self):
        """Test resetting the global context."""
        # Create some symbols
        P = Predicate("P", 1)
        f = Function("f", 1)
        
        # Reset context
        reset_context()
        
        # Check that context is empty
        self.assertEqual(len(get_predicates()), 0)
        self.assertEqual(len(get_functions()), 0)
    
    def test_context_tracking(self):
        """Test that context tracks created symbols."""
        reset_context()
        
        # Create symbols
        P = Predicate("P", 1)
        Q = Predicate("Q", 2)
        f = Function("f", 1)
        a = Constant("a")
        
        # Check they're tracked
        preds = get_predicates()
        funcs = get_functions()
        
        self.assertEqual(len(preds), 2)
        self.assertEqual(len(funcs), 2)  # f and a


if __name__ == '__main__':
    unittest.main()