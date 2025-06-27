"""Tests for unification algorithm."""

import unittest

from proofatlas.core.logic import Variable, Constant, Function, Predicate, Term
from proofatlas.core.unification import unify_terms, Substitution, occurs_check, rename_variables


class TestUnification(unittest.TestCase):
    """Test cases for unification."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Variables
        self.X = Variable('X')
        self.Y = Variable('Y')
        self.Z = Variable('Z')
        
        # Constants
        self.a = Constant('a')
        self.b = Constant('b')
        self.socrates = Constant('socrates')
        
        # Functions
        self.f = Function('f', 1)
        self.g = Function('g', 2)
        
        # Predicates
        self.P = Predicate('P', 1)
        self.Human = Predicate('Human', 1)
        self.Mortal = Predicate('Mortal', 1)
    
    def test_unify_identical_constants(self):
        """Test unifying identical constants."""
        result = unify_terms(self.a, self.a)
        self.assertIsNotNone(result)
        self.assertEqual(len(result.mapping), 0)
    
    def test_unify_different_constants(self):
        """Test unifying different constants fails."""
        result = unify_terms(self.a, self.b)
        self.assertIsNone(result)
    
    def test_unify_variable_with_constant(self):
        """Test unifying variable with constant."""
        result = unify_terms(self.X, self.a)
        self.assertIsNotNone(result)
        self.assertEqual(result.mapping[self.X], self.a)
    
    def test_unify_constant_with_variable(self):
        """Test unifying constant with variable."""
        result = unify_terms(self.a, self.X)
        self.assertIsNotNone(result)
        self.assertEqual(result.mapping[self.X], self.a)
    
    def test_unify_two_variables(self):
        """Test unifying two variables."""
        result = unify_terms(self.X, self.Y)
        self.assertIsNotNone(result)
        # Either X -> Y or Y -> X is valid
        self.assertTrue(self.X in result.mapping or self.Y in result.mapping)
    
    def test_unify_function_terms(self):
        """Test unifying function terms."""
        # f(a) with f(X)
        f_a = self.f(self.a)
        f_X = self.f(self.X)
        
        result = unify_terms(f_a, f_X)
        self.assertIsNotNone(result)
        self.assertEqual(result.mapping[self.X], self.a)
    
    def test_unify_nested_functions(self):
        """Test unifying nested function terms."""
        # g(X, f(Y)) with g(a, f(b))
        term1 = self.g(self.X, self.f(self.Y))
        term2 = self.g(self.a, self.f(self.b))
        
        result = unify_terms(term1, term2)
        self.assertIsNotNone(result)
        self.assertEqual(result.mapping[self.X], self.a)
        self.assertEqual(result.mapping[self.Y], self.b)
    
    def test_unify_predicate_terms(self):
        """Test unifying predicate terms (for resolution)."""
        # Human(X) with Human(socrates)
        human_X = self.Human(self.X)
        human_socrates = self.Human(self.socrates)
        
        result = unify_terms(human_X, human_socrates)
        self.assertIsNotNone(result)
        self.assertEqual(result.mapping[self.X], self.socrates)
    
    def test_occurs_check(self):
        """Test occurs check prevents infinite structures."""
        # Try to unify X with f(X)
        f_X = self.f(self.X)
        
        # Direct occurs check
        self.assertTrue(occurs_check(self.X, f_X))
        self.assertFalse(occurs_check(self.X, self.a))
        
        # Unification should fail due to occurs check
        result = unify_terms(self.X, f_X)
        self.assertIsNone(result)
    
    def test_substitution_apply(self):
        """Test applying substitutions."""
        # Create substitution {X -> a, Y -> b}
        subst = Substitution({self.X: self.a, self.Y: self.b})
        
        # Apply to variable
        self.assertEqual(subst.apply(self.X), self.a)
        self.assertEqual(subst.apply(self.Y), self.b)
        self.assertEqual(subst.apply(self.Z), self.Z)  # Unchanged
        
        # Apply to constant
        self.assertEqual(subst.apply(self.a), self.a)
        
        # Apply to function term f(X)
        f_X = self.f(self.X)
        result = subst.apply(f_X)
        self.assertEqual(str(result), "f(a)")
        
        # Apply to nested term g(X, f(Y))
        nested = self.g(self.X, self.f(self.Y))
        result = subst.apply(nested)
        self.assertEqual(str(result), "g(a, f(b))")
    
    def test_substitution_compose(self):
        """Test composing substitutions."""
        # σ1 = {X -> Y}
        s1 = Substitution({self.X: self.Y})
        # σ2 = {Y -> a}
        s2 = Substitution({self.Y: self.a})
        
        # σ1 ∘ σ2 should give {X -> a, Y -> a}
        composed = s1.compose(s2)
        self.assertEqual(composed.apply(self.X), self.a)
        self.assertEqual(composed.apply(self.Y), self.a)
    
    def test_rename_variables(self):
        """Test renaming variables in terms."""
        # Rename variables in f(X)
        f_X = self.f(self.X)
        renamed = rename_variables(f_X, "_1")
        # Should be f(X_1)
        self.assertNotEqual(str(renamed), str(f_X))
        self.assertIn("X_1", str(renamed))
        
        # Rename in predicate term Human(X)
        human_X = self.Human(self.X)
        renamed = rename_variables(human_X, "_c2")
        self.assertIn("X_c2", str(renamed))
    
    def test_unify_with_multiple_occurrences(self):
        """Test unifying terms with multiple variable occurrences."""
        # Create a binary predicate
        P2 = Predicate('P', 2)
        
        # P(X, X) with P(a, b) should fail
        p_xx = P2(self.X, self.X)
        p_ab = P2(self.a, self.b)
        
        result = unify_terms(p_xx, p_ab)
        self.assertIsNone(result)  # Can't unify X to both a and b
        
        # P(X, X) with P(a, a) should succeed
        p_aa = P2(self.a, self.a)
        result = unify_terms(p_xx, p_aa)
        self.assertIsNotNone(result)
        self.assertEqual(result.mapping[self.X], self.a)


if __name__ == '__main__':
    unittest.main()