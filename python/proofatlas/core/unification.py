"""Unification algorithm for first-order logic terms."""

from typing import Dict, Optional, Union, List
from dataclasses import dataclass

from .logic import Term, Variable, Constant, Function


@dataclass
class Substitution:
    """Represents a substitution mapping variables to terms."""
    mapping: Dict[Variable, Term]
    
    def __init__(self, mapping: Optional[Dict[Variable, Term]] = None):
        self.mapping = mapping or {}
    
    def apply(self, term: Term) -> Term:
        """Apply substitution to a term."""
        if isinstance(term, Variable):
            # Apply substitution recursively
            if term in self.mapping:
                return self.apply(self.mapping[term])
            return term
        elif isinstance(term, Constant):
            return term
        elif isinstance(term, Function):
            # Apply to all arguments
            new_args = [self.apply(arg) for arg in term.args]
            return Function(term.symbol, *new_args)
        elif isinstance(term, Term):
            # Handle Term objects (which have symbol and args)
            if hasattr(term, 'args') and term.args:
                new_args = [self.apply(arg) for arg in term.args]
                # Create new Term with same symbol and substituted args
                return Term(term.symbol, *new_args)
            return term
        else:
            return term
    
    def compose(self, other: 'Substitution') -> 'Substitution':
        """Compose this substitution with another."""
        # Apply other to all values in self
        new_mapping = {var: other.apply(term) for var, term in self.mapping.items()}
        
        # Add mappings from other that aren't in self
        for var, term in other.mapping.items():
            if var not in new_mapping:
                new_mapping[var] = term
        
        return Substitution(new_mapping)
    
    def __str__(self):
        if not self.mapping:
            return "{}"
        items = [f"{var} -> {term}" for var, term in self.mapping.items()]
        return "{" + ", ".join(items) + "}"


def occurs_check(var: Variable, term: Term) -> bool:
    """Check if variable occurs in term (prevents infinite structures)."""
    if var == term:
        return True
    if isinstance(term, (Variable, Constant)):
        return False
    if hasattr(term, 'args'):
        return any(occurs_check(var, arg) for arg in term.args)
    return False


def unify_terms(term1: Term, term2: Term) -> Optional[Substitution]:
    """Unify two terms, returning the most general unifier if it exists."""
    return unify(term1, term2, Substitution())


def unify(term1: Term, term2: Term, subst: Substitution) -> Optional[Substitution]:
    """Unify two terms given an existing substitution."""
    # Apply current substitution
    term1 = subst.apply(term1)
    term2 = subst.apply(term2)
    
    # If terms are identical after substitution
    if term1 == term2:
        return subst
    
    # If either is a variable
    if isinstance(term1, Variable):
        if occurs_check(term1, term2):
            return None
        new_mapping = subst.mapping.copy()
        new_mapping[term1] = term2
        return Substitution(new_mapping)
    
    if isinstance(term2, Variable):
        if occurs_check(term2, term1):
            return None
        new_mapping = subst.mapping.copy()
        new_mapping[term2] = term1
        return Substitution(new_mapping)
    
    # If both are constants
    if isinstance(term1, Constant) and isinstance(term2, Constant):
        return subst if term1.name == term2.name else None
    
    # If both are compound terms (Functions or Terms with args)
    if hasattr(term1, 'symbol') and hasattr(term2, 'symbol'):
        # Check if same functor/predicate
        if term1.symbol != term2.symbol:
            return None
        
        # Check if same arity
        args1 = getattr(term1, 'args', [])
        args2 = getattr(term2, 'args', [])
        
        if len(args1) != len(args2):
            return None
        
        # Unify arguments pairwise
        current_subst = subst
        for arg1, arg2 in zip(args1, args2):
            current_subst = unify(arg1, arg2, current_subst)
            if current_subst is None:
                return None
        
        return current_subst
    
    # Otherwise, terms don't unify
    return None


def rename_variables(term: Term, suffix: str) -> Term:
    """Rename all variables in a term by adding a suffix."""
    if isinstance(term, Variable):
        return Variable(term.name + suffix)
    elif isinstance(term, (Constant, str)):
        return term
    elif hasattr(term, 'args') and term.args:
        new_args = [rename_variables(arg, suffix) for arg in term.args]
        if isinstance(term, Function):
            return Function(term.symbol, *new_args)
        else:
            # For Term objects
            return Term(term.symbol, *new_args)
    else:
        return term