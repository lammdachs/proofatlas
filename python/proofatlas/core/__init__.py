"""Core theorem proving data structures."""

from .logic import (
    Variable, Constant, Function, Predicate,
    Term, Literal, Clause, Problem
)
from .serialization import (
    problem_to_json, problem_from_json,
    save_problem, load_problem
)
from .unification import (
    Substitution, unify, unify_terms, rename_variables,
    occurs_check
)

__all__ = [
    # Logic
    'Variable', 'Constant', 'Function', 'Predicate',
    'Term', 'Literal', 'Clause', 'Problem',
    # Serialization
    'problem_to_json', 'problem_from_json',
    'save_problem', 'load_problem',
    # Unification
    'Substitution', 'unify', 'unify_terms', 'rename_variables',
    'occurs_check'
]