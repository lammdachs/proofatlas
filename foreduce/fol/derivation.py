from foreduce.fol.logic import Clause, Literal, Predicate, Term, eq, Variable
from itertools import chain


class Derivation(Clause):
    def children_rec(self, n=None):
        if n <= 0:
            return [self]
        else:
            return [self] + list(chain(*[child.children_rec(n - 1) for child in self.children]))


class Axiom(Derivation):
    def __init__(self, c):
        if not isinstance(c, Clause):
            raise TypeError(f"Expected Clause, got {c}")
        self.children = tuple()
        self.literals = c.literals


class DerivedClause(Derivation):
    def __init__(self, children, literals):
        self.children = children
        self.literals = literals

