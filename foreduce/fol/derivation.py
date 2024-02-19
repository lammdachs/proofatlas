from foreduce.fol.logic import Clause, Literal, Predicate, Term, eq, Variable


class DerivedClause(Clause):
    def leaves(self):
        result = set()
        for child in self.children:
            result.update(child.leaves())

    def __eq__(self, other):
        if not isinstance(other, DerivedClause):
            return False
        return self.literals == other.literals and \
            self.children == other.children

    def __hash__(self):
        return hash((self.literals, self.children))


class Axiom(DerivedClause):
    def __init__(self, c):
        if not isinstance(c, Clause):
            raise TypeError(f"Expected Clause, got {c}")
        self.children = tuple()
        self.literals = c.literals

    def leaves(self):
        return {self}


class Substitution(DerivedClause):
    @staticmethod
    def check(c, var, term):
        if not isinstance(c, DerivedClause):
            raise TypeError(f"Expected Clause, got {c}")
        if not isinstance(var, Variable):
            raise TypeError(f"Expected Variable, got {var}")
        if not isinstance(term, Term):
            raise TypeError(f"Expected Term, got {term}")

    def __init__(self, c, var, term):
        Substitution.check(c, var, term)
        self.children = (c,)
        self.var = var
        self.term = term
        self.literals = list(set([
            l.substitute(var, term)
            for l in c.literals
        ]))
        for l in self.literals:
            if l.polarity:
                continue
            if l.predicate != eq:
                continue
            if l.args[0] != l.args[1]:
                continue
            self.literals.remove(l)


class Resolution(DerivedClause):
    @staticmethod
    def check(c1, c2, p):
        if not isinstance(c1, DerivedClause):
            raise TypeError(f"Expected Clause, got {c1}")
        if not isinstance(c2, DerivedClause):
            raise TypeError(f"Expected Clause, got {c2}")
        if not isinstance(p, Predicate):
            raise TypeError(f"Expected Predicate, got {p}")
        if not ((Literal(p, True) in c1.literals and not Literal(p, False) in c2.clause.literals) or
                (Literal(p, False) in c1.literals and not Literal(p, True) in c2.clause.literals)):
            raise TypeError(
                f"Expected {p} to be complementary in {c1} and {c2}")

    def __init__(self, c1, c2, p):
        Resolution.check(c1, c2, p)
        self.children = (c1, c2)
        self.p = p
        self.literals = list(set([
            l for l in c1.literals if l.predicate != p
        ] + [
            l for l in c2.literals if l.predicate != p
        ]))


class Superposition(DerivedClause):
    @staticmethod
    def check(c1, c2, equality, left_to_right=True):
        if not isinstance(c1, DerivedClause):
            raise TypeError(f"Expected Clause, got {c1}")
        if not isinstance(c2, DerivedClause):
            raise TypeError(f"Expected Clause, got {c2}")
        if equality.symbol != eq:
            raise TypeError(f"Expected equality, got {equality}")
        if not left_to_right in (True, False):
            raise TypeError(f"Expected bool, got {left_to_right}")
        if not equality in c1.literals:
            raise TypeError(f"Expected {equality} to be in {c1}")

    def __init__(self, c1, c2, equality, left_to_right=True):
        Superposition.check(c1, c2, equality, left_to_right)
        self.children = (c1, c2)
        self.equality = equality
        self.literals = list(set([
            l for l in c1.literals if l != equality
        ] + [
            l.substitute(equality.predicate.args[0], equality.predicate.args[1])
            if left_to_right else
            l.substitute(equality.predicate.args[1], equality.predicate.args[0])
            for l in c2.literals
        ]))
