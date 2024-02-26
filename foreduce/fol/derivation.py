from foreduce.fol.logic import Clause, Literal, Predicate, Term, eq, Variable


class DerivedClause(Clause):
    def leaves(self):
        result = []
        for child in self.children:
            result += child.leaves()
        return result

    def nodes(self):
        result = {self}
        for child in self.children:
            result.update(child.nodes())
        return result

    def __eq__(self, other):
        if not isinstance(other, DerivedClause):
            return False
        return self.literals == other.literals and \
            self.children == other.children

    def __hash__(self):
        return hash((
            tuple(self.literals),
            self.children
        ))

    def substitution_proof(self):
        axioms = []
        substitutions = []
        for axiom in self.leaves():
            axioms.append(axiom)
            substitution = axiom
            for k, v in axiom.verifying_substitution.items():
                substitution = substitution.substitute(k, v)
            substitutions.append(substitution)
        return tuple(axioms), tuple(substitutions)

    def potential_steps(self):
        return self._potential_steps(self.leaves())

    def _potential_steps(self, axioms):
        potential_next = []
        for node in self.nodes():
            if all([child in axioms for child in node.children]):
                potential_next.append(node)


class Axiom(DerivedClause):
    def __init__(self, c, verifying_substitution=dict()):
        if not isinstance(c, Clause):
            raise TypeError(f"Expected Clause, got {c}")
        self.children = tuple()
        self.literals = c.literals
        self.verifying_substitution = verifying_substitution

    def leaves(self):
        return [self]


class Substitution(DerivedClause):
    @staticmethod
    def check(c, var, term, verifying_substitution=dict()):
        if not isinstance(c, DerivedClause):
            raise TypeError(f"Expected Clause, got {c}")
        if not isinstance(var, Variable):
            raise TypeError(f"Expected Variable, got {var}")
        if not isinstance(term, Term):
            raise TypeError(f"Expected Term, got {term}")

    def __init__(self, c, var, term, verifying_substitution=dict()):
        Substitution.check(c, var, term)
        self.children = (c,)
        self.var = var
        self.term = term
        self.literals = list(set([
            l.substitute(var, term)
            for l in c.literals
        ]))
        self.verifying_substitution = verifying_substitution
        for k, v in self.verifying_substitution.items():
            term.substitute(k, v)
        self.children[0].verifying_substitution = {
            k: v for k, v in self.verifying_substitution.items()
        }
        self.children[0].verifying_substitution[var] = term


class Resolution(DerivedClause):
    # Left clause contains positive clause instance
    @staticmethod
    def check(c1, c2, p):
        if not isinstance(c1, DerivedClause):
            raise TypeError(f"Expected Clause, got {c1}")
        if not isinstance(c2, DerivedClause):
            raise TypeError(f"Expected Clause, got {c2}")
        if not isinstance(p.symbol, Predicate):
            raise TypeError(f"Expected Predicate, got {p}")
        if not (Literal(p, True) in c1.literals and Literal(p, False) in c2.literals):
            raise TypeError(
                f"Expected {p} to be complementary in {c1} and {c2}")

    def __init__(self, c1, c2, p, verifying_substitution=dict()):
        Resolution.check(c1, c2, p)
        self.children = (c1, c2)
        self.p = p
        self.literals = list(set([
            l for l in c1.literals if l.predicate != p
        ] + [
            l for l in c2.literals if l.predicate != p
        ]))
        self.verifying_substitution = verifying_substitution
        for child in self.children:
            child.verifying_substitution = self.verifying_substitution


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
        if not Literal(equality, True) in c1.literals:
            raise TypeError(f"Expected {equality} to be in {c1}")

    def __init__(self, c1, c2, equality, left_to_right=True, verifying_substitution=dict()):
        Superposition.check(c1, c2, equality, left_to_right)
        self.children = (c1, c2)
        self.equality = equality
        self.literals = list(set([
            l for l in c1.literals if l != equality
        ] + [
            l.substitute(equality.args[0], equality.args[1])
            if left_to_right else
            l.substitute(equality.args[1], equality.args[0])
            for l in c2.literals
        ]))
        self.verifying_substitution = verifying_substitution
        for child in self.children:
            child.verifying_substitution = self.verifying_substitution
