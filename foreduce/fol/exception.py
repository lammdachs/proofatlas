class NotFunctionError(Exception):
    def __init__(self, notfunction):
        super().__init__(f"Expected function, got {notfunction}")
        self.notfunction = notfunction


class NotTermError(Exception):
    def __init__(self, notterm):
        super().__init__(f"Expected term, got {notterm}")
        self.notterm = notterm


class NotAtomError(Exception):
    def __init__(self, notatom):
        super().__init__(f"Expected atom, got {notatom}")
        self.notatom = notatom


class NotPredicateError(Exception):
    def __init__(self, notpredicate):
        super().__init__(f"Expected predicate, got {notpredicate}")
        self.notpredicate = notpredicate


class NotLiteralError(Exception):
    def __init__(self, notliteral):
        super().__init__(f"Expected literal, got {notliteral}")
        self.notliteral = notliteral


class NotClauseError(Exception):
    def __init__(self, notclause):
        super().__init__(f"Expected clause, got {notclause}")
        self.notclause = notclause


class ArguementNumberError(Exception):
    def __init__(self, expected, got):
        super().__init__(f"Expected {expected} arguments, got {got}")
        self.expected = expected
        self.got = got


class PrdicateInsteadOfTermError(Exception):
    def __init__(self, predicate):
        super().__init__(f"Expected term, got predicate {predicate}")
        self.predicate = predicate


class TermInsteadOfPredicateError(Exception):
    def __init__(self, term):
        super().__init__(f"Expected predicate, got term {term}")
        self.term = term
