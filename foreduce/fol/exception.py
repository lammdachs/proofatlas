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


class FunctionArityError(Exception):
    def __init__(self, arity, config):
        super().__init__(f"Expected function arity between 0 and {
            len(config.function_arity)}, got {arity}")
        self.arity = arity
        self.config = config


class FunctionCountError(Exception):
    def __init__(self, count, arity, config):
        super().__init__(f"Expected at most {config.function_arity[arity]} \
                          functions with arity {arity}, got {count}")
        self.count = count
        self.arity = arity
        self.config = config


class PredicateArityError(Exception):
    def __init__(self, arity, config):
        super().__init__(f"Expected predicate arity between 0 and {
            len(config.predicate_arity)}, got {arity}")
        self.arity = arity
        self.config = config


class PredicateCountError(Exception):
    def __init__(self, count, arity, config):
        super().__init__(f"Expected at most {config.predicate_arity[arity]} \
                         predicates with arity {arity}, got {count}")
        self.count = count
        self.arity = arity
        self.config = config


class VariableCountError(Exception):
    def __init__(self, count, config):
        super().__init__(f"Expected at most {
            config.variable_count} variables, got {count}")
        self.count = count
        self.config = config
