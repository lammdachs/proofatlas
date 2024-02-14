from enum import Enum
from .exception import *


class _Context:
    def __init__(self):
        self.predicates = set()
        self.functions = set()
        self.variables = set()


_context = _Context()


def get_predicates():
    global _context
    return set(_context.predicates)


def get_functions():
    global _context
    return set(_context.functions)


class _Symbol:
    def __init__(self, name, arity):
        self.name = name
        self.arity = arity

    def __hash__(self):
        return hash((self.name, self.arity))

    def __repr__(self):
        return f"{self.name}/{self.arity}"

    def __call__(self, *args):
        return Term(self, *args)


class Function(_Symbol):
    def __init__(self, name, arity):
        super().__init__(name, arity)
        global _context
        _context.functions.add(name)


class Predicate(_Symbol):
    def __init__(self, name, arity):
        super().__init__(name, arity)
        global _context
        _context.predicates.add(name)


class Term:
    @staticmethod
    def check(symbol, *args):
        if not isinstance(symbol, _Symbol):
            raise NotFunctionError(symbol)
        if len(args) != symbol.arity:
            raise ArguementNumberError(symbol.arity, len(args))
        for arg in args:
            if not isinstance(arg, Term):
                raise NotTermError(arg)
            if isinstance(arg, Predicate):
                raise PrdicateInsteadOfTermError(arg)

    def __init__(self, symbol, *args):
        self.symbol = symbol
        self.args = args

    def __hash__(self):
        return hash((self.symbol, *self.args))

    def __eq__(self, other):
        return self.symbol == other.symbol and \
            self.args == other.args

    def __repr__(self):
        return f"{self.symbol.name}({', '.join(map(str, self.args))})"

    def function_symbols(self):
        if isinstance(self.symbol, Function):    
            symbols = {self.symbol}
        else:
            symbols = set()
        for arg in self.args:
            symbols |= arg.function_symbols()
        return symbols

    def variables(self):
        if isinstance(self.symbol, Variable):
            variables = {self.symbol}
        else:
            variables = set()
        for arg in self.args:
            variables |= arg.variables()
        return variables

    def terms(self):
        if isinstance(self.symbol, Predicate):    
            terms = set()
        else:
            terms = {self}
        for arg in self.args:
            terms |= arg.terms()
        return terms


class Constant(Function, Term):
    def __init__(self, name):
        Function.__init__(self, name, 0)
        Term.__init__(self, self)

    def __repr__(self):
        return self.name


class Variable(_Symbol, Term):
    def __init__(self, name):
        _Symbol.__init__(self, name, 0)
        Term.__init__(self, self)
        global _context
        _context.variables.add(name)

    def __repr__(self):
        return self.name


eq = Predicate('eq', 2)


class Literal:
    @staticmethod
    def check(predicate, polarity):
        if not isinstance(predicate.symbol, Predicate):
            raise NotPredicateError(predicate)
        if not isinstance(polarity, bool):
            raise TypeError(f"Expected bool, got {polarity}")
    
    def __init__(self, predicate, polarity=True):
        Literal.check(predicate, polarity)
        self.predicate = predicate
        self.polarity = polarity

    def __repr__(self):
        return f"{'¬' if not self.polarity else ''}{self.predicate}"

    def __hash__(self):
        return hash((self.predicate, self.polarity))


class Clause:
    @staticmethod
    def check(literals):
        for literal in literals:
            if not isinstance(literal, Literal):
                raise NotLiteralError(literal)

    def __init__(self, *literals):
        Clause.check(literals)
        self.literals = literals

    def __repr__(self):
        return f"{' | '.join(map(str, self.literals))}"

    def predicate_symbols(self):
        predicates = set()
        for literal in self.literals:
            predicates.add(literal.predicate.symbol)
        return predicates

    def function_symbols(self):
        functions = set()
        for literal in self.literals:
            for arg in literal.predicate.args:
                functions |= arg.function_symbols()
        return functions

    def variables(self):
        variables = set()
        for literal in self.literals:
            for arg in literal.predicate.args:
                variables |= arg.variables()
        return variables

    def terms(self):
        terms = set()
        for literal in self.literals:
            terms |= literal.predicate.terms()
        return terms


class Problem:
    @staticmethod
    def check(clauses):
        for clause in clauses:
            if not isinstance(clause, Clause):
                raise NotClauseError(clause)

    def __init__(self, *clauses):
        Problem.check(clauses)
        self.clauses = clauses

    def __repr__(self):
        return '\n'.join(map(str, self.clauses))

    def predicate_symbols(self):
        predicates = set()
        for clause in self.clauses:
            predicates |= clause.predicate_symbols()
        return predicates

    def function_symbols(self):
        functions = set()
        for clause in self.clauses:
            functions |= clause.function_symbols()
        return functions

    def variables(self):
        variables = set()
        for clause in self.clauses:
            variables |= clause.variables()
        return variables

    def terms(self):
        terms = set()
        for clause in self.clauses:
            terms |= clause.terms()
        return terms
