from foreduce.transformer.tokenizer import TokenConfig


class _Context:
    def __init__(self):
        self.predicates = set()
        self.functions = set()
        self.variables = set()


_context = _Context()


def reset_context():
    global _context
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
        if isinstance(self, Constant) or isinstance(self, Variable):
            return self
        return Term(self, *args)


class Function(_Symbol):
    def __init__(self, name, arity):
        super().__init__(name, arity)
        global _context
        _context.functions.add(self)

    def __eq__(self, other):
        if not isinstance(other, Function):
            return False
        return self.name == other.name and \
            self.arity == other.arity

    def __hash__(self):
        return hash((self.name, self.arity))


class Predicate(_Symbol):
    def __init__(self, name, arity):
        super().__init__(name, arity)
        global _context
        _context.predicates.add(self)

    def __eq__(self, other):
        return self.name == other.name and \
            self.arity == other.arity

    def __hash__(self):
        return hash((self.name, self.arity))

_EQ = Predicate('eq', 2)
_TRUE = Predicate('$true', 0)
_FALSE = Predicate('$false', 0)
_PREDEFINED = {_EQ, _TRUE, _FALSE}

class Term:
    @staticmethod
    def check(symbol, *args):
        if not isinstance(symbol, _Symbol):
            raise TypeError(f"Expected Symbol, got {symbol}")
        if len(args) != symbol.arity:
            raise TypeError(f"Expected {symbol.arity} arguments, got {len(args)}")
        for arg in args:
            if not isinstance(arg, Term):
                raise TypeError(f"Expected Term, got {arg}")
            if isinstance(arg, Predicate):
                raise TypeError(f"Expected Term, got Predicate {arg}")

    def __init__(self, symbol, *args):
        self.symbol = symbol
        self.args = args

    def __hash__(self):
        return hash((self.symbol, *self.args))

    def __eq__(self, other):
        if not isinstance(other, Term):
            return False
        if isinstance(self, Variable):
            if not isinstance(other, Variable):
                return False
            else:
                return self.name == other.name
        if isinstance(self, Constant):
            if not isinstance(other, Constant):
                return False
            else:
                return self.name == other.name
        return self.symbol == other.symbol and \
            self.args == other.args

    def __repr__(self):
        if self.symbol.arity == 0:
            return self.symbol.name
        elif self.symbol == _EQ:
            return f"{self.args[0]} = {self.args[1]}"
        else:
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

    def encode(self, mapping):
        result = [mapping[self.symbol]]
        for arg in self.args:
            result += arg.encode(mapping)
        return result

    def substitute(self, t1, t2):
        if self == t1:
            return t2
        return self.symbol(*[arg.substitute(t1, t2) for arg in self.args])

    def tokenize(self, mapping):
        result = [mapping[self.symbol.name]]
        for arg in self.args:
            result += arg.tokenize(mapping)
        return result

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
        _context.variables.add(self)

    def __repr__(self):
        return self.name


class Literal(Term):
    @staticmethod
    def check(predicate, polarity):
        if not isinstance(predicate.symbol, Predicate):
            raise TypeError(f"Expected Predicate, got {predicate.symbol}")
        if not isinstance(polarity, bool):
            raise TypeError(f"Expected bool, got {polarity}")

    def __init__(self, predicate, polarity=True):
        Literal.check(predicate, polarity)
        self.predicate = predicate
        self.polarity = polarity
        self.symbol = predicate.symbol
        self.args = predicate.args

    def __repr__(self):
        return f"{'~' if not self.polarity else ''}{self.predicate}"

    def __eq__(self, other):
        if not isinstance(other, Literal):
            return False
        return self.predicate == other.predicate and \
            self.polarity == other.polarity

    def __hash__(self):
        return hash((self.predicate, self.polarity))

    def substitute(self, t1, t2):
        return Literal(self.predicate.substitute(t1, t2), self.polarity)

    def tokenize(self, mapping):
        result = [mapping["~"]] if not self.polarity else []
        result += self.predicate.tokenize(mapping)
        return result


class Clause:
    @staticmethod
    def check(literals):
        for literal in literals:
            if not isinstance(literal, Literal):
                raise TypeError(f"Expected Literal, got {literal}")

    def __init__(self, *literals):
        Clause.check(literals)
        self.literals = tuple(literals)

    def __repr__(self):
        return f"{' | '.join(map(str, self.literals))}"

    def __eq__(self, other):
        if not isinstance(other, Clause):
            return False
        return self.literals == other.literals

    def __hash__(self):
        return hash(tuple(self.literals))

    def predicate_symbols(self):
        symbols = set()
        for literal in self.literals:
            symbols.add(literal.predicate.symbol)
        for predicate in _PREDEFINED:
            symbols.discard(predicate)
        return symbols

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

    def substitute(self, t1, t2):
        return Clause(*[literal.substitute(t1, t2) for literal in self.literals])

    def to_tptp(self, i=0):
        return f"cnf({i}, axiom, {self})."

    def tokenize(self, config, mapping):
        mapping = mapping | config.variable_mapping([var.name for var in self.variables()])
        result = [config.reserved_token_mapping["<START>"]]
        for literal in self.literals:
            result += literal.tokenize(mapping)
            result.append(mapping["|"])
        result[-1] = config.reserved_token_mapping["<END>"]
        return result


class Problem:
    @staticmethod
    def check(clauses):
        for clause in clauses:
            if not isinstance(clause, Clause):
                raise TypeError(f"Expected Clause, got {clause}")

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

    def to_tptp(self):
        result = []
        for i, clause in enumerate(self.clauses):
            result.append(clause.to_tptp(i))
        return '\n'.join(result)

    def random_mapping(self, config=TokenConfig()):
        symbols = [[] for _ in config.num_functions]
        for function in self.function_symbols() | self.predicate_symbols():
            symbols[function.arity].append(function.name)
        return dict(config.reserved_token_mapping) | dict(config.random_function_mapping(symbols))  

    def tokenize(self, config=TokenConfig(), mapping=None, limit=-1):
        if mapping is None:
            mapping = self.random_mapping(config)
        result = []
        for clause in self.clauses[:limit]:
            result.append(clause.tokenize(config, mapping))
        return result, mapping
