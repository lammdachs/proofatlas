import networkx as nx

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
_PREDEFINED = {_EQ, _FALSE}


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

    def depth(self):
        if self.args:
            return 1 + max(arg.depth() for arg in self.args)
        return 0
    
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

    def to_graph(self, graph, mapping, pos, clause, depth=None):
        if repr(self.symbol) not in mapping:
            mapping[repr(self.symbol)] = len(graph.nodes)
            graph.add_node(mapping[repr(self.symbol)], type='symbol', arity=self.symbol.arity, pos=None)
        mapping[repr(self) + repr(clause)] = len(graph.nodes)
        graph.add_node(mapping[repr(self) + repr(clause)], type='term', arity=None, pos=pos)
        graph.add_edge(mapping[repr(self) + repr(clause)], mapping[repr(self.symbol)])
        if depth is None or depth > 0:
            for i, arg in enumerate(self.args):
                mapping = arg.to_graph(graph, mapping, pos=i, clause=clause, depth=None if depth is None else depth-1)
                graph.add_edge(mapping[repr(self) + repr(clause)], mapping[repr(arg) + repr(clause)])
        return mapping


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
    
    def to_graph(self, graph, mapping, pos, clause, depth=None):
        if repr(self) + repr(clause) not in mapping:
            mapping[repr(self) + repr(clause)] = len(graph.nodes)
            graph.add_node(mapping[repr(self) + repr(clause)], type='variable', arity=None, pos=pos)
        return mapping


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

    def depth(self):
        return self.predicate.depth()

    def substitute(self, t1, t2):
        return Literal(self.predicate.substitute(t1, t2), self.polarity)

    def tokenize(self, mapping):
        result = [mapping["~"]] if not self.polarity else []
        result += self.predicate.tokenize(mapping)
        return result

    def to_graph(self, graph, mapping, clause, depth=None):
        mapping[repr(self) + repr(clause)] = len(graph.nodes)
        graph.add_node(len(graph.nodes), type='literal' if self.polarity else 'negated_literal', arity=None, pos=None)
        mapping = self.predicate.to_graph(graph, mapping, pos=None, clause=clause, depth=depth)
        graph.add_edge(mapping[repr(self) + repr(clause)], mapping[repr(self.predicate) + repr(clause)])
        return mapping


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

    def depth(self):
        return max(literal.predicate.depth() for literal in self.literals)

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

    def to_graph(self, graph, mapping, depth=None):
        if repr(self) in mapping:
            return mapping
        mapping[repr(self)] = len(graph.nodes)
        graph.add_node(len(graph.nodes), type='clause', arity=None, pos=None)
        if not self.literals:
            graph.add_edge(mapping[repr(self)], mapping[repr(_FALSE)])
        for literal in self.literals:
            mapping = literal.to_graph(graph, mapping, self, depth=depth)
            graph.add_edge(mapping[repr(self)], mapping[repr(literal) + repr(self)])
        return mapping


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

    def depth(self):
        return max((clause.depth() for clause in self.clauses), default=0)

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

    def tokenize(self, config=TokenConfig(), mapping=None, limit=None):
        if mapping is None:
            mapping = self.random_mapping(config)
        if limit is None:
            limit = len(self.clauses)
        result = []
        for clause in self.clauses[:limit]:
            result.append(clause.tokenize(config, mapping))
        return result, mapping
    
    def to_graph(self, limit=None, depth=None):
        graph = nx.Graph()
        mapping = {}
        for predefined in _PREDEFINED:
            mapping[repr(predefined)] = len(graph.nodes)
            graph.add_node(len(graph.nodes), type=f'{predefined.name}', arity=predefined.arity, pos=None)
        if limit is None:
            limit = len(self.clauses)
        for idx, clause in enumerate(self.clauses[:limit]):
            mapping = clause.to_graph(graph, mapping, depth=depth)
        return graph, mapping, sorted(mapping[r] for r in set(repr(clause) for clause in self.clauses[:limit]))
    
    def extend_graph(self, graph, mapping, prev_limit, limit=None, depth=8):
        if limit is None:
            limit = len(self.clauses)
        for idx, clause in enumerate(self.clauses[prev_limit:limit]):
            mapping = clause.to_graph(graph, mapping, depth=depth)
        return graph, mapping, sorted(mapping[r] for r in set(repr(clause) for clause in self.clauses[:limit]))
    
    def to_graph_data(self, tree, limit=None, depth=8):
        if limit is None:
            limit = len(self.clauses)
        graph, mapping, _ = self.to_graph(limit, depth=8)
        dependencies = [set() for _ in self.clauses]
        for idx in range(len(self.clauses)):
            if tree[idx]:
                dependencies[idx] = {idx} | set.union(*[dependencies[j] for j in tree[idx]])
            else:
                dependencies[idx] = {idx}
        labels =  {repr(clause): False for clause in self.clauses[:limit]}
        for idx in range(limit):
            if idx in dependencies[-1]:
                labels[repr(self.clauses[idx])] = True
        clauses, labels = zip(*[(mapping[r], labels[r]) for r in sorted(labels.keys())])
        return graph, mapping, list(clauses), list(labels)
