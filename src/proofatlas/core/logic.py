from functools import cache
import networkx as nx


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
        if not hasattr(self, 'hash'):
            self.hash = hash((self.name, self.arity))
        return self.hash

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
        if not hasattr(self, 'hash'):
            self.hash = hash((self.name, self.arity))
        return self.hash


class Predicate(_Symbol):
    def __init__(self, name, arity):
        super().__init__(name, arity)
        global _context
        _context.predicates.add(self)

    def __eq__(self, other):
        return self.name == other.name and \
            self.arity == other.arity
            
    def __hash__(self):
        if not hasattr(self, 'hash'):
            self.hash = hash((self.name, self.arity))
        return self.hash


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
        if not hasattr(self, 'hash'):
            self.hash = hash((self.symbol, *self.args))
        return self.hash

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

    def to_graph(self, graph, mapping, pos, prefix, clause, depth=None):
        mapping[(prefix + 'symbol', clause)] = len(graph.nodes)
        if self.symbol in _PREDEFINED:
            graph.add_node(len(graph.nodes), type=self.symbol, name=None, arity=self.symbol.arity, pos=None)
        else:
            graph.add_node(mapping[(prefix + 'symbol', clause)], type='symbol', name=self.symbol, arity=self.symbol.arity, pos=None)
        mapping[(prefix, clause)] = len(graph.nodes)
        graph.add_node(mapping[(prefix, clause)], type='term', name=None, arity=None, pos=pos)
        graph.add_edge(mapping[(prefix, clause)], mapping[(prefix + 'symbol', clause)])
        if depth is None or depth > 0:
            for i, arg in enumerate(self.args):
                mapping = arg.to_graph(graph, mapping, i, prefix + repr(i), clause=clause, depth=None if depth is None else depth-1)
                if isinstance(arg, Variable):
                    graph.add_edge(mapping[(prefix, clause)], mapping[(arg, clause)])
                else:
                    graph.add_edge(mapping[(prefix, clause)], mapping[(prefix + repr(i), clause)])
        else:
            for i, _ in enumerate(self.args):
                graph.add_node(len(graph.nodes), type='placeholder', name=None, arity=None, pos=i)
                graph.add_edge(mapping[(prefix, clause)], len(graph.nodes)-1)
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
    
    def to_graph(self, graph, mapping, pos, prefix, clause, depth=None):
        if repr(self) + repr(clause) not in mapping:
            mapping[(self, clause)] = len(graph.nodes)
            graph.add_node(mapping[(self, clause)], type='variable', name=None, arity=None, pos=pos)
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

    def __repr__(self):
        return f"{'~' if not self.polarity else ''}{self.predicate}"

    def __eq__(self, other):
        if not isinstance(other, Literal):
            return False
        return self.predicate == other.predicate and \
            self.polarity == other.polarity

    def __hash__(self):
        if not hasattr(self, 'hash'):
            self.hash = hash((self.predicate, self.polarity))
        return self.hash

    def depth(self):
        return self.predicate.depth()

    def substitute(self, t1, t2):
        return Literal(self.predicate.substitute(t1, t2), self.polarity)

    def tokenize(self, mapping):
        result = [mapping["~"]] if not self.polarity else []
        result += self.predicate.tokenize(mapping)
        return result

    def to_graph(self, graph, mapping, clause, depth=None):
        mapping[(self, clause)] = len(graph.nodes)
        graph.add_node(len(graph.nodes), type='literal' if self.polarity else 'negated_literal', name=None, arity=None, pos=None)
        mapping = self.predicate.to_graph(graph, mapping, pos=None, prefix='0', clause=clause, depth=depth)
        graph.add_edge(mapping[(self, clause)], mapping[('0', clause)])
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
        if not hasattr(self, 'hash'):
            self.hash = hash(self.literals)
        return self.hash

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


    def to_graph(self, depth=None):
        mapping = {self: 0}
        graph = nx.Graph()
        graph.add_node(len(graph.nodes), type='clause', name=None, arity=None, pos=None)
        if not self.literals:
            graph.add_edge(mapping[self], mapping[_FALSE])
        for literal in self.literals:
            mapping = literal.to_graph(graph, mapping, self, depth=depth)
            graph.add_edge(mapping[self], mapping[(literal, self)])
        return graph


class Problem:
    @staticmethod
    def check(clauses):
        for clause in clauses:
            if not isinstance(clause, Clause):
                raise TypeError(f"Expected Clause, got {clause}")

    def __init__(self, *clauses, conjecture_indices=None):
        Problem.check(clauses)
        self.clauses = clauses
        # Track which clauses are from conjectures (default: empty set)
        self.conjecture_indices = set(conjecture_indices) if conjecture_indices else set()

    def __repr__(self):
        return '\n'.join(map(str, self.clauses))
    
    def is_conjecture_clause(self, index):
        """Check if the clause at the given index is from a conjecture."""
        return index in self.conjecture_indices
    
    def get_conjecture_clauses(self):
        """Get all clauses that came from conjectures."""
        return [(i, self.clauses[i]) for i in self.conjecture_indices if i < len(self.clauses)]

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
    
    def to_graphs(self, limit=None, depth=None):
        graphs = []
        if limit is None:
            limit = len(self.clauses)
        for idx, clause in enumerate(self.clauses[:limit]):
            graph = clause.to_graph(depth=depth)
            graphs.append(graph)
        return graphs
    
    def extend(self, graphs, prev_limit, limit=None, depth=8):
        if limit is None:
            limit = len(self.clauses)
        for idx, clause in enumerate(self.clauses[prev_limit:limit]):
            graph = clause.to_graph(depth=depth)
            graphs.append(graph)
        return graphs
    
    def to_graph_data(self, tree, limit=None, depth=8):
        if limit is None:
            limit = len(self.clauses)
        graphs = self.to_graphs(limit, depth=8)
        dependencies = [set() for _ in self.clauses]
        for idx in range(len(self.clauses)):
            if tree[idx]:
                dependencies[idx] = {idx} | set.union(*[dependencies[j] for j in tree[idx]])
            else:
                dependencies[idx] = {idx}
        labels =  [False for _ in range(limit)]
        for idx in range(limit):
            if idx in dependencies[-1]:
                labels[idx] = True
        return graphs, labels
