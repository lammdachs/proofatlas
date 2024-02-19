import numpy.random as random
from foreduce.fol.logic import Clause, Constant, Function, Literal, Predicate, Variable, eq


class RandomSignature:
    def __init__(self, config):
        self.config = config
        self.functions = {}
        for arity, count in enumerate(config.function_arity):
            if arity == 0:
                self.functions[0] = [
                    Constant(f"c_{i}")
                    for i in range(count)
                ]
                continue
            else:
                self.functions[arity] = [
                    Function(f"f{arity}_{i}", arity)
                    for i in range(count)
                ]
        self.predicates = {}
        for arity, count in enumerate(config.proposition_arity):
            self.predicates[arity] = [
                Predicate(f"p{arity + 1}_{i}", arity + 1)
                for i in range(count)
            ]
        self.variables = [Variable(f"X{i}")
                          for i in range(config.variable_count)]


def RandomTerm(signature, depth=2.0):
    max_arity = len(signature.config.function_arity) - 1
    if max_arity == 0 or depth <= 0:
        arity = 0
    else:
        arity = random.choice(range(max_arity + 1))
    if arity == 0:
        symbol = random.choice(signature.variables + signature.functions[0])
    else:
        symbol = random.choice(signature.functions[arity])
    return symbol(*[RandomTerm(signature, depth - 1) for i in range(arity)])


def RandomPredicate(signature, depth=2.0):
    max_arity = len(signature.config.proposition_arity)
    arity_minus_one = random.choice(range(max_arity))
    symbol = random.choice(signature.predicates[arity_minus_one])
    return symbol(*[RandomTerm(signature, depth - 1) for i in range(arity_minus_one + 1)])


def RandomEquality(signature, depth=2.0):
    return eq(*[RandomTerm(signature, depth - 1) for i in range(2)])


def RandomLiteral(signature, depth=2.0, p_eq=0.25):
    polarity = random.choice([True, False]).item()
    if random.rand() > p_eq:
        return Literal(RandomPredicate(signature, depth), polarity)
    else:
        return Literal(RandomEquality(signature, depth), polarity)

def RandomClause(signature, size, depth=2.0, p_eq=0.25):
    return Clause(*[RandomLiteral(signature, depth, p_eq) for i in range(size)])
    