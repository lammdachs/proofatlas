import numpy.random as random
from foreduce.fol.logic import Clause, Constant, Function, Literal, Predicate, Variable, eq
from foreduce.fol.derivation import DerivedClause, Axiom, Substitution, Resolution, Superposition

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
        for arity, count in enumerate(config.predicate_arity):
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
    max_arity = len(signature.config.predicate_arity)
    arity_minus_one = random.choice(range(max_arity))
    symbol = random.choice(signature.predicates[arity_minus_one])
    return symbol(*[RandomTerm(signature, depth - 1) for i in range(arity_minus_one + 1)])


def RandomEquality(signature, depth=2.0):
    return eq(*[RandomTerm(signature, depth - 1) for i in range(2)])


def RandomLiteral(signature, depth=2.0, p_eq=0.25):
    polarity = random.choice([True, False]).item()
    if random.rand() < p_eq:
        return Literal(RandomEquality(signature, depth), polarity)
    else:
        return Literal(RandomPredicate(signature, depth), polarity)

def RandomClause(signature, size, depth=2.0, p_eq=0.25):
    return Clause(*[RandomLiteral(signature, depth, p_eq) for i in range(size)])


def RandomAxiom(signature, size, depth=2.0, p_eq=0.25, verifying_substitution=dict()):
    return Axiom(RandomClause(signature, size, depth, p_eq))


def RandomSubstitution(signature, clause, verifying_substitution=dict()):
    candidates = list(set(signature.variables) - clause.variables())
    if not candidates:
        return Axiom(clause, verifying_substitution)
    candiate_terms = list(clause.terms() - clause.variables())
    if not candiate_terms:
        return Axiom(clause, verifying_substitution)
    variable = random.choice(candidates)
    term = random.choice(candiate_terms)
    return Substitution(Axiom(clause.substitute(term, variable)), variable, term, verifying_substitution)


def RandomResolution(signature, clause, depth=2.0, p_eq=0.25, p_double=0.2, verifying_substitution=dict()):
    left = []
    right = []
    for literal in clause.literals:
        if random.rand() < p_double:
            left.append(literal)
            right.append(literal)
        elif random.rand() < 0.5:
            left.append(literal)
        else:
            right.append(literal)
    resolvent = RandomLiteral(signature, depth, p_eq)
    left.append(Literal(resolvent.predicate, True))
    right.append(Literal(resolvent.predicate, False))
    return Resolution(
        Axiom(Clause(*left)),
        Axiom(Clause(*right)),
        resolvent.predicate,
        verifying_substitution
    )


def RandomSuperposition(signature, clause, depth=2.0, p_eq=0.25, p_double=0.2, verifying_substitution=dict()):
    left = []
    right = [random.choice(clause.literals)]
    for literal in [lit for lit in clause.literals if lit != right[0]]:
        if random.rand() < p_double:
            left.append(literal)
            right.append(literal)
        elif random.rand() < 0.5:
            left.append(literal)
        else:
            right.append(literal)
    term1 = RandomTerm(signature, depth)
    term2 = random.choice([lit for r in right for lit in r.terms()])
    right = [lit.substitute(term2, term1) for lit in right]
    left_to_right = random.choice([True, False]).item()
    equality = eq(term1, term2) if left_to_right else eq(term2, term1)
    left.append(Literal(equality, True))
    return Superposition(
        Axiom(Clause(*left)),
        Axiom(Clause(*right)),
        equality, left_to_right,
        verifying_substitution
    )


def RandomDerivation(
    signature,
    clause,
    derivation_depth,
    p_rules = None,
    depth=2.0,
    p_eq=0.25,
    p_double=0.2,
    verifying_substitution=dict()
):
    if derivation_depth == 0:
        return Axiom(clause, verifying_substitution)
    if not clause.literals:
        result = RandomResolution(signature, clause, depth, p_eq, p_double, verifying_substitution)
    else:
        rule = random.choice(range(4), p=p_rules)
        if rule == 0:
            result = RandomSubstitution(signature, clause, verifying_substitution)
        elif rule == 1:
            result = RandomResolution(signature, clause, depth, p_eq, p_double, verifying_substitution)
        elif rule == 2:
            result = RandomSuperposition(signature, clause, depth, p_eq, p_double, verifying_substitution)
        else:
            result = Axiom(clause, verifying_substitution)
    children = tuple(RandomDerivation(
        signature, child, derivation_depth - 1, p_rules, depth, p_eq, p_double, child.verifying_substitution
    ) for child in result.children)
    result.children = children
    return result


def RandomSubstitutionProof(
    signature,
    derivation_depth,
    p_rules=None,
    depth=2.0,
    p_eq=0.25,
    p_double=0.2,
    random_axiom=0.0
):
    derivation = RandomDerivation(
        signature,
        Clause(),
        derivation_depth,
        [0, 1/3, 1/3, 1/3],
        depth,
        p_eq,
        p_double
    )
    for node in derivation.nodes():
        for i, child in enumerate(node.children):
            if isinstance(child, Axiom):
                node.children = tuple(
                    node.children[:i] +
                    (RandomSubstitution(signature, child, child.verifying_substitution),) +
                    node.children[i+1:]
                )
    proof = derivation.substitution_proof()
    axioms = proof[0]
    result = [
        RandomAxiom(signature, random.choice(range(2)) + 1, depth, p_eq)
        for i in range(random.binomial(len(axioms), random_axiom/len(axioms)))
    ]
    for axiom in axioms:
        result.append(axiom)
        result += [
            RandomAxiom(signature, random.choice(range(2)) + 1, depth, p_eq)
            for i in range(random.binomial(len(axioms), random_axiom/len(axioms)))
        ]
    return (tuple(result), proof[1])
    
