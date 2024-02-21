from dataclasses import dataclass
from itertools import groupby, takewhile
from operator import attrgetter

import numpy.random as random
from foreduce.fol import eq
from bidict import bidict

from foreduce.fol.exception import FunctionArityError, FunctionCountError, PredicateArityError, PredicateCountError, VariableCountError
from foreduce.fol.logic import Clause, Literal, Problem


@dataclass
class Config:
    model_dim : int = 128
    embed_dim : int = 32
    seq_len : int = 128
    num_heads : int = 4
    dropout : int = 0
    inner_dim : int = 128
    num_layers : int = 2
    proposition_arity = [4, 2]
    function_arity = [4, 4, 2]
    variable_count = 4
    defined = ["&", "|", "+", "-", eq]

    def vocab_size(self):
        return len(self.defined) + sum(self.function_arity) + \
            sum(self.proposition_arity) + \
            self.variable_count

    def mapping(self, problem):
        random.seed(hash(problem) % 2**32)
        mapping = bidict({symbol: i for i, symbol in enumerate(self.defined)})
        reserved = len(mapping)
        function_symbols = list(problem.function_symbols())
        predicate_symbols = list(problem.predicate_symbols())
        variables = list(problem.variables())
        function_symbols.sort(key=attrgetter('arity'))
        if eq in predicate_symbols:
            predicate_symbols.remove(eq)
        predicate_symbols.sort(key=attrgetter('arity'))
        for arity, functions in groupby(
                function_symbols, attrgetter('arity')):
            if arity >= len(self.function_arity):
                raise FunctionArityError(arity, self)
            functions_list = list(functions)
            if len(functions_list) > self.function_arity[arity]:
                raise FunctionCountError(len(functions_list), arity, self)
            ids = random.choice(
                range(self.function_arity[arity]),
                len(functions_list), replace=False)
            for i, function in enumerate(functions_list):
                mapping[function] = reserved + ids[i] + sum(
                    self.function_arity[:arity]
                )
        for arity, predicates in groupby(
                predicate_symbols, attrgetter('arity')):
            if arity == 0 or arity >= len(self.proposition_arity) + 1:
                raise PredicateArityError(arity, self)
            predicates_list = list(predicates)
            if len(predicates_list) > self.proposition_arity[arity - 1]:
                raise PredicateCountError(len(predicates_list), arity, self)
            ids = random.choice(
                range(self.proposition_arity[arity - 1]),
                len(predicates_list), replace=False)
            for i, predicate in enumerate(predicates_list):
                mapping[predicate] = reserved + ids[i] + sum(
                    self.function_arity
                    ) + sum(
                    self.proposition_arity[:arity-1]
                )
        if len(variables) > self.variable_count:
            raise VariableCountError(len(variables), self)
        ids = random.choice(
            range(self.variable_count), len(variables), replace=False)
        for i, variable in enumerate(variables):
            mapping[variable] = reserved + ids[i] + sum(
                self.function_arity
                ) + sum(
                self.proposition_arity
                )
        return mapping

    def encode(self, problem: Problem):
        return problem.encode(self.mapping(problem))

    def decode(self, encoding, mapping):
        clauses = []
        it = iter(encoding)
        while True:
            clause = list(takewhile(lambda x: x != mapping["&"], it))
            if not clause:
                break
            clauses.append(self.decode_clause(
                clause,
                mapping
            ))
        return Problem(*clauses)

    def decode_clause(self, clause, mapping):
        literals = []
        it = iter(clause)
        while True:
            literal = list(takewhile(lambda x: x != mapping["|"], it))
            if not literal:
                break
            literals.append(self.decode_literal(
                literal,
                mapping
            ))
        return Clause(*literals)

    def decode_literal(self, literal: list, mapping):
        polarity = literal[0] == mapping["+"]
        predicate = self.decode_term(iter(literal[1:]), mapping)
        return Literal(predicate, polarity)

    def decode_term(self, it, mapping):
        symbol = mapping.inverse[next(it)]
        if symbol.arity == 0:
            return symbol
        args = []
        while len(args) < symbol.arity:
            args.append(self.decode_term(it, mapping))
        return symbol(*args)

    def encode_substitution_proof(self, substitution_proof):
        axioms = substitution_proof[0]
        substitutions = substitution_proof[1]
        problem = Problem(*axioms + substitutions)
        mapping = self.mapping(problem)
        return (len(axioms), problem.encode(mapping))
