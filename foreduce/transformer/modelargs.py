from dataclasses import dataclass, field
from itertools import groupby, takewhile
from operator import attrgetter

import numpy.random as random
import torch
from foreduce.fol import eq
from bidict import bidict

from foreduce.fol.exception import FunctionArityError, FunctionCountError, PredicateArityError, PredicateCountError, VariableCountError
from foreduce.fol.logic import Clause, Constant, Function, Literal, Predicate, Problem, Variable


@dataclass
class ModelArgs:
    embed_dim : int = 128
    seq_len : int = 128
    num_heads : int = 4
    dropout : int = 0
    inner_dim : int = 128
    num_layers : int = 2
    predicate_arity : list[int] = field(default_factory=lambda: [4, 2])
    function_arity : list[int] = field(default_factory=lambda: [4, 4, 2])
    variable_count : int = 4
    defined = ["&", "|", "+", "-", eq]

    def default_mapping(self):
        mapping = bidict({symbol: i for i, symbol in enumerate(self.defined)})
        counter = len(self.defined)
        for arity, i in enumerate(self.function_arity):
            for j in range(i):
                if arity == 0:
                    mapping[Constant(f"c_{j}")] = counter
                else:
                    mapping[Function(f"f{arity}_{j}", arity)] = counter
                counter += 1
        for arity, i in enumerate(self.predicate_arity):
            for j in range(i):
                mapping[Predicate(f"p{arity+1}_{j}", arity+1)] = counter
                counter += 1
        for i in range(self.variable_count):
            mapping[Variable(f"X{i}")] = counter
            counter += 1
        return mapping

    def vocab_size(self):
        return len(self.defined) + sum(self.function_arity) + \
            sum(self.predicate_arity) + \
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
            if arity == 0 or arity >= len(self.predicate_arity) + 1:
                raise PredicateArityError(arity, self)
            predicates_list = list(predicates)
            if len(predicates_list) > self.predicate_arity[arity - 1]:
                raise PredicateCountError(len(predicates_list), arity, self)
            ids = random.choice(
                range(self.predicate_arity[arity - 1]),
                len(predicates_list), replace=False)
            for i, predicate in enumerate(predicates_list):
                mapping[predicate] = reserved + ids[i] + sum(
                    self.function_arity
                    ) + sum(
                    self.predicate_arity[:arity-1]
                )
        if len(variables) > self.variable_count:
            raise VariableCountError(len(variables), self)
        ids = random.choice(
            range(self.variable_count), len(variables), replace=False)
        for i, variable in enumerate(variables):
            mapping[variable] = reserved + ids[i] + sum(
                self.function_arity
                ) + sum(
                self.predicate_arity
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
        encoding = problem.encode(mapping)
        index = [i for i, a in enumerate(encoding) if a == mapping["&"]][len(axioms) - 1]
        return (index, problem.encode(mapping))

    def next_clause(self, problem, mapping, model):
        encoding = problem.encode(mapping)
        next = -1
        while next != mapping["&"] and len(encoding) < self.seq_len:
            p_next = model(torch.tensor([encoding]))[0, -1, :]
            next = int(torch.argmax(p_next))
            encoding.append(next)
        return self.decode(encoding, mapping).clauses[-1]
        