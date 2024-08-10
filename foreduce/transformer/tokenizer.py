from collections import deque
from dataclasses import dataclass, field
import random
import torch
from typing import List


@dataclass
class TokenConfig:
    RESERVED_TOKENS: int = 9 # padding, start, end, |, ~, $true, $false, equality
    reserved_token_mapping: dict = field(default_factory=lambda: {
        '<PAD>' : 0, '<START>' : 1, '<END>' : 2, '|' : 3, '~' : 4, '$true' : 5, '$false' : 6, 'eq' : 7})
    num_functions: List[int] = field(default_factory=lambda: [16, 16, 8, 4, 2, 1])
    num_variables: int = 8
    embed_dim: int = 128
    
    def random_function_mapping(self, function_symbols : list[list[str]]):
        mapping = {}
        offset = self.RESERVED_TOKENS
        if len(function_symbols) > len(self.num_functions):
            raise ValueError(f"Too high arity")
        for arity, symbols in enumerate(function_symbols):
            if len(symbols) > self.num_functions[arity]:
                raise ValueError(f"Too many function symbols of arity {arity}")
            configuration = random.sample(range(self.num_functions[arity]), len(symbols))
            for i, symbol in enumerate(symbols):
                mapping[symbol] = offset + configuration[i]
            offset += self.num_functions[arity]
        return mapping
    
    def variable_mapping(self, variable_symbols : List[str]):
        mapping = {}
        offset = self.RESERVED_TOKENS + sum(self.num_functions)
        self.num_variables = max(self.num_variables, len(variable_symbols))
        for i, symbol in enumerate(variable_symbols):
            mapping[symbol] = offset + i
        return mapping


class ProofTokenizer:
    def __init__(self, config, max_steps=1024, max_tokens=128, seed=42):
        self.config = config
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.generator = random.Random(seed)

    def __call__(self, problem, tree, mapping=None, goal='random'):        
        if goal == 'random':  
            goal = self.generator.choice(range(len(tree) // 2, len(tree)))
        elif goal == 'last':
            goal = len(tree) - 1
        
        tokens, mapping = problem.tokenize(self.config, limit=self.max_steps, mapping=mapping)
        x = torch.zeros(self.max_steps, self.max_tokens)
        y = torch.zeros(self.max_steps)
        target = torch.zeros(self.max_tokens)
        for i, clause in enumerate(tokens):
            for j, token in enumerate(clause[:self.max_tokens]):
                x[i, j] = token
        queue = deque([goal])
        seen = {i}
        while queue:
            i = queue.popleft()
            if i < self.max_steps:
                y[i] = 1
            for j in tree[i]:
                if j not in seen:
                    queue.append(j)
                    seen.add(j)
        for i, token in enumerate(problem.clauses[goal].tokenize(self.config, mapping=mapping)[:self.max_tokens]):
            target[i] = token
        return x, y, target, mapping


class ProblemTokenizer:
    def __init__(self, config, max_clauses=1024, max_tokens=128):
        self.config = config
        self.max_clauses = max_clauses
        self.max_tokens = max_tokens

    def __call__(self, problem, mapping=None):
        tokens, mapping = problem.tokenize(self.config, mapping=mapping, limit=self.max_clauses)
        x = torch.zeros(self.max_clauses, self.max_tokens)
        for i, clause in enumerate(tokens[:self.max_clauses]):
            for j, token in enumerate(clause[:self.max_tokens]):
                x[i, j] = token
        return x, mapping

 