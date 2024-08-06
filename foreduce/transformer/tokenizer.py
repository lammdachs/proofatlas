from dataclasses import dataclass, field
import random
from typing import List

@dataclass
class TokenConfig:
    RESERVED_TOKENS: int = 8 # padding, start, end, |, ~, $true, $false, equality
    reserved_token_mapping: dict = field(default_factory=lambda: {
        '<PAD>' : 0, '<START>' : 1, '<END>' : 2, '|' : 3, '~' : 4, '$true' : 5, '$false' : 6, '=' : 7})
    num_variables: int = 8
    num_functions: List[int] = field(default_factory=lambda: [16, 16, 8, 4, 2, 1])
    embed_dim: int = 128
    
    def random_variable_mapping(self, variable_symbols : List[str]):
        mapping = {}
        if len(variable_symbols) > self.num_variables:
            raise ValueError(f"Too many variable symbols")
        configuration = random.sample(range(self.num_variables), len(variable_symbols))
        for i, symbol in enumerate(variable_symbols):
            mapping[symbol] = self.RESERVED_TOKENS + configuration[i]
        return mapping
    
    def random_function_mapping(self, function_symbols : list[list[str]]):
        mapping = {}
        offset = self.RESERVED_TOKENS + self.num_variables
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