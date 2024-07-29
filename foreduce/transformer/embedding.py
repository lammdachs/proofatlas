from dataclasses import dataclass, field
from typing import List
import torch
from torch import nn
from torchtune.modules import RotaryPositionalEmbeddings


from foreduce.fol.logic import Problem


@dataclass
class EmbeddingConfig:
    RESERVED_TOKENS: int = 8 # padding, start, end, |, ~, $true, $false, equality
    num_variables: int = 8
    num_functions: List[int] = field(default_factory=lambda: [8, 8, 4, 2, 1])
    embed_dim: int = 128


class FormulaEmbedding(nn.Module):
    def __init__(self, config : EmbeddingConfig = EmbeddingConfig()):
        super().__init__()

        self.config = config

        self.embeddings = nn.Embedding(
            config.RESERVED_TOKENS + config.num_variables + sum(config.num_functions),
            config.embed_dim
        )
        self.positional = RotaryPositionalEmbeddings(config.embed_dim)

    def update_config(self, num_variables, num_functions):
        if num_variables <= self.config.num_variables and len(num_functions) <= len(self.config.num_functions) \
            and all(n <= m for n, m in zip(num_functions, self.config.num_functions)):
            return
        slices = [self.embeddings.weight.data[:self.config.RESERVED_TOKENS]]
        index = self.config.RESERVED_TOKENS
        if num_variables > self.config.num_variables:
            slices.append(torch.cat([
                self.embeddings.weight.data[index:index + self.config.num_variables],
                self.embeddings.weight.data[index].repeat(num_variables - self.config.num_variables, 1)
            ], dim=0))
            index += self.config.num_variables
        else:
            slices.append(self.embeddings.weight.data[index:index + num_variables])
            index += num_variables
        for i, (n, m) in enumerate(zip(num_functions, self.config.num_functions)):
            if n > m:
                slices.append(torch.cat([
                    self.embeddings.weight.data[index:index + m],
                    self.embeddings.weight.data[index].repeat(n - m, 1)
                ], dim=0))
            else:
                slices.append(self.embeddings.weight.data[index:index + n])
            index += m
        for i in num_functions[len(self.config.num_functions):]:
            slices.append(self.embeddings.weight.data[-1].repeat(i, 1))
        self.embeddings.weight.data = torch.cat(slices, dim=0)
        self.config.num_variables = num_variables
        self.config.num_functions = num_functions
            
    def forward(self, x):
        """
        Expects formulas as a BxL tensor, where B is the batch size, i.e. number of formulas,
        and L is the length of the longest formula in the batch.
        """
        x = self.embeddings(x)
        x = self.positional(x.view(x.size(0), x.size(1), 1, -1)).view(x.size(0), x.size(1), -1)
        return x
