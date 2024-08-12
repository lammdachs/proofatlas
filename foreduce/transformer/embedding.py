import torch
from torch import nn

from foreduce.transformer.tokenizer import TokenConfig


class FormulaEmbedding(nn.Module):
    def __init__(self, config : TokenConfig, args):
        super().__init__()

        self.config = config

        self.embeddings = nn.Embedding(
            config.RESERVED_TOKENS + sum(config.num_functions) + config.num_variables,
            args["embed_dim"]
        )

    def update_config(self, num_variables, num_functions):
        if num_variables <= self.config.num_variables and len(num_functions) <= len(self.config.num_functions) \
            and all(n <= m for n, m in zip(num_functions, self.config.num_functions)):
            return
        slices = [self.embeddings.weight.data[:self.config.RESERVED_TOKENS]]
        index = self.config.RESERVED_TOKENS
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
            slices.append(self.embeddings.weight.data[-self.config.num_variables-1].repeat(i, 1))
        if num_variables > self.config.num_variables:
            slices.append(torch.cat([
                self.embeddings.weight.data[index:index + self.config.num_variables],
                self.embeddings.weight.data[index].repeat(num_variables - self.config.num_variables, 1)
            ], dim=0))
            index += self.config.num_variables
        else:
            slices.append(self.embeddings.weight.data[index:index + num_variables])
            index += num_variables
        self.embeddings.weight.data = torch.cat(slices, dim=0)
        self.config.num_variables = num_variables
        self.config.num_functions = num_functions
            
    def forward(self, x):
        """
        Expects formulas as a BxPxL tensor, where B is the batch size, i.e. number of problems,
        P is the size of each problem andand L is the length of each formula.
        """
        B, P, L = x.size()
        x = self.embeddings(x)
        return x
