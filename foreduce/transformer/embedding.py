from lightning import LightningModule
import torch
from torch import nn

from foreduce.transformer.tokenizer import TokenConfig
from foreduce.transformer.transformer import TransformerLayer


class FormulaEmbedding(LightningModule):
    def __init__(self, config : TokenConfig, seq_len=128, dim=256, n_layers=4, n_heads=4):
        super().__init__()

        self.config = config
        self.seq_len = seq_len
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.embeddings = nn.Embedding(
            config.RESERVED_TOKENS + sum(config.num_functions) + 1,
            dim,
            padding_idx=0
        )
        self.layers = nn.ModuleList([
            TransformerLayer(dim, n_heads, True, seq_len) for _ in range(n_layers)
        ])
        self.out = nn.Linear(dim, dim)

        self._init()
        self.save_hyperparameters()
        
    def _init(self):
        nn.init.xavier_uniform_(self.embeddings.weight)
        nn.init.xavier_uniform_(self.out.weight)

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
        Expects formulas as a BxL tensor, where B is the batch size, i.e. number of clauses,
        L is the length of each clause.
        """
        offset = self.config.RESERVED_TOKENS + sum(self.config.num_functions)
        vars = x >= offset
        attn_mask = torch.zeros(x.size(0), x.size(1), x.size(1)).to(x.device)
        attn_mask = attn_mask.masked_fill(vars.unsqueeze(1) | vars.unsqueeze(2), -10e9)
        for var in range(offset, offset + self.config.num_variables):
            _vars = x == var
            attn_mask = attn_mask.masked_fill(_vars.unsqueeze(1) & _vars.unsqueeze(2), 0)
        x = x.masked_fill(vars, self.config.RESERVED_TOKENS + sum(self.config.num_functions))
        x = self.embeddings(x)
        x = self.layers[0](x, attn_mask)
        for layer in self.layers[1:]:
            x = layer(x)
        return self.out(x).sum(dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2.5*1e-4)
        return [optimizer], []
    
    def training_step(self, batch, batch_idx):
        x, y, target, weight = batch
        similarities = torch.cosine_similarity(self(x), self(y), dim=-1)
        _fill = torch.full(target.size(), 1e-2).to(target)
        loss = torch.where(
            similarities >= target,
            (similarities - target) * torch.min(similarities, _fill),
            (target - similarities) * torch.min(1 - similarities, _fill)
        ) *  torch.max(torch.ones_like(weight).to(weight), weight)
        loss = loss.mean()
        self.log("train_loss", loss, on_step=True, 
                 on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, target, weight = batch
        similarities = torch.cosine_similarity(self(x), self(y), dim=-1)
        _fill = torch.full(target.size(), 1e-2).to(target)
        loss = torch.where(
            similarities >= target,
            (similarities - target) * torch.min(similarities, _fill),
            (target - similarities) * torch.min(1 - similarities, _fill)
        ) * torch.max(torch.ones_like(weight).to(weight), weight)
        loss = loss.mean()
        self.log("val_loss", loss, on_step=True, 
                 on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
        return loss