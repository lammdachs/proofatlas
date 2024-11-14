import torch
import torch.nn as nn
from lightning import LightningModule

from foreduce.transformer.embedding import FormulaEmbedding
from foreduce.transformer.transformer import TransformerLayer, MultiHeadAttention
from foreduce.transformer.gnn import GNN


class GraphModel(LightningModule):
    def __init__(self, num_types, max_arity, layers, dim, conv="GCN", activation="ReLU", lr=1e-4):
        super().__init__()
        self.num_types = num_types
        self.max_arity = max_arity
        self.layers = layers
        self.dim = dim
        self.conv = conv
        self.activation = activation
        self.gnn = GNN(num_types, max_arity, layers, dim, conv, activation)
        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )
        self.lr = lr
        
        self.save_hyperparameters("num_types", "max_arity", "layers", "dim", "conv", "activation", "lr")
        
    def forward(self, batch):
        x = self.gnn(batch)[batch.clauses]
        return self.out(x).squeeze(-1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.ConstantLR(optimizer),
        }
    
    def training_step(self, batch, batch_idx):
        preds = self(batch)
        loss = nn.functional.binary_cross_entropy_with_logits(
            preds, batch.labels.to(torch.float)
        )
        self.log("train_loss", loss, on_step=True, logger=True, sync_dist=True)
        return loss



class Model(LightningModule):
    default_args = {
        "clause_length": 128,
        "clause_num_heads": 8,
        "clause_embed_layers": 8,
        "clause_embed_dim": 64,
        "problem_length": 1024,
        "problem_num_heads": 8,
        "problem_embed_layers": 8,
        "problem_embed_dim": 512,
    }
    
    def __init__(self, args, token_config):
        super().__init__()
        self.args = args

        self.formula_embedding = FormulaEmbedding(
            token_config, {"embed_dim": args["clause_embed_dim"]}
        )
        self.clause_transformer = nn.ModuleList([
            DecoderLayer(
                {
                    "seq_len": args["clause_length"],
                    "num_heads": args["clause_num_heads"],
                    "embed_dim": args["clause_embed_dim"],
                }
            ) for _ in range(args["clause_embed_layers"])
        ])
        self.norm = nn.LayerNorm(args["clause_embed_dim"])
        self.pool_attention = MultiHeadAttention(
            {
                "seq_len": args["clause_length"],
                "num_heads": args["problem_num_heads"],
                "embed_dim": args["clause_embed_dim"],
            },
            outdim=args["problem_embed_dim"]
        )
        self.problem_transformer = nn.ModuleList([
            DecoderLayer(
            {
                "seq_len": args["problem_length"] + 1,
                "num_heads": args["problem_num_heads"],
                "embed_dim": args["problem_embed_dim"],          
            }
            ) for _ in range(args["problem_embed_layers"])
        ])
        self.out_query = nn.Linear(args["problem_embed_dim"], args["problem_embed_dim"])
        self.out_key = nn.Linear(args["problem_embed_dim"], args["problem_embed_dim"])
        
        self.save_hyperparameters()

    def forward(self, x, t=None):
        # x: BxPxL
        # t: BxL
        B, P, L = x.size()
        if t is None:
            # if no target is given, we use $false as the target
            t = torch.zeros(B, L, dtype=torch.int).to(x.device)
            t[:, 0] = 1; t[:, 1] = 6; t[:, 2] = 2
        x = torch.cat((t.unsqueeze(1), x), dim=1)
        P += 1
        x = self.clause_embeddings(x)
        for layer in self.problem_transformer:
            x = layer(x)
        query = self.out_query(x[:, 0])
        key = self.out_key(x[:, 1:])
        return torch.einsum("bd,bld->bl", query, key)

    def clause_embeddings(self, x):
        B, P, L = x.size()
        x = self.formula_embedding(x).view(B * P, L, -1)
        for layer in self.clause_transformer:
            x = layer(x)
        x = self.pool_attention(self.norm(x))
        x = torch.sum(x, dim=1)/torch.count_nonzero(x, dim=1)
        return x.view(B, P, -1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.ConstantLR(optimizer),
        }

    def training_step(self, batch, batch_idx):
        x, y, t = batch
        preds = self(x, t)
        loss = nn.functional.binary_cross_entropy_with_logits(
            preds, y.to(torch.float)
        )
        self.log("train_loss", loss,
            on_step=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, t = batch
        preds = self(x, t)
        loss = nn.functional.binary_cross_entropy_with_logits(
            preds, y.to(torch.float)
        )
        self.log("val_loss", loss,
            on_step=False, logger=True, sync_dist=True)
        return loss
    