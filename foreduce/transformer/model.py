import torch
import torch.nn as nn
from lightning import LightningModule

from foreduce.transformer.modelargs import ModelArgs
from foreduce.transformer.decoder import DecoderLayer, RMSNorm

class Transformer(LightningModule):
    @staticmethod
    def default_args():
        ModelArgs()
    
    def __init__(self, args):
        super().__init__()
        self.seq_len = args.seq_len
        self.embed_dim = args.embed_dim
        if self.embed_dim % 2 != 0:
            raise ValueError("Embedding dimension must be even")
        self.vocab_size = args.vocab_size()
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(args.dropout)
        self.norm = RMSNorm(self.embed_dim)

        for _ in range(args.num_layers):
            dec = DecoderLayer(
                args
            )
            self.layers.append(dec)
            
        self.out_layer = nn.Linear(self.embed_dim, self.vocab_size)
        
        self.save_hyperparameters()

    def forward(self, x):
        x = self.dropout(self.embed(x))
        for dec in self.layers:
            x = dec(x)
        x = self.norm(x)
        return self.out_layer(x)

    def next(self, x):
        p_next = self(x)
        return torch.argmax(p_next, dim=-1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.01, total_iters=16*1000
            )
        }

    def training_step(self, batch, batch_idx):
        x, mask = batch
        preds = self(x)[:,:-1,:]
        loss = nn.functional.cross_entropy(
            mask[:, 1:].reshape(-1, 1) * preds.flatten(start_dim=0, end_dim=1),
            mask[:, 1:].flatten() * x[:, 1:].flatten()
        )
        self.log("train_loss", loss,
            on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, mask = batch
        preds = self(x)[:,1:,:]
        loss = nn.functional.cross_entropy(
            mask[:, 1:].reshape(-1, 1) * preds.flatten(start_dim=0, end_dim=1),
            mask[:, :-1].flatten() * x[:, :-1].flatten()
        )
        self.log("val_loss", loss,
            on_epoch=True, logger=True, sync_dist=True)
        return loss
