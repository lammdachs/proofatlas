import torch
from torch import nn
from decoder import DecoderLayer
from config import Config
from lightning import LightningModule
from ..fol import *


class Transformer(LightningModule):
    @staticmethod
    def default_config():
        config = Config()
        config.model_dim = 128
        config.vocab_size = 10
        config.embed_dim = 32
        config.seg_len = 39
        config.num_heads = 4
        config.dropout = 0
        config.inner_dim = 128
        config.num_layers = 2
        return config
    
    def __init__(self, config):
        super().__init__()
        self.seg_len = config.seg_len
        self.model_dim = config.model_dim
        self.vocab_size = config.vocab_size
        self.embed = nn.Embedding(config.vocab_size, config.model_dim)
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(config.dropout)
        total_len = config.seg_len
        pos_encoding = self.get_sinusoid_pos_encoding(total_len, config.model_dim).clone()
        self.register_buffer("R", pos_encoding)
        
        for _ in range(config.num_layers):
            dec = DecoderLayer(
                config.model_dim,
                config.embed_dim,
                config.num_heads,
                config.inner_dim,
                config.dropout,
                self.R
            )
            self.layers.append(dec)
            
        self.out_layer = nn.Linear(config.model_dim, config.vocab_size)
        
        self.save_hyperparameters()
    
    def forward(self, x):
        x = self.dropout(self.embed(x))
        for dec in self.layers:
            x = dec(x)
        return self.out_layer(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.01, total_iters=16*1000
            )
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = nn.functional.cross_entropy(
            preds[:, 19:].reshape(-1, 10), y.flatten()
        )
        self.log("train_loss", loss,
            on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = nn.functional.cross_entropy(
            preds[:, 19:].reshape(-1, 10), y.flatten()
        )
        self.log("val_loss", loss,
            on_epoch=True, logger=True, sync_dist=True)
        return loss

    def get_sinusoid_pos_encoding(self, total_len, embed_dim):
        """
        Standard sinusoid positional encoding method outlined in the original
        Transformer paper. In this case, we use the encodings not to represent
        each token's position in a sequence but to represent the distance
        between two tokens (i.e. as a *relative* positional encoding).
        """
        pos = torch.arange(total_len).unsqueeze(1)
        enc = torch.arange(embed_dim).float()
        enc = enc.unsqueeze(0).repeat(total_len, 1)
        enc[:, ::2] = torch.sin(pos / 10000**(2*enc[:, ::2]/embed_dim))
        enc[:, 1::2] = torch.cos(pos / 10000**(2*enc[:, 1::2]/embed_dim))
        return enc
