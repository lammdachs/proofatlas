import torch
import torch.nn as nn
from lightning import LightningModule
from torchtune.modules import RotaryPositionalEmbeddings

from foreduce.transformer.transformer import TransformerLayer
from foreduce.transformer.gnn import GNN


class GraphModel(LightningModule):
    def __init__(self, num_types, max_arity, layers, dim, conv="GCN", activation="ReLU", lr=1e-6):
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
        
    def forward(self, data):
        x = self.gnn(data)
        return self.out(x).squeeze(-1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.ConstantLR(optimizer),
        }
    
    def training_step(self, batch, batch_idx):
        data, labels = batch
        preds = self(data)
        y = labels.to(torch.float).view(-1) / torch.sum(labels)
        loss = nn.functional.cross_entropy(
            preds, y, reduction='none', label_smoothing=self.current_epoch / self.trainer.max_epochs
        )
        self.log("train_loss", loss, on_step=True, logger=True, sync_dist=True, batch_size=len(data))
        return loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        preds = self(data)
        y = labels.to(torch.float).view(-1) / torch.sum(labels)
        loss = nn.functional.cross_entropy(
            preds, y, reduction='none', label_smoothing=self.current_epoch / self.trainer.max_epochs
        )
        self.log("val_loss", loss, on_step=False, logger=True, sync_dist=True, batch_size=len(data))
        return loss



class Model(LightningModule):    
    def __init__(self, num_types, max_arity, gnn_layers, transformer_layers, dim, conv="GCN", activation="ReLU", n_heads=8, lr=1e-7):
        super().__init__()
        self.num_types = num_types
        self.max_arity = max_arity
        self.gnn_layers = gnn_layers
        self.transformer_layers = transformer_layers
        self.dim = dim
        self.conv = conv
        self.activation = activation
        self.n_heads = n_heads
        self.lr = lr
        
        self.gnn = GNN(num_types, max_arity, gnn_layers, dim, conv, activation)
        self.transformer = nn.ModuleList(
            TransformerLayer(dim, n_heads, rope=True) for _ in range(transformer_layers)
        )
        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )
        
        self.save_hyperparameters("num_types", "max_arity", "gnn_layers", "transformer_layers", "dim", "conv", "activation", "n_heads", "lr")

    def forward(self, data, input_pos=None):
        x = self.gnn(data).unsqueeze(0)
        for layer in self.transformer:
            x = layer(x, input_pos=input_pos)
        x = self.out(x)
        return x.reshape(-1)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.ConstantLR(optimizer),
        }
        
    def training_step(self, batch, batch_idx):
        data, labels = batch
        preds = self(data)
        y = labels.to(torch.float).view(-1) / torch.sum(labels)
        loss = nn.functional.cross_entropy(
            preds, y, label_smoothing = 1 - self.current_epoch / self.trainer.max_epochs
        ) / len(y)
        self.log("train_loss", loss, on_step=True, logger=True, sync_dist=True, batch_size=len(data))
        return loss
        
    
    def validation_step(self, batch, batch_idx):
        data, labels = batch
        preds = self(data)
        y = labels.to(torch.float).view(-1) / torch.sum(labels)
        loss = nn.functional.cross_entropy(
            preds, y, label_smoothing = 1 - self.current_epoch / self.trainer.max_epochs
        ) / len(y)
        self.log("val_loss", loss, on_step=False, logger=True, sync_dist=True, batch_size=len(data))
        return loss
        

class Readout(nn.Module):
    def __init__(self, dim, batched=True):
        super().__init__()
        self.l1 = nn.Linear(dim, dim)
        self.l2 = nn.Linear(dim, 1)
        self.rope = RotaryPositionalEmbeddings(dim)
    
    def forward(self, x, batch=None):
        if batch is not None:
            x = self.l1(x)
            for i in range(batch[-1].item()):
                mask = batch == i
                x[mask] = self.rope(x[mask].view(1, -1, 1, x.size(-1))).view(-1, x.size(-1))
            x = self.l2(x)
            return x
        else:
            x = self.l1(x)
            x = self.rope(x.view(1, -1, 1, x.size(-1))).view(-1, x.size(-1))
            x = self.l2(x)
            return x