import torch
import torch.nn as nn
from lightning import LightningModule
from torch_geometric.data import Batch
from torchtune.modules import RotaryPositionalEmbeddings

from foreduce.transformer.embedding import FormulaEmbedding
from foreduce.transformer.transformer import TransformerLayer, MultiHeadAttention
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
            preds, batch.labels.to(torch.float), reduction='mean'
        )
        self.log("train_loss", loss, on_step=True, logger=True, sync_dist=True, batch_size=batch.ptr.size(0) - 1)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self(batch)
        loss = nn.functional.binary_cross_entropy_with_logits(
            preds, batch.labels.to(torch.float), reduction='mean'
        )
        self.log("val_loss", loss, on_step=False, logger=True, sync_dist=True, batch_size=batch.ptr.size(0) - 1)
        return loss



class Model(LightningModule):    
    def __init__(self, num_types, max_arity, gnn_layers, transformer_layers, dim, conv="GCN", activation="ReLU", n_heads=8, topk=64, lr=1e-7):
        super().__init__()
        self.num_types = num_types
        self.max_arity = max_arity
        self.gnn_layers = gnn_layers
        self.transformer_layers = transformer_layers
        self.dim = dim
        self.conv = conv
        self.activation = activation
        self.n_heads = n_heads
        self.topk = topk
        self.lr = lr
        
        self.gnn = GNN(num_types, max_arity, gnn_layers, dim, conv, activation)
        self.readout = Readout(dim)
        self.out_gnn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.transformer = nn.ModuleList(
            TransformerLayer(dim, n_heads, rope=True) for _ in range(transformer_layers)
        )
        self.out_transformer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )
        
        self.save_hyperparameters("num_types", "max_arity", "gnn_layers", "transformer_layers", "dim", "conv", "activation", "n_heads", "topk", "lr")

    def predict(self, graph):
        x = self.gnn(graph)[graph.clauses]
        gnn_out = self.readout(x)
        x = self.out_gnn(x)
        topk = torch.topk(gnn_out, min(self.topk, gnn_out.size(0)), dim=0, largest=True, sorted=False).indices.reshape(-1)
        transformer_input = x[topk, :].reshape(1, -1, self.dim)
        input_pos = topk.reshape(1, -1)
        x = transformer_input
        for layer in self.transformer:
            x = layer(x, input_pos=input_pos)
        transformer_out = self.out_transformer(x)
        return transformer_out.reshape(-1), [i.item() for i in topk]

    def forward(self, batch):
        x = self.gnn(batch)[batch.clauses]
        gnn_out = self.readout(x, batch.batch[batch.clauses])
        x = self.out_gnn(x)
        topk = []
        transformer_input = torch.zeros(batch.ptr.size(0) - 1, self.topk, self.dim, device=x.device, dtype=x.dtype)
        input_pos = torch.zeros(batch.ptr.size(0) - 1, self.topk, device=x.device, dtype=torch.int)
        for i in range(batch.ptr.size(0) - 1):
            mask = batch.batch[batch.clauses] == i
            topk_i = torch.topk(
                gnn_out[mask],
                min(self.topk, mask.sum().item()),
                dim=0,
                largest=True,
                sorted=False
            ).indices.squeeze() + (batch.batch[batch.clauses] < i).sum().item()
            topk_i = torch.sort(topk_i).values
            topk.append(topk_i)
            transformer_input[i, :topk_i.size(0), :] = x[topk_i, :]
            input_pos[i, :topk_i.size(0)] = topk_i
        x = transformer_input
        for layer in self.transformer:
            x = layer(x, input_pos=input_pos)
        x = self.out_transformer(x)
        return x.squeeze(-1), gnn_out.squeeze(-1), topk
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.ConstantLR(optimizer),
        }
        
    def training_step(self, batch, batch_idx):
        preds_transformer, preds_gnn, topk = self(batch)
        loss_transformer = torch.tensor(0., device=batch.ptr.device)
        for i, k in enumerate(topk):
            loss_transformer += nn.functional.binary_cross_entropy_with_logits(
                preds_transformer[i, :k.size(0)],
                batch.labels.to(torch.float)[k],
                reduction='mean'
            )
        loss_gnn = nn.functional.binary_cross_entropy_with_logits(
            preds_gnn, batch.labels.to(torch.float), reduction='mean'
        )
        loss = loss_transformer + loss_gnn
        self.log("train_loss_transformer", loss_transformer, on_step=True, logger=True, sync_dist=True, batch_size=batch.ptr.size(0) - 1)
        self.log("train_loss_gnn", loss_gnn, on_step=True, logger=True, sync_dist=True, batch_size=batch.ptr.size(0) - 1)
        self.log("train_loss", loss, on_step=True, logger=True, sync_dist=True, batch_size=batch.ptr.size(0) - 1)
        return loss
    
    def validation_step(self, batch, batch_idx):
        preds_transformer, preds_gnn, topk = self(batch)
        loss_transformer = torch.tensor(0., device=batch.ptr.device)
        for i, k in enumerate(topk):
            loss_transformer += nn.functional.binary_cross_entropy_with_logits(
                preds_transformer[i, :k.size(0)],
                batch.labels.to(torch.float)[k],
                reduction='mean'
            )
        loss_gnn = nn.functional.binary_cross_entropy_with_logits(
            preds_gnn, batch.labels.to(torch.float), reduction='mean'
        )
        loss = loss_transformer + loss_gnn
        self.log("val_loss_transformer", loss_transformer, on_step=False, logger=True, sync_dist=True, batch_size=batch.ptr.size(0) - 1)
        self.log("val_loss_gnn", loss_gnn, on_step=False, logger=True, sync_dist=True, batch_size=batch.ptr.size(0) - 1)
        self.log("val_loss", loss, on_step=False, logger=True, sync_dist=True, batch_size=batch.ptr.size(0) - 1)
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