from itertools import islice
import torch
import torch.nn.functional as F
from torch.nn import ReLU, Linear, Sequential, Tanh, Sigmoid, Embedding, Module
from torch_geometric.nn import GINConv, GCNConv, global_add_pool, global_mean_pool, GraphNorm
from lightning import LightningModule

from foreduce.transformer.spherical_code import SphericalCode

def gin_conv(in_channels, out_channels):
    nn = Sequential(Linear(in_channels, 2 * in_channels), GraphNorm(2 * in_channels), ReLU(), Linear(2 * in_channels, out_channels))
    return GINConv(nn)


class GNN(LightningModule):
    def __init__(self, num_types, max_arity, layers, dim, conv="GCN", activation="ReLU"):
        super().__init__()
        self.num_types = num_types
        self.max_arity = max_arity
        self.layers = layers
        self.dim = dim
        self.conv = conv
        self.conv_name = conv

        self.type_embedding = Embedding(num_types, dim)
        self.spherical_code = SphericalCode(dim, requires_grad=False)
        self.arity_embedding = Embedding(max_arity + 2, dim, padding_idx=0)
        self.position_embedding = Embedding(max_arity + 2, dim, padding_idx=0)
        match activation:
            case "ReLU": self.act = ReLU()
            case "Tanh": self.act = Tanh()
            case "Sigmoid": self.act = Sigmoid()
            case _: raise ValueError(f"Unknown activation {activation}")
        
        match conv:
            case "GCN": self.conv = GCNConv
            case "GIN": self.conv = gin_conv
        
        self.norms = torch.nn.ModuleList(
            [GraphNorm(dim) for _ in range(layers)])
        self.conv_layers = torch.nn.ModuleList(
            [self.conv(dim, dim) for _ in range(layers)])

    def forward(self, data):
        typ, name, arity, pos, edge_index, batch = data.type, data.name, data.arity, data.pos, data.edge_index, data.batch
        x = self.type_embedding(typ) + self.spherical_code(name) + self.arity_embedding(arity) + self.position_embedding(pos)
        for conv, norm in zip(self.conv_layers, self.norms):
            x = x + norm(conv(x, edge_index))
            x = self.act(x)
        return x

