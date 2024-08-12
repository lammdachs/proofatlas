from torch import nn
import torch
from foreduce.transformer.attention import MultiHeadAttention


class PositionwiseFeedForward(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(args["embed_dim"], int(1.3*args["embed_dim"])),
            nn.SiLU(),
            nn.Linear(int(1.3*args["embed_dim"]), args["embed_dim"]),
        )
        self.norm = RMSNorm(args["embed_dim"])
        
    def forward(self, x):
        return self.norm(self.net(x) + x)


class DecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attn = MultiHeadAttention(args)
        self.pos_ff = PositionwiseFeedForward(args)
        self.n1 = RMSNorm(args["embed_dim"])
        self.n2 = RMSNorm(args["embed_dim"])
    
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.pos_ff(self.n2(x))
        return x


class RMSNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.register_buffer("eps", torch.tensor(eps))

    def forward(self, x):
        return x * torch.rsqrt(
            torch.mean(x**2, dim=-1, keepdim=True) + self.eps
        ) * self.scale
