import torch
from torch import nn
from torchtune.modules import RotaryPositionalEmbeddings
from einops import einsum, rearrange


class TransformerLayer(nn.Module):
    def __init__(self, dim, n_heads, rope=False, seq_len=None):
        super().__init__()
        self.attn = MultiHeadAttention(dim, n_heads, rope=rope, seq_len=seq_len)
        self.pos_ff = PositionwiseFeedForward(dim)
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
    
    def forward(self, x, mask=None):
        x = x + self.attn(self.n1(x), mask)
        x = x + self.pos_ff(self.n2(x))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads, rope=False, seq_len=None):
        super().__init__()
        self.n_heads = n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        if rope:
            self.rope = RotaryPositionalEmbeddings(dim // n_heads, max_seq_len=seq_len)
        self.norm = nn.LayerNorm(dim)
        
        self._init()

    def _init(self):
        nn.init.xavier_uniform_(self.wq.weight)
        nn.init.xavier_uniform_(self.wk.weight)
        nn.init.xavier_uniform_(self.wv.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, mask=None):
        q = rearrange(self.wq(x), "b n (h d) -> b n h d", h=self.n_heads)
        k = rearrange(self.wk(x), "b n (h d) -> b n h d", h=self.n_heads)
        if hasattr(self, "rope"):
            q = self.rope(q)
            k = self.rope(k)
        q = rearrange(q, "b n h d -> b h n d")
        k = rearrange(k, "b n h d -> b h n d")
        
        v = rearrange(self.wv(x), "b n (h d) -> b h n d", h=self.n_heads)
        
        similarity = einsum(q, k, "b h n d, b h s d -> b h n s")
        if mask is not None:
            attention =  nn.functional.softmax(similarity + mask.unsqueeze(1), dim=-1) / (k.size(-1) ** 0.5)
        else:
            attention = nn.functional.softmax(similarity, dim=-1) / (k.size(-1) ** 0.5)
        x = einsum(attention, v, "b h n s, b h s d -> b h n d")
        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.out_proj(self.norm(x))

        return x

    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(dim, 2*dim),
            nn.SiLU(),
            nn.LayerNorm(2*dim),
            nn.Linear(2*dim, dim),
        )
        
        self._init()
        
    def _init(self):
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.xavier_uniform_(self.net[3].weight)
        
    def forward(self, x):
        return self.net(x) + x


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.register_buffer("eps", torch.tensor(eps))

    def forward(self, x):
        return x * torch.rsqrt(
            torch.mean(x**2, dim=-1, keepdim=True) + self.eps
        ) * self.scale
