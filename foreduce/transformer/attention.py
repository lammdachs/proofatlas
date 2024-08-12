import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, args, outdim=None):
        super().__init__()
        self.seq_len = args["seq_len"]
        self.num_heads = args["num_heads"]
        self.embed_dim = args["embed_dim"]
        self.out_dim = outdim if outdim is not None else self.embed_dim
        self.head_dim = self.out_dim // self.num_heads
        
        angles = torch.tensor([10000**(2 * i / self.head_dim) for i in range(self.head_dim // 2)])
        self.register_buffer("rotations", torch.view_as_real(torch.exp(
            1j * torch.matmul(
                torch.arange(self.seq_len).float().reshape(-1, 1),
                angles.reshape(1, -1)
            )
        ).reshape(1, self.seq_len, 1, self.head_dim // 2)))
        
        self.w_q = nn.Linear(self.embed_dim, self.head_dim*self.num_heads, bias=False)
        self.w_k = nn.Linear(self.embed_dim, self.head_dim*self.num_heads, bias=False)
        self.w_v = nn.Linear(self.embed_dim, self.head_dim*self.num_heads, bias=False)
        if outdim is None:
            self.w_o = nn.Linear(self.num_heads*self.head_dim, self.embed_dim, bias=False)
        else:
            self.w_o = nn.Linear(self.num_heads*self.head_dim, outdim, bias=False)
    
    def forward(self, x):
        
        batch_size, seq_len, _ = x.shape
        
        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        q, k = rotary_positional_embedding(q, k, self.rotations)
        
        att = torch.einsum('bqhd,bkhd->bhqk', q, k) / (self.head_dim ** 0.5)
        att = nn.functional.softmax(att, dim=-1)
        y = torch.einsum('bhqk,bkhd->bqhd', att, v)
        return self.w_o(y.reshape(batch_size, seq_len, -1))


def rotary_positional_embedding(q, k, rotations):
    q_shape, k_shape = q.shape, k.shape
    q = torch.view_as_complex(q.view(*q.shape[:-1], -1, 2))
    k = torch.view_as_complex(k.view(*k.shape[:-1], -1, 2))
    q = torch.view_as_real(q * torch.view_as_complex(rotations[:, :q.shape[1], :, :]))
    k = torch.view_as_real(k * torch.view_as_complex(rotations[:, :k.shape[1], :, :]))
    return q.view(q_shape), k.view(k_shape)
