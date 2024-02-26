import torch
from torch import nn
        
class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_heads = args.num_heads
        self.embed_dim = args.embed_dim
        
        angles = torch.tensor([10000**(2 * i / args.embed_dim) for i in range(args.embed_dim // 2)])
        self.register_buffer("rotations", torch.view_as_real(torch.exp(
            1j * torch.matmul(
                torch.arange(args.seq_len).float().reshape(-1, 1),
                angles.reshape(1, -1)
            )
        ).reshape(1, args.seq_len, 1, args.embed_dim // 2)))
        self.register_buffer("mask", torch.tril(
            torch.ones(args.seq_len, args.seq_len)).reshape(
                1, 1, args.seq_len, args.seq_len))
        
        self.w_q = nn.Linear(args.embed_dim, args.num_heads*args.embed_dim, bias=False)
        self.w_k = nn.Linear(args.embed_dim, args.num_heads*args.embed_dim, bias=False)
        self.w_v = nn.Linear(args.embed_dim, args.num_heads*args.embed_dim, bias=False)
        self.w_o = nn.Linear(args.num_heads*args.embed_dim, args.embed_dim, bias=False)
        self.dropout = nn.Dropout(args.dropout)
    
    def forward(self, x):
        
        batch_size, seq_len, _ = x.shape
        
        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.embed_dim)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.embed_dim)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.embed_dim)

        q, k = rotary_positional_embedding(q, k, self.rotations)
        
        att = torch.einsum('bqhd,bkhd->bhqk', q, k) / (self.embed_dim ** 0.5)
        att = att.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))
        att = nn.functional.softmax(att, dim=-1)
        y = torch.einsum('bhqk,bkhd->bqhd', att, v)
        return self.dropout(self.w_o(y.reshape(batch_size, seq_len, -1)))


def rotary_positional_embedding(q, k, rotations):
    q_shape, k_shape = q.shape, k.shape
    q = torch.view_as_complex(q.view(*q.shape[:-1], -1, 2))
    k = torch.view_as_complex(k.view(*k.shape[:-1], -1, 2))
    q = torch.view_as_real(q * torch.view_as_complex(rotations[:, :q.shape[1], :, :]))
    k = torch.view_as_real(k * torch.view_as_complex(rotations[:, :k.shape[1], :, :]))
    return q.view(q_shape), k.view(k_shape)
