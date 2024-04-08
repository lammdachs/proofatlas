from bidict import bidict
import numpy.random as random
from tqdm import tqdm
import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from mamba_ssm import Mamba
from pytorch_lightning import LightningModule
from os.path import isfile


from foreduce.transformer.decoder import RMSNorm

mapping = bidict({"0": 0, "1" : 1, "incr" : 2, "decr" : 3, "initial" : 4, "set" : 5, " " : 6})

digits=32

def encode(x):
    encoding = [i for i in f"{x:0{digits}b}"]
    return [mapping[i] for i in encoding]

def randomSequences(n, length=100, p_incr=0.8, p_decr=0.15, p_initial=0.03, p_set=0.01):
    sequences = []
    results = []
    initial = random.randint(0, 2**digits-1, size=n)
    p_none = 1 - (p_incr + p_decr + p_initial + p_set)
    events = random.choice(
        ["incr", "decr", "initial", "set", " "],
        p=[p_incr, p_decr, p_initial, p_set, p_none],
        size=(n, length-1)
    )
    for n, eventchain in tqdm(zip(initial, events), desc="Generating sequences", total=n):
        sequence, result = [], []
        starting, counter = n, n
        sequence.append(
            [mapping["set"]] + encode(counter)
        )
        result.append(
            encode(counter)
        )
        for event in eventchain:
            if event == "incr":
                sequence.append([mapping["incr"]] + encode(random.randint(0, 2**digits-1)))
                counter = min(2**digits-1, counter + 1)
                result.append(
                    encode(counter)
                )
            if event == "decr":
                sequence.append([mapping["decr"]] + encode(random.randint(0, 2**digits-1)))
                counter = max(0, counter - 1)
                result.append(
                    encode(counter)
                )
            if event == "initial":
                sequence.append([mapping["initial"]] + encode(random.randint(0, 2**digits-1)))
                result.append(
                    encode(starting)
                )
            if event == "set":
                starting = random.randint(0, 2**digits-1)
                sequence.append(
                    [mapping["set"]] + encode(starting)
                )
                counter = starting
                result.append(
                    encode(counter)
                )
            if event == " ":
                sequence.append([mapping[" "]] + encode(random.randint(0, 2**digits-1)))
                result.append(
                    encode(counter)
                )
        sequences.append(sequence)
        results.append(result)
    return sequences, results

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, n=10000, length=100, p_incr=0.8, p_decr=0.15, p_initial=0.03, p_set=0.01, use_file=True, dir="./datasets/"):
        self.n = n
        self.length = length
        self.p_incr = p_incr
        self.p_decr = p_decr
        self.p_initial = p_initial
        self.p_set = p_set
        if use_file and isfile(dir + repr(self) + ".pt"):
            self.x, self.y = torch.load(dir + repr(self) + ".pt")
        else:
            x, y = randomSequences(n, length, p_incr, p_decr, p_initial, p_set)
            self.x, self.y = torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
            if use_file:    
                torch.save((self.x, self.y), dir +  repr(self) + ".pt")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __repr__(self):
        return f"SequenceDataset(n={self.n},length={self.length},p_incr={self.p_incr},p_decr={self.p_decr},p_initial={self.p_initial},p_set={self.p_set})"


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        assert dim % n_head == 0
        self.c_attn = nn.Linear(dim, 3 * dim)
        self.c_proj = nn.Linear(dim, dim)
        self.n_head = n_head
        self.n_embd = dim

    def forward(self, x):
        B, T, C = x.size()
        
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_head)

    def forward(self, x):
        return x + self.attn(self.norm(x))


class MambaBlock(nn.Module):
    def __init__(self, dim=16, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba(dim, d_state, d_conv, expand)
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = x + self.mamba(self.norm(x))
        return x

    def inference(self, batch_size=1):
        """Prepare the model for inference.

        Args:
            batch_size (int, optional): The batch size of sequences to infer for. Defaults to 1.
        """
        self.conv_state = torch.zeros(
            batch_size, self.mamba.d_model * 2, self.mamba.d_conv
        ).to(self.mamba.D)
        self.ssm_state = torch.zeros(
            batch_size, self.mamba.d_model * 2, self.mamba.d_state
        ).to(self.mamba.D)

    def step(self, x):
        """Step the model forward by one token.

        Args:
            x (torch.Tensor): The input token.

        Returns:
            torch.Tensor: The output token.
        """
        if not hasattr(self, "conv_state") or not hasattr(self, "ssm_state"):
            raise ValueError("Model not in inference mode. Call `model.inference()` first.")
        out, self.conv_state, self.ssm_state = self.mamba.step(
            self.norm(x), self.conv_state, self.ssm_state)
        return x + out

class MambaModel(LightningModule):
    def __init__(self, n_layers=8, dim=1024, d_tf=128, dropout=0.0, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_len = digits + 1
        self.output_len = digits
        self.n_layers = n_layers
        self.d_tf = d_tf
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        
        self.dropout = nn.Dropout(dropout)
        self.t_emb = torch.nn.Embedding(len(mapping), self.d_tf)
        self.p_emb = torch.nn.Embedding(self.input_len, self.d_tf)
        self.encoder = Block(self.d_tf, 4)
        self.proj = torch.nn.Linear(self.input_len * self.d_tf, self.dim)
        self.mamba_blocks = torch.nn.ModuleList([
            MambaBlock(self.dim, d_state, d_conv, expand) for _ in range(n_layers)
        ])
        self.out = torch.nn.Linear(self.dim, digits*len(mapping))
        
        self.register_buffer("pos", torch.arange(self.input_len, dtype=torch.int))
        
        self.save_hyperparameters()

    def forward(self, x):
        batches, length, input_len = x.shape
        emb = self.dropout(self.t_emb(x) + self.p_emb(self.pos)).view(-1, input_len, self.d_tf)
        enc = self.encoder(emb).view(batches, length, input_len * self.d_tf)
        out = self.proj(enc)
        for block in self.mamba_blocks[:-1]:
            out = self.dropout(block(out))
        out = self.mamba_blocks[-1](out)
        return F.log_softmax(self.out(out).view(batches, length, self.output_len, len(mapping)), dim=-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.nll_loss(y_hat.view(-1, len(mapping)), y.view(-1))
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat.view(-1, len(mapping)), y.view(-1))
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def inference(self, batch_size=1):
        for layer in self.mamba_blocks:
            layer.inference(batch_size=batch_size)

    def step(self, x):
        batches, length, input_len = x.shape
        emb = self.dropout(self.t_emb(x) + self.p_emb(self.pos)).view(-1, input_len, self.d_tf)
        enc = self.encoder(emb).view(batches, length, self.input_len * self.d_tf)
        out = self.proj(enc)
        for i in range(length):
            for block in self.mamba_blocks[:-1]:
                out[:, i:i+1] = block.step(out[:, i:i+1])
            out[:, i:i+1] = self.mamba_blocks[-1].step(out[:, i:i+1])
        return F.log_softmax(self.out(out).view(batches, length, self.output_len, len(mapping)), dim=-1)

    def get_state(self):
        return [(layer.conv_state, layer.ssm_state) for layer in self.mamba_blocks]

    def set_state(self, state):
        for layer, (conv, ssm) in zip(self.mamba_blocks, state):
            layer.conv_state = conv
            layer.ssm_state = ssm