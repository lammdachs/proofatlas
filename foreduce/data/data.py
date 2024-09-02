import torch
from torch.utils.data import Dataset

from foreduce.transformer.tokenizer import TokenConfig, ProofEmbedder, ProofTokenizer

class ProofTokens(Dataset):
    def __init__(self, config, seq_len=128, seed=42):
        self.max_tokens = seq_len
        self.config = config
        self.problems = []
        
        self.tokenizer = ProofTokenizer(config, seq_len, seed)
        
        self.index = 0
        self.x = torch.zeros(1, seq_len, dtype=torch.int)
        self.y = torch.zeros(1, seq_len, dtype=torch.int)
        self.target = torch.zeros(1, dtype=torch.float)

    def add_proof(self, problem, tree, mapping=None):
        assert self.index < len(self.x)
        x, y, target, _ = self.tokenizer(problem, tree, mapping)
        while self.index + len(x) > len(self.x):
            self.x = torch.cat([self.x, torch.zeros_like(self.x)], dim=0)
            self.y = torch.cat([self.y, torch.zeros_like(self.y)], dim=0)
            self.target = torch.cat([self.target, torch.zeros_like(self.target)], dim=0)
        self.x[self.index:self.index+len(x)] = x
        self.y[self.index:self.index+len(y)] = y
        self.target[self.index:self.index+len(target)] = target
        self.index += len(x)
        
    def to_file(self, path):
        assert self.index == len(self.x)
        torch.save((self.x, self.y, self.target, self.config), path)
        
    def from_file(path):
        torch.serialization.add_safe_globals([TokenConfig])
        x, y, target, config = torch.load(path, weights_only=True)
        tokens = ProofTokens(config, len(x))
        tokens.x = x
        tokens.y = y
        tokens.target = target
        return tokens
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.target[idx]


class ProofEmbeddings(Dataset):
    def __init__(self, config, proofs, max_steps=1024, max_tokens=128, seed=42):
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.config = config
        self.problems = []
        
        self.tokenizer = ProofEmbedder(config, max_steps, max_tokens, seed)
        
        self.index = 0
        self.x = torch.zeros(proofs, max_steps, max_tokens, dtype=torch.int)
        self.y = torch.zeros(proofs, max_steps, dtype=torch.int)
        self.target = torch.zeros(proofs, max_tokens, dtype=torch.int)

    def add_proof(self, problem, tree, mapping=None, goal='random'):
        assert self.index < len(self.x)
        x, y, target, _ = self.tokenizer(problem, tree, mapping, goal)
        self.x[self.index] = x
        self.y[self.index] = y
        self.target[self.index] = target
        self.index += 1

    def to_file(self, path):
        assert self.index == len(self.x)
        torch.save((self.x, self.y, self.target, self.config), path)

    def from_file(path):
        torch.serialization.add_safe_globals([TokenConfig])
        x, y, target, config = torch.load(path, weights_only=True)
        proofs = ProofEmbeddings(config, len(x))
        proofs.x = x
        proofs.y = y
        proofs.target = target
        return proofs

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.target[idx]
