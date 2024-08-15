import torch
from torch.utils.data import Dataset

from foreduce.transformer.tokenizer import TokenConfig, ProofTokenizer

class VampireProofs(Dataset):
    def __init__(self, config, proofs, max_steps=1024, max_tokens=128, seed=42):
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.config = config
        self.problems = []
        
        self.tokenizer = ProofTokenizer(config, max_steps, max_tokens, seed)
        
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
        proofs = VampireProofs(config)
        proofs.x = x
        proofs.y = y
        proofs.target = target
        return proofs

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.target[idx]
