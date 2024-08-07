import torch
from torch.utils.data import Dataset

from foreduce.transformer.tokenizer import ProofTokenizer

class VampireProofs(Dataset):
    def __init__(self, config, max_steps=1024, max_tokens=128, seed=42):
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.config = config
        self.problems = []
        
        self.tokenizer = ProofTokenizer(config, max_steps, max_tokens, seed)
        
        self.x = torch.zeros(0, max_steps, max_tokens)
        self.y = torch.zeros(0, max_steps)
        self.target = torch.zeros(0, max_tokens)

    def add_proof(self, problem, tree, mapping=None, goal='random'):
        x, y, target, _ = self.tokenizer(problem, tree, mapping, goal)
        self.x = torch.cat((self.x, x.unsqueeze(0)))
        self.y = torch.cat((self.y, y.unsqueeze(0)))
        self.target = torch.cat((self.target, target.unsqueeze(0)))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.target[idx]
