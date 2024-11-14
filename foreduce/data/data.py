import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import index_to_mask
from torch_geometric.utils.convert import from_networkx

from foreduce.transformer.tokenizer import TokenConfig, ProofEmbedder, ProofTokenizer


_type_mapping = {
    'symbol': 0,
    'term': 1,
    'variable': 2,
    'literal': 3,
    'negated_literal': 4,
    'clause': 5,
    '$false': 6,
    'eq': 7
}

class GraphDataset(InMemoryDataset):
    def __init__(self, max_arity=8):
        super().__init__()
        self.max_arity = max_arity
        self._data = []

    def add_proof(self, problem, tree, limit=None):
        graph, _, clauses, labels = problem.to_graph_data(tree, limit)
        data = from_networkx(graph)
        data.type = torch.tensor([_type_mapping[t] for t in data.type], dtype=torch.int)
        data.arity = torch.tensor([min(self.max_arity + 1, a + 1) if a is not None else 0 for a in data.arity], dtype=torch.int)
        data.pos = torch.tensor([min(self.max_arity + 1, a + 1) if a is not None else 0 for a in data.pos], dtype=torch.int)
        data.clauses = index_to_mask(torch.tensor(clauses), size=data.num_nodes)
        data.labels = torch.tensor(labels, dtype=torch.bool)
        self._data.append(data)

    def __len__(self):
        return len(self._data)

    def get(self, idx):
        return self._data[idx]

    def __getitem__(self, idx):
        return self._data[idx]
    
    def save(self, path):
        torch.save((self.max_arity, self._data), path)
        
    def load(path):
        max_arity, data = torch.load(path)
        dataset = GraphDataset(max_arity)
        dataset._data = data
        return dataset


class ProofTokens(Dataset):
    def __init__(self, config, seq_len=128, seed=42):
        self.max_tokens = seq_len
        self.config = config
        self.problems = []
        
        self.tokenizer = ProofTokenizer(config, seq_len, seed)
        
        self.index = 0
        self._tokens = torch.zeros(0, seq_len, dtype=torch.int)
        self.x = torch.zeros(1, 2, dtype=torch.int)
        self.target = torch.zeros(1, dtype=torch.float)
        self.weight = torch.zeros(1, dtype=torch.float)

    def add_proof(self, problem, tree, mapping=None):
        assert self.index < len(self.x)
        _tokens, x, target, weight, _ = self.tokenizer(problem, tree, mapping)
        while self.index + len(x) > len(self.x):
            self.x = torch.cat([self.x, torch.zeros_like(self.x)], dim=0)
            self.target = torch.cat([self.target, torch.zeros_like(self.target)], dim=0)
            self.weight = torch.cat([self.weight, torch.zeros_like(self.weight)], dim=0)
        self.x[self.index:self.index+len(x)] = x + len(self._tokens)
        self.target[self.index:self.index+len(x)] = target
        self.weight[self.index:self.index+len(x)] = weight
        self.index += len(x)

        self._tokens = torch.cat([self._tokens, _tokens], dim=0)
    
    def to_file(self, path):
        torch.save((
            self._tokens,
            self.x[:self.index],
            self.target[:self.index],
            self.weight[:self.index],
            self.config,
            self.index
        ), path)
        
    def from_file(path):
        torch.serialization.add_safe_globals([TokenConfig])
        _tokens, x, target, weight, config, index = torch.load(path, weights_only=True)
        tokens = ProofTokens(config, len(_tokens[0]))
        tokens._tokens = _tokens
        tokens.x = x
        tokens.target = target
        tokens.weight = weight
        tokens.index = index
        return tokens
    
    def __len__(self):
        return self.index
    
    def __getitem__(self, idx):   
        return self._tokens[self.x[idx, 0]], self._tokens[self.x[idx, 1]], self.target[idx], self.weight[idx]


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
        self.weight = torch.zeros(proofs, dtype=torch.float)

    def add_proof(self, problem, tree, mapping=None, goal='random'):
        assert self.index < len(self.x)
        x, y, target, weight, _ = self.tokenizer(problem, tree, mapping, goal)
        self.x[self.index] = x
        self.y[self.index] = y
        self.target[self.index] = target
        self.weight[self.index] = weight
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
        return self.x[idx], self.y[idx], self.target[idx], self.weight[idx]
