import torch
import os
from lightning import LightningDataModule
from foreduce.fol.random import RandomSignature, RandomSubstitutionProof
from tqdm import tqdm

from foreduce.transformer.modelargs import ModelArgs


class FirstOrderConfig:
    def __init__(self, args=ModelArgs(), num_samples=100_000, derivation_depth=3):
        self.args = args
        self.num_samples = num_samples
        self.derivation_depth = derivation_depth

    def __repr__(self):
        return f"FO-{self.args.function_arity}-\
            {self.args.predicate_arity}-\
            {self.args.variable_count}-\
            {self.num_samples}-\
            {self.derivation_depth}".replace(" ", "")


class FirstOrderDataset(torch.utils.data.Dataset):
    def __init__(self, config=FirstOrderConfig(), overwrite=False):
        self.config = config
        self.overwrite = overwrite
        self.generate()
        inputs, self.masks = torch.load(repr(self.config) + ".pt")
        self.inputs = inputs.long()

    def __len__(self):
        return self.config.num_samples

    def __getitem__(self, idx):
        name = repr(self.config)
        return self.inputs[idx], self.masks[idx]

    def generate(self):
        name = repr(self.config)
        if not self.overwrite and os.path.exists(name + ".pt"):
            return
        sig = RandomSignature(self.config.args)
        proofs = [self.config.args.encode_substitution_proof(
            RandomSubstitutionProof(sig, self.config.derivation_depth)
        ) for _ in tqdm(
            range(int(self.config.num_samples)),
            desc='Generating proofs',
            miniters=1000
        )]
        inputs = [p[1][:self.config.args.seq_len] for p in proofs]
        inputs = [torch.concatenate([
            torch.tensor(input, dtype=torch.int8),
            torch.zeros(self.config.args.seq_len - len(input), dtype=torch.int8)
        ]) for input in inputs]
        inputs = torch.stack(inputs)
        lengths = [(min(p[0], self.config.args.seq_len), min(len(p[1]), self.config.args.seq_len)) for p in proofs]
        masks = torch.stack([torch.concatenate([
            torch.zeros(a, dtype=torch.int8),
            torch.ones(b - a, dtype=torch.int8),
            torch.zeros(self.config.args.seq_len - b, dtype=torch.int8)
        ]) for a, b in lengths])
        with open(name + ".pt", "wb") as f:
            torch.save((inputs, masks), f)


class FirstOrderDataModule(LightningDataModule):
    def __init__(self, config=FirstOrderConfig(), batch_size=32):
        super().__init__()
        self.config = config
        self.batch_size = batch_size

    def setup(self, stage=None):
        data = FirstOrderDataset(self.config)
        n = len(data)
        train, val = torch.utils.data.random_split(
            data, [int(n*0.8), n - int(n*0.8)]
        )
        self.train_data = train
        self.val_data = val

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

    def test_dataloader(self):
        return self.val_dataloader()
