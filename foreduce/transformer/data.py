import torch
import os
from lightning import LightningDataModule


class IntegerSortingConfig:
    def __init__(self, low=0, high=10, seq_len=20, num_samples=50000):
        self.low = low
        self.high = high
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __repr__(self):
        return f"IntegerConfig{self.low}-{self.high}-{self.seq_len}-{self.num_samples}"


class IntegerSortingDataset(torch.utils.data.Dataset):
    def __init__(self, config=IntegerSortingConfig(), overwrite=False):
        self.config = config
        self.overwrite = overwrite
        self.generate(self.config.num_samples)
        self.inputs, self.targets = torch.load(repr(self.config) + ".pt")

    def __len__(self):
        return self.config.num_samples

    def __getitem__(self, idx):
        name = repr(self.config)
        return self.inputs[idx], self.targets[idx]

    def generate(self, n):
        name = repr(self.config)
        if not self.overwrite and os.path.exists(name + ".pt"):
            return
        inputs = torch.randint(
            self.config.low, self.config.high,
            size=(n, self.config.seq_len)
        )
        targets, _ = inputs.sort()
        inputs = torch.cat((inputs, targets), dim=-1)[:, :-1]
        with open(name + ".pt", "wb") as f:
            torch.save((inputs, targets), f)


class IntegerSortingDataModule(LightningDataModule):
    def __init__(self, config=IntegerSortingConfig(), batch_size=32):
        super().__init__()
        self.config = config
        self.batch_size = batch_size

    def setup(self, stage=None):
        data = IntegerSortingDataset(self.config)
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
