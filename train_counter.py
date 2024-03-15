from counter_model import *

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

torch.set_float32_matmul_precision('medium')

data = SequenceDataset(10000, 50, .1, .0, .0, .01)
train_ratio = 0.8
train_samples = int(train_ratio * len(data))
train, val  = torch.utils.data.random_split(data, [train_samples, len(data) - train_samples])

train_loader = torch.utils.data.DataLoader(train, batch_size=500, num_workers=15)
val_loader = torch.utils.data.DataLoader(val, batch_size=500, num_workers=15)

model = MambaModel.load_from_checkpoint("mamba-counter.ckpt")

logger = WandbLogger(project="fo-reduce", group="mamba-counter")
trainer = pl.Trainer(max_epochs=10, logger=logger, log_every_n_steps=4)

trainer.fit(
    model,
    train_loader,
    val_loader,
)

trainer.save_checkpoint("mamba-counter.ckpt")

wandb.finish()