import torch
import wandb
import argparse
from lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from foreduce.transformer.model import Transformer
from foreduce.transformer.data import IntegerSortingConfig, IntegerSortingDataModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--sequence_length", type=int, default=20)
    parser.add_argument("--num_digits", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=50000)

    args = parser.parse_args()
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    seq_len = args.sequence_length
    num_digits = args.num_digits
    num_samples = args.num_samples
    
    torch.set_float32_matmul_precision('medium')
    model = Transformer(Transformer.default_config())
    
    datamodule = IntegerSortingDataModule(
        config=IntegerSortingConfig(
            low=0, high=num_digits, seq_len=seq_len, num_samples=num_samples
        ),
        batch_size=batch_size
    )
    logger = WandbLogger(project="fo-reduce", group="integer-sorting")
    
    trainer = Trainer(
        min_epochs=num_epochs,
        max_epochs=num_epochs,
        logger=logger
    )
    trainer.fit(
        model,
        datamodule=datamodule
    )
    wandb.finish()
    