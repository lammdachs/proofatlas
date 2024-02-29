import os
import torch
import wandb
import argparse
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from foreduce.transformer.modelargs import ModelArgs

from foreduce.transformer.model import Transformer
from foreduce.transformer.data import FirstOrderConfig, FirstOrderDataModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=100_000)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--inner_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--predicate_arity", type=int, nargs="+", default=[4, 2])
    parser.add_argument("--function_arity", type=int, nargs="+", default=[4, 4, 2])
    parser.add_argument("--variable_count", type=int, default=4)
    parser.add_argument("--derivation_depth", type=int, default=3)
    parser.add_argument("--random_axioms", type=float, default=0.0)
    parser.add_argument("--from_checkpoint", type=str, default=None)

    args = parser.parse_args()
    
    torch.set_float32_matmul_precision('medium')
    
    margs = ModelArgs(
        embed_dim=args.embed_dim,
        seq_len=args.seq_len,
        num_heads=args.num_heads,
        dropout=args.dropout,
        inner_dim=args.inner_dim,
        num_layers=args.num_layers,
        predicate_arity=args.predicate_arity,
        function_arity=args.function_arity,
        variable_count=args.variable_count
    )
    if args.from_checkpoint is not None:
        model = Transformer.load_from_checkpoint(args.from_checkpoint)
        assert model.embed_dim == args.embed_dim
        assert model.seq_len == args.seq_len
        assert model.num_heads == args.num_heads
        assert model.inner_dim == args.inner_dim
        assert model.num_layers == args.num_layers
    else:
        model = Transformer(margs)
    
    foconfig = FirstOrderConfig(
        args=margs,
        num_samples=args.num_samples,
        derivation_depth=args.derivation_depth,
        random_axioms=args.random_axioms
    )
    datamodule = FirstOrderDataModule(
        foconfig,
        batch_size=args.batch_size
    )
    logger = WandbLogger(project="foreduce", group="foreduce")
    
    trainer = Trainer(
        min_epochs=args.num_epochs,
        max_epochs=args.num_epochs,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[ModelCheckpoint(
            dirpath="checkpoints",
            filename=str(foconfig) + "-{epoch:02d}",
        )]
    )
    trainer.fit(
        model,
        datamodule=datamodule
    )
    wandb.finish()
    