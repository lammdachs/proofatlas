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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=10_000_000)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--inner_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--predicate_arity", type=int, nargs="+", default=[4, 2])
    parser.add_argument("--function_arity", type=int, nargs="+", default=[4, 4, 2])
    parser.add_argument("--variable_count", type=int, default=4)
    parser.add_argument("--derivation_depth", type=int, default=3)

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
    model = Transformer(margs)
    
    foconfig = FirstOrderConfig(
        args=margs,
        num_samples=args.num_samples,
        derivation_depth=args.derivation_depth
    )
    datamodule = FirstOrderDataModule(
        foconfig,
        batch_size=args.batch_size
    )
    logger = WandbLogger(project="fo-reduce", group="integer-sorting")
    
    trainer = Trainer(
        min_epochs=args.num_epochs,
        max_epochs=args.num_epochs,
        logger=logger,
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
    