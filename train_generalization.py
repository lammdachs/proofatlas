import argparse
import os
import torch
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
import wandb

from foreduce.data.data import ProofTokens
from foreduce.transformer.embedding import FormulaEmbedding


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--accumulate_grad_batches", type=int)
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--n_layers", type=int)
    parser.add_argument("--n_heads", type=int)
    args = parser.parse_args()
    
    torch.set_float32_matmul_precision('medium')

    train_dataset = ProofTokens.from_file('generalization/train_dataset.pt')
    val_dataset = ProofTokens.from_file('generalization/test_dataset.pt')

    if os.path.exists('generalization/model.ckpt'):
       embedding = FormulaEmbedding.load_from_checkpoint('generalization/model.ckpt')
    else:
       embedding = FormulaEmbedding(train_dataset.config, seq_len=args.seq_len, dim=args.dim, n_layers=args.n_layers, n_heads=args.n_heads)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, persistent_workers=True, drop_last=False)
    
    logger = WandbLogger(project='vampire_select', group='generalization')
    trainer = Trainer(max_epochs=args.epochs, logger=logger, accumulate_grad_batches=args.accumulate_grad_batches, log_every_n_steps=1, devices=4, enable_progress_bar=False)
    trainer.fit(embedding, train_loader, val_loader)
    wandb.finish()
    
    trainer.save_checkpoint('generalization/model.ckpt')
