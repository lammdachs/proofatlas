import argparse
import os
import torch
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
import wandb
from torch_geometric.data.lightning import LightningDataset

from foreduce.data.data import GraphDataset
from foreduce.transformer.model import GraphModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--accumulate_grad_batches", type=int)
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--n_layers", type=int)
    args = parser.parse_args()
    
    torch.set_float32_matmul_precision('medium')

    dataset = GraphDataset.load('gnn/dataset.pt')
    model = GraphModel(8, dataset.max_arity, args.n_layers, args.dim)
    dataset = LightningDataset(dataset)
    
    
    logger = WandbLogger(project='vampire_select', group='generalization')
    trainer = Trainer(max_epochs=args.epochs, logger=logger, accumulate_grad_batches=args.accumulate_grad_batches, log_every_n_steps=1, devices=4, enable_progress_bar=False)
    trainer.fit(model, DataLoader(dataset, batch_size=args.batch_size, num_workers=4))
    wandb.finish()
    
    trainer.save_checkpoint('generalization/model.ckpt')
