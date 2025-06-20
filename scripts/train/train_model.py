import argparse
import torch
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger
from torch_geometric.loader import DataLoader
import os

from proofatlas.training.datasets.data import GraphDataset, _type_mapping
from proofatlas.models.hybrid.model import GraphModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--accumulate_grad_batches", type=int)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--name", type=str)
    args = parser.parse_args()
    
    torch.set_float32_matmul_precision('medium')

    trainset = GraphDataset.load(f'./data/{args.dataset}_train.pt')
    valset = GraphDataset.load(f'./data/{args.dataset}_val.pt')
    model = GraphModel(len(_type_mapping), trainset.max_arity, args.layers, args.dim)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create CSV logger
    log_dir = f'./.logs/{args.name}'
    os.makedirs(log_dir, exist_ok=True)
    logger = CSVLogger(save_dir=log_dir, name='', version='')
    
    trainer = Trainer(
        max_epochs=args.epochs,
        logger=logger,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=1,
        devices=4,
        enable_progress_bar=False,
        default_root_dir=log_dir  # This prevents lightning_logs creation
    )
    trainer.fit(model, trainloader, valloader)
    
    # Save checkpoint
    os.makedirs('./.models', exist_ok=True)
    trainer.save_checkpoint(f'./.models/{args.name}.ckpt')
