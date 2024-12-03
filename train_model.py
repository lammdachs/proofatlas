import argparse
import torch
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
import wandb
from torch_geometric.loader import DataLoader

from foreduce.data.data import GraphDataset
from foreduce.transformer.model import GraphModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--accumulate_grad_batches", type=int)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--name", type=str)
    args = parser.parse_args()
    
    torch.set_float32_matmul_precision('medium')

    trainset = GraphDataset.load(f'./data/{args.dataset}_train.pt')
    valset = GraphDataset.load(f'./data/{args.dataset}_val.pt')
    model = GraphModel(8, trainset.max_arity, args.layers, args.dim)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    logger = WandbLogger(project='vampire_select', group='gnn')
    trainer = Trainer(
        max_epochs=args.epochs,
        logger=logger,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=1,
        devices=4,
        enable_progress_bar=False
    )
    trainer.fit(model, trainloader, valloader)
    wandb.finish()
    
    trainer.save_checkpoint(f'./models/{args.name}.ckpt')
