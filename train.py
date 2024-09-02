import argparse
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import signal
import sys
import torch
import wandb

from foreduce.transformer.model import Model
from foreduce.data.data import ProofEmbeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--clause_embed_layers", type=int, default=8)
    parser.add_argument("--clause_num_heads", type=int, default=8)
    parser.add_argument("--clause_embed_dim", type=int, default=64)
    parser.add_argument("--problem_embed_layers", type=int, default=16)
    parser.add_argument("--problem_num_heads", type=int, default=16)
    parser.add_argument("--problem_embed_dim", type=int, default=512)
    parser.add_argument("--from_checkpoint", type=str, default=None)
    args = parser.parse_args()
    
    
    def signal_handler(sig, frame):
        signal.signal(sig, signal.SIG_IGN)
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    
    torch.set_float32_matmul_precision('medium')
    
    dataset = ProofEmbeddings.from_file("./proofs_last.pt")
    
    if args.from_checkpoint is not None:
        model = Model.load_from_checkpoint(args.from_checkpoint)
    else:
        model = Model({
            "clause_length": dataset.x.size(2),
            "clause_embed_layers": args.clause_embed_layers,
            "clause_num_heads": args.clause_num_heads,
            "clause_embed_dim": args.clause_embed_dim,
            "problem_length": dataset.x.size(1),
            "problem_embed_layers": args.problem_embed_layers,
            "problem_num_heads": args.problem_num_heads,
            "problem_embed_dim": args.problem_embed_dim,
        }, dataset.config)

    logger = WandbLogger(project="vampire_select", group="test")
    
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        logger=logger,
        accumulate_grad_batches=1024,
        log_every_n_steps=1,
        gradient_clip_algorithm='value',
        callbacks=[ModelCheckpoint(every_n_train_steps=1)],
        gradient_clip_val=1.0,
    )
    trainer.fit(model, train_loader)
    wandb.finish()

