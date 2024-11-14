VAMPIRE_PATH = '/home/apluska/.vampire/bin/vampire_z3_rel_static_casc2023_6749'
TPTP_PATH = '/home/apluska/TPTP-v8.2.0/'


import argparse
import os
from collections import defaultdict
import json
from subprocess import Popen, PIPE, STDOUT
import torch
from sortedcontainers import SortedList

from foreduce.vampire.vampire import VampireInteractive, VampireAutomatic
from foreduce.data.data import GraphDataset
from foreduce.transformer.tokenizer import TokenConfig
from foreduce.tptp.parser import read_file as read_tptp
from foreduce.vampire.parser import read_file as read_vampire
from foreduce.transformer.model import GraphModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=256)
    parser.add_argument("--clause_selection", type=str, default='10')
    parser.add_argument("--cont", type=bool, default=False)
    args = parser.parse_args()
    
    if not os.path.exists('gnn/dataset.pt'):
        os.makedirs('gnn/problems/', exist_ok=True)
        symbols = set()
        for problem in os.listdir(TPTP_PATH + 'Problems/PUZ/'):
            print(f"Processing {problem}...", end=' ')
            if not os.path.exists('gnn/problems/' + problem):
                try:
                    p = read_tptp(TPTP_PATH + 'Problems/PUZ/' + problem, include_path=TPTP_PATH, max_size=100_000)
                except:
                    print('Failed to parse')
                    continue
                with open('gnn/problems/' + problem, 'w') as f:
                    f.write(p.to_tptp())
                print('Success')
            else:
                p = read_tptp('gnn/problems/' + problem)        
                    

        dataset = GraphDataset(max_arity=8)
        os.makedirs('gnn/proofs/', exist_ok=True)
        for problem in os.listdir('gnn/problems/'):
            print(f"Proving {problem}...", end=' ')
            if not os.path.exists('gnn/proofs/' + problem):
                vampire = VampireAutomatic(
                    VAMPIRE_PATH,
                    'gnn/problems/' + problem,
                    selection=args.clause_selection,
                    activation_limit=64,
                )
                try:
                    vampire.run() 
                except:
                    print('Failed parsing')
                    continue
                if 'Refutation found' not in vampire.proof:
                    print('Failed proving')
                    continue
                with open('gnn/proofs/' + problem, 'w') as f:
                    f.write(vampire.proof)
                print('Success')
            dataset.add_proof(vampire.problem, vampire.tree)
        dataset.save('gnn/dataset.pt')
        
    if not os.path.exists('gnn/model.ckpt') or args.cont:
        with Popen(
            [
                'python', '-u', 'train_gnn.py',
                '--epochs', str(args.epochs),
                '--batch_size', str(args.batch_size), '--accumulate_grad_batches', str(args.accumulate_grad_batches),
                '--dim', str(args.dim), '--n_layers', str(args.n_layers)
            ],
            bufsize=1, universal_newlines=True, stdout=PIPE, stderr=STDOUT
        ) as proc:
            for line in proc.stdout:
                print(line, end='')
            