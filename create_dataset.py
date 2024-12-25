import argparse
import os
from random import choices

from sortedcontainers import SortedDict
from torch import randperm
from tqdm import tqdm

from foreduce.data.data import GraphDataset
from foreduce.fol.logic import Problem
from foreduce.vampire.parser import read_file as read_vampire

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default='666')
    parser.add_argument("--data_per_proof", type=int, default=64)
    parser.add_argument("--max_arity", type=int, default=8)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--train_val_split", type=float, default=0.8)
    args = parser.parse_args()
    
    trainset = GraphDataset(max_arity=args.max_arity)
    valset = GraphDataset(max_arity=args.max_arity)
    
    size = len([_ for d in os.listdir(f'./proofs/{args.strategy}') for _ in os.listdir(f'./proofs/{args.strategy}/{d}')])
    perm = randperm(size)
    
    for i, (dir, problem) in enumerate(pbar := tqdm(
        [
            (d, p)
            for d in os.listdir(f'./proofs/{args.strategy}')
            for p in os.listdir(f'./proofs/{args.strategy}/{d}')
        ],
        desc='Creating Dataset'
    )):
        problem, tree, mapping = read_vampire(
            f'./proofs/{args.strategy}/{dir}/{problem}',
            Problem(), [], SortedDict({}
        ))
        try:
            minimum = [d == [] for d in tree].index(False)
            limits = choices(range(minimum, len(tree)), k=args.data_per_proof)
        except ValueError:
            limits = [None for _ in range(args.data_per_proof)]
        for limit in limits:
            if perm[i] < size * args.train_val_split:
                trainset.add_proof(problem, tree, limit, depth=args.max_depth)
            else:
                valset.add_proof(problem, tree, limit, depth=args.max_depth)


    os.makedirs('data', exist_ok=True)
    trainset.save(f'data/{args.strategy}_train.pt')
    valset.save(f'data/{args.strategy}_val.pt')
