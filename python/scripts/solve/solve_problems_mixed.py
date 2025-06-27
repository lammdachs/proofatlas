import argparse
from dotenv import load_dotenv
import os
import numpy as np
import sys
import torch
from torch_geometric.data import Batch
from torch_geometric.utils.convert import from_networkx
from tqdm import tqdm

from proofatlas.parsers.vampire.vampire import VampireInteractive, VampireAutomatic
from proofatlas.training.datasets.data import _type_mapping
from proofatlas.models.hybrid.model import Model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=64)
    parser.add_argument("--strategy", type=str, default='666')
    parser.add_argument("--max_arity", type=int, default=8)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--max_nodes", type=int, default=2**10)
    args = parser.parse_args()
    
    load_dotenv()
    VAMPIRE = os.getenv('VAMPIRE')
    sys.setrecursionlimit(2**14)

    success = 0
    if args.strategy.isdigit():
        for dir, problem in (pbar := tqdm(
            [(d, p) for d in os.listdir('./problems/') for p in os.listdir('./problems/' + d)],
            desc=f'Proving with Vampire/{args.strategy}'
        )):
            pbar.set_postfix({'Success': success})
            os.makedirs(f'./proofs/{args.strategy}/{dir}', exist_ok=True)
            vampire = VampireAutomatic(
                VAMPIRE,
                f'./problems/{dir}/{problem}',
                selection=args.strategy,
                activation_limit=args.max_steps
            )
            try:
                vampire.run() 
            except:
                continue
            if 'Refutation found' not in vampire.proof:
                continue
            with open(f'./proofs/{args.strategy}/{dir}/{problem}', 'w') as f:
                f.write(vampire.proof)
                success += 1
        pbar.set_postfix({'Success': success})
    else:
        model = Model.load_from_checkpoint(f'./models/{args.strategy}.ckpt').to('cpu')

        for dir, problem in (pbar := tqdm(
            [(d, p) for d in os.listdir('./problems/') for p in os.listdir('./problems/' + d)],
            desc='Proving'
        )):
            os.makedirs(f'./proofs/{args.strategy}/{dir}', exist_ok=True)
            graphs = []
            datalist = []
            active = torch.zeros(1024, dtype=torch.bool)
            pbar.set_postfix({'Success': success, 'Problem': problem})
            with VampireInteractive(VAMPIRE, f'./problems/{dir}/{problem}') as interactive:                
                symbols = interactive.problem.function_symbols() | interactive.problem.predicate_symbols()
                permutation = torch.randperm(32)
                name = {s: permutation[i] + 1 for i, s in enumerate(symbols)}
                while not interactive.finished and interactive.step_count < args.max_steps and (
                    sum(len(g) for g in graphs) < args.max_nodes
                ) and (len(graphs) == 0 or not active[:len(graphs)].all()):
                    graphs = interactive.problem.extend(graphs, len(graphs), depth=args.max_depth)
                    for graph in graphs[len(datalist):]:
                        data = from_networkx(graph)
                        data.type = torch.tensor([_type_mapping[t] for t in data.type], dtype=torch.int)
                        data.name = torch.tensor([name[s] if s is not None else 0 for s in data.name], dtype=torch.int)
                        data.arity = torch.tensor([min(args.max_arity + 1, a + 1) if a is not None else 0 for a in data.arity], dtype=torch.int)
                        data.pos = torch.tensor([min(args.max_arity + 1, a + 1) if a is not None else 0 for a in data.pos], dtype=torch.int)
                        datalist.append(data)
                    score = model(Batch.from_data_list(datalist))
                    #given = torch.where(active[:len(graphs)], -10000, score).argmax().item()
                    given = np.random.choice(len(graphs), p=torch.where(active[:len(graphs)], -10000, score).softmax(dim=0).detach().numpy())
                    active[given] = True
                    interactive.step(given)
                if 'Refutation found.' in interactive.proof:
                    success += 1
                    with open(f'./proofs/{args.strategy}/{dir}/{problem}', 'w') as f:
                        f.write(interactive.proof)
            pbar.set_postfix({"Success" : success})
                    
