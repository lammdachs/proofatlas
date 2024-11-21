VAMPIRE_PATH = '/home/apluska/.vampire/bin/vampire_z3_rel_static_casc2023_6749'
TPTP_PATH = '/home/apluska/TPTP-v8.2.0/'


import argparse
import os
from subprocess import Popen, PIPE, STDOUT
from random import choices
import torch
from torch_geometric.utils import index_to_mask
from torch_geometric.utils.convert import from_networkx
from tqdm import tqdm

from foreduce.vampire.vampire import VampireInteractive, VampireAutomatic
from foreduce.data.data import GraphDataset, _type_mapping
from foreduce.tptp.parser import read_file as read_tptp
from foreduce.transformer.model import GraphModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--accumulate_grad_batches", type=int, default=4)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=64)
    parser.add_argument("--clause_selection", type=str, default='10')
    parser.add_argument("--cont", type=bool, default=False)
    parser.add_argument("--max_arity", type=int, default=8)
    parser.add_argument("--graphs_per_problem", type=int, default=8)
    args = parser.parse_args()
    
    if not os.path.exists('gnn/dataset.pt'):
        os.makedirs('gnn/problems/', exist_ok=True)
        symbols = set()
        for dir in os.listdir(TPTP_PATH + 'Problems'):
            os.makedirs('gnn/problems/' + dir, exist_ok=True)
            for problem in os.listdir(TPTP_PATH + 'Problems/' + dir):
                print(f"Processing {problem}...", end=' ')
                if not os.path.exists('gnn/problems/' + dir + '/' + problem):
                    try:
                        p = read_tptp(TPTP_PATH + 'Problems' + dir +'/' + problem, include_path=TPTP_PATH, max_size=100_000)
                    except:
                        print('Failed to parse')
                        continue
                    with open('gnn/problems/' + dir + '/' + problem, 'w') as f:
                        f.write(p.to_tptp())
                    print('Success')

        dataset = GraphDataset(max_arity=args.max_arity)
        for dir in os.listdir('gnn/problems/'):
            os.makedirs('gnn/proofs/' + dir, exist_ok=True)
            for problem in os.listdir('gnn/problems/' + dir):
                print(f"Proving {problem}...", end='')
                if not os.path.exists('gnn/proofs/' + dir + '/' + problem):
                    vampire = VampireAutomatic(
                        VAMPIRE_PATH,
                        'gnn/problems/' + dir + '/' + problem,
                        selection=args.clause_selection,
                        activation_limit=64,
                    )
                    try:
                        vampire.run() 
                    except:
                        print('Failed parsing')
                        continue
                    if 'Refutation found' not in vampire.proof:
                        print('Failed')
                        continue
                    with open('gnn/proofs/' + dir + '/' + problem, 'w') as f:
                        f.write(vampire.proof)
                    print('Success')
                    minimum = [d == [] for d in vampire.tree].index(False)
                    limits = choices(range(minimum, len(vampire.tree)), k=args.graphs_per_problem)
                    for limit in limits:
                        dataset.add_proof(vampire.problem, vampire.tree, limit)
        
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

    model = GraphModel.load_from_checkpoint('gnn/model.ckpt')

    os.makedirs('gnn/model_proofs/', exist_ok=True)
    for dir, problem in (pbar := tqdm(
        [(d, p) for d in os.listdir('gnn/problems/') for p in os.listdir('gnn/problems/' + d)],
        desc='Proving',
    )):
        graph = None
        with VampireInteractive(VAMPIRE_PATH, 'gnn/problems/' + dir + '/' + problem) as interactive:
            pbar.set_postfix({'Problem': problem})
            while not interactive.finished and interactive.step_count < args.max_steps:
                if graph is None:
                    graph, mapping, clauses = interactive.problem.to_graph()
                else:
                    graph, mapping, clauses = interactive.problem.extend_graph(graph, mapping, len(clauses))
                data = from_networkx(graph)
                data.type = torch.tensor([_type_mapping[t] for t in data.type], dtype=torch.int)
                data.arity = torch.tensor([min(args.max_arity + 1, a + 1) if a is not None else 0 for a in data.arity], dtype=torch.int)
                data.pos = torch.tensor([min(args.max_arity + 1, a + 1) if a is not None else 0 for a in data.pos], dtype=torch.int)
                data.clauses = index_to_mask(torch.tensor(clauses), size=data.num_nodes)
                score = model(data)
                deduped = []
                for clause in interactive.problem.clauses:
                    if clause not in deduped:
                        deduped.append(clause)
                _mapping = {i: deduped.index(clause) for i, clause in enumerate(interactive.problem.clauses)}
                vals = [(score[_mapping[i]].item(), i) for i in range(len(interactive.problem.clauses)) if not interactive.active[i]]
                if not vals:
                    print(interactive)
                    print(_mapping)
                    print(score)
                    print(interactive.mapping)
                    print([i for i in range(len(interactive.problem.clauses)) if not interactive.active[i]])
                _, next_clause = max(vals)
                interactive.step(next_clause)
            
            os.makedirs('gnn/model_proofs/' + dir, exist_ok=True)
            with open('gnn/model_proofs/' + dir + '/' + problem, 'w') as f:
                f.write(interactive.proof)
