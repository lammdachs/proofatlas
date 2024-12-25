import argparse
from dotenv import load_dotenv
import os
import sys
import torch
from torch_geometric.utils import index_to_mask
from torch_geometric.utils.convert import from_networkx
from tqdm import tqdm

from foreduce.vampire.vampire import VampireInteractive, VampireAutomatic
from foreduce.data.data import _type_mapping
from foreduce.transformer.model import GraphModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=32)
    parser.add_argument("--strategy", type=str, default='666')
    parser.add_argument("--max_arity", type=int, default=8)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--max_nodes", type=int, default=2**12)
    args = parser.parse_args()
    
    load_dotenv()
    VAMPIRE = os.getenv('VAMPIRE')

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
        model = GraphModel.load_from_checkpoint(f'./models/{args.strategy}.ckpt')

        for dir, problem in (pbar := tqdm(
            [(d, p) for d in os.listdir('./problems/') for p in os.listdir('./problems/' + d)],
            desc='Proving'
        )):
            os.makedirs(f'./proofs/{args.strategy}/{dir}', exist_ok=True)
            graph = None
            pbar.set_postfix({'Success': success, 'Problem': problem})
            with VampireInteractive(VAMPIRE, f'./problems/{dir}/{problem}') as interactive:
                while not interactive.finished and interactive.step_count < args.max_steps and (
                    graph is None or len(graph) < args.max_nodes
                ):
                    if graph is None:
                        graph, mapping, clauses = interactive.problem.to_graph(depth=args.max_depth)
                    else:
                        try:
                            graph, mapping, clauses = interactive.problem.extend_graph(graph, mapping, len(clauses), depth=args.max_depth)
                        except RecursionError:
                            break
                    symbols = problem.function_symbols() | problem.predicate_symbols()
                    permutation = torch.randperm(32)
                    name = {s: permutation[i] + 1 for i, s in enumerate(symbols)}
                    data = from_networkx(graph)
                    data.type = torch.tensor([_type_mapping[t] for t in data.type], dtype=torch.int)
                    data.name = torch.tensor([name[s] if s is not None else 0 for s in data.name], dtype=torch.int)
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
                    _, next_clause = max(vals)
                    interactive.step(next_clause)
                if 'Refutation found.' in interactive.proof:
                    success += 1
                    with open(f'./proofs/{args.strategy}/{dir}/{problem}', 'w') as f:
                        f.write(interactive.proof)
            pbar.set_postfix({"Success" : success})
                    
