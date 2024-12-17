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
from foreduce.transformer.model import GraphModel, Model


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

    if args.strategy.isdigit():
        success = 0
        for dir, problem in (pbar := tqdm(
            [(d, p) for d in os.listdir('./problems/') for p in os.listdir('./problems/' + d)],
            desc=f'Proving with {args.strategy}'
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
            if 'Refutation found' not in vampire.proof and 'Satisfiable' not in vampire.proof:
                continue
            with open(f'./proofs/{args.strategy}/{dir}/{problem}', 'w') as f:
                f.write(vampire.proof)
                success += 1
        pbar.set_postfix({'Success': success})
    else:
        # keep tracks of reasons for termination
        success = 0
        reached_max_steps = 0
        recursion_error = 0
        node_limit = 0
        other = 0
        others = []
        
        model = Model.load_from_checkpoint(f'./models/{args.strategy}.ckpt').to('cpu')

        for dir, problem in (pbar := tqdm(
            [(d, p) for d in os.listdir('./problems/') for p in os.listdir('./problems/' + d)],
            desc='Proving'
        )):
            os.makedirs(f'./proofs/{args.strategy}/{dir}', exist_ok=True)
            graph = None
            pbar.set_postfix({ 'Problem': problem, 'Success': success, 'Max Steps': reached_max_steps, 'Recursion Error': recursion_error, 'Node Limit': node_limit, 'Other': other })
            with VampireInteractive(VAMPIRE, f'./problems/{dir}/{problem}') as interactive:
                symbols = interactive.problem.function_symbols() | interactive.problem.predicate_symbols()
                permutation = torch.randperm(32)
                name = {s: permutation[i] + 1 for i, s in enumerate(symbols)}
                while not interactive.finished and interactive.step_count < args.max_steps and (
                    graph is None or args.max_nodes is None or len(graph) < args.max_nodes
                ):
                    if graph is None:
                        graph, mapping, clauses = interactive.problem.to_graph(depth=args.max_depth)
                    else:
                        try:
                            graph, mapping, clauses = interactive.problem.extend_graph(graph, mapping, len(clauses), depth=args.max_depth)
                        except RecursionError:
                            recursion_error += 1
                            break
                    data = from_networkx(graph)
                    data.type = torch.tensor([_type_mapping[t] for t in data.type], dtype=torch.int)
                    data.name = torch.tensor([name[s] if s is not None else 0 for s in data.name], dtype=torch.int)
                    data.arity = torch.tensor([min(args.max_arity + 1, a + 1) if a is not None else 0 for a in data.arity], dtype=torch.int)
                    data.pos = torch.tensor([min(args.max_arity + 1, a + 1) if a is not None else 0 for a in data.pos], dtype=torch.int)
                    data.clauses = index_to_mask(torch.tensor(clauses), size=data.num_nodes)
                    score, topk = model.predict(data)
                    deduped = []
                    for clause in interactive.problem.clauses:
                        if clause not in deduped:
                            deduped.append(clause)
                    _mapping = {i: deduped.index(clause) for i, clause in enumerate(interactive.problem.clauses)}
                    vals = [(score[topk.index(_mapping[i])].item(), i) for i in range(len(interactive.problem.clauses)) if (not interactive.active[i] and _mapping[i] in topk)]
                    _, next_clause = max(vals)
                    interactive.step(next_clause)
                if 'Refutation found.' in interactive.proof or 'Satisfiable' in interactive.proof:
                    success += 1
                    with open(f'./proofs/{args.strategy}/{dir}/{problem}', 'w') as f:
                        f.write(interactive.proof)
                elif interactive.step_count >= args.max_steps:
                    reached_max_steps += 1
                elif graph is None or (args.max_nodes is not None and len(graph) >= args.max_nodes):
                    node_limit += 1
                else:
                    other += 1
                    others.append(problem)
                    
            pbar.set_postfix({"Success" : success, "Max Steps" : reached_max_steps, "Recursion Error" : recursion_error, "Node Limit" : node_limit, "Other" : other})
        print("Uncategorized failure:")
        for o in others:
            print(o)
