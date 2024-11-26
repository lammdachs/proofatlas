import argparse
from dotenv import load_dotenv
import os
from random import choices

from foreduce.vampire.vampire import VampireAutomatic
from foreduce.data.data import GraphDataset, _type_mapping
from foreduce.transformer.model import GraphModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--accumulate_grad_batches", type=int, default=8)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=32)
    parser.add_argument("--clause_selection", type=str, default='666')
    parser.add_argument("--cont", type=bool, default=False)
    parser.add_argument("--max_arity", type=int, default=8)
    parser.add_argument("--graphs_per_problem", type=int, default=8)
    parser.add_argument("--max_nodes_problem", type=int, default=2**10)
    parser.add_argument("--max_nodes_proof", type=int, default=2**12)
    args = parser.parse_args()
    
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
                    activation_limit=args.max_steps
                )
                try:
                    vampire.run() 
                except:
                    print('Failed to parse')
                    continue
                if 'Refutation found' not in vampire.proof:
                    print('Failed')
                    continue
                with open('gnn/proofs/' + dir + '/' + problem, 'w') as f:
                    f.write(vampire.proof)
                graph, _, _ = vampire.problem.to_graph()
                if graph.number_of_nodes() > args.max_nodes_proof:
                    print('Proof too large')
                    continue
                print('Success')
                minimum = [d == [] for d in vampire.tree].index(False)
                limits = choices(range(minimum, len(vampire.tree)), k=args.graphs_per_problem)
                for limit in limits:
                    dataset.add_proof(vampire.problem, vampire.tree, limit)
    
    dataset.save('gnn/dataset.pt')