VAMPIRE_PATH = '/home/apluska/.vampire/bin/vampire_z3_rel_static_casc2023_6749'
TPTP_PATH = '/home/apluska/TPTP-v8.2.0/'


problems = [
    'GRP001-1.p',
    'GRP002-1.p',
    'GRP003-1.p',
    'GRP004-1.p',
    'GRP005-1.p',
    'GRP006-1.p',
    'GRP007-1.p',
    'GRP008-1.p',
    'GRP009-1.p',
    'GRP010-4.p',
    'GRP011-4.p',
    'GRP012-1.p',
    'GRP013-1.p',
    'GRP014-1.p',
    'GRP017-1.p',
    'GRP018-1.p',
    'GRP019-1.p',
    'GRP020-1.p',
    'GRP021-1.p',
    'GRP022-1.p',
    'GRP023-1.p',
    'GRP024-5.p',
    'GRP030-1.p',
    'GRP031-1.p'
]

import argparse
import os
from collections import defaultdict
import json
from subprocess import Popen, PIPE, CalledProcessError, STDOUT
import torch
from sortedcontainers import SortedList

from foreduce.vampire.vampire import VampireInteractive
from foreduce.vampire.vampire import VampireAutomatic
from foreduce.data.data import ProofTokens
from foreduce.transformer.tokenizer import TokenConfig
from foreduce.tptp.parser import read_file as read_tptp
from foreduce.vampire.parser import read_file as read_vampire
from foreduce.transformer.embedding import FormulaEmbedding
from torch.utils.data import DataLoader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="GRP/GRP001-1.p")
    parser.add_argument("--epochs", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accumulate_grad_batches", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=96)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=64)
    parser.add_argument("--clause_selection", type=str, default='10')
    args = parser.parse_args()

    os.makedirs('overfit/' + args.problem, exist_ok=True)
    
    if not os.path.exists('overfit/' + args.problem + '/problem.p'):
        problem = read_tptp(TPTP_PATH + 'Problems/' + args.problem, include_path=TPTP_PATH, max_size=100_000)
        with open('overfit/' + args.problem + '/problem.p', 'w') as f:
            f.write(problem.to_tptp())
    else:
        problem = read_tptp('overfit/' + args.problem + '/problem.p')

    if not os.path.exists('overfit/' + args.problem + '/config.json') or not os.path.exists('overfit/' + args.problem + '/mapping.json'):
        symbols = set()
        symbols.update(problem.function_symbols() | problem.predicate_symbols())
        arity_dict = defaultdict(list)
        for s in symbols:
            arity_dict[s.arity].append(s.name)
        arity_list = list(arity_dict.values())
        config = TokenConfig(num_functions=[len(s) for s in arity_list])
        with open('overfit/' + args.problem + '/config.json', 'w') as f:
            json.dump(config.to_dict(), f)
        function_mapping = config.random_function_mapping(arity_list)
        mapping = config.reserved_token_mapping | config.random_function_mapping(arity_list)
        with open('overfit/' + args.problem + '/mapping.json', 'w') as f:
            json.dump(mapping, f)
    else:
        with open('overfit/' + args.problem + '/config.json', 'r') as f:
            config = TokenConfig.from_dict(json.load(f))
        with open('overfit/' + args.problem + '/mapping.json', 'r') as f:
            mapping = json.load(f)

    if not os.path.exists('overfit/' + args.problem + '/proof.p') or not os.path.exists('overfit/' + args.problem + '/dataset.pt'):
        vampire = VampireAutomatic(VAMPIRE_PATH, 'overfit/' + args.problem + '/problem.p', selection=args.clause_selection)
        vampire.run() 
        with open('overfit/' + args.problem + '/proof.p', 'w') as f:
            f.write(vampire.proof)
        dataset = ProofTokens(config, seq_len=args.seq_len)
        dataset.add_proof(vampire.problem, vampire.tree, mapping)
        dataset.to_file('overfit/' + args.problem + '/dataset.pt')
    else:
        dataset = ProofTokens.from_file('overfit/' + args.problem + '/dataset.pt')
        
    if not os.path.exists('overfit/' + args.problem + '/model.ckpt'):
        with Popen(
            [
                'python', '-u', 'train_overfit.py', '--problem', args.problem,
                '--epochs', str(args.epochs),
                '--batch_size', str(args.batch_size), '--accumulate_grad_batches', str(args.accumulate_grad_batches),
                '--seq_len', str(args.seq_len),
                '--dim', str(args.dim), '--n_layers', str(args.n_layers), '--n_heads', str(args.n_heads)
            ],
            bufsize=1, universal_newlines=True, stdout=PIPE, stderr=STDOUT
        ) as proc:
            for line in proc.stdout:
                print(line, end='')
    
    embedding = FormulaEmbedding.load_from_checkpoint('overfit/' + args.problem + '/model.ckpt')
    
    goal = torch.tensor([mapping['<START>'], mapping['$false'], mapping['<END>']] + [mapping['<PAD>']] * (args.seq_len - 3), dtype=torch.long)
    goal_embedding = embedding(goal.unsqueeze(0))

    with VampireInteractive(VAMPIRE_PATH, 'overfit/' + args.problem + '/problem.p') as interactive:
        seen = 0
        similarities = SortedList()
        premise_count = []
        
        while not interactive.finished and interactive.step_count < args.max_steps:
            new_clauses = interactive.problem.clauses[seen:]
            if new_clauses:
                tokens = [clause.tokenize(config, mapping) for clause in new_clauses]
                __x = torch.zeros((len(new_clauses), args.seq_len), dtype=torch.long)
                for i, clause in enumerate(tokens):
                    for j, token in enumerate(clause[:args.seq_len]):
                        __x[i, j] = clause[j]
                with torch.no_grad():
                    sim = torch.nn.functional.cosine_similarity(embedding(__x), goal_embedding, dim=-1)
                
                for i, (s, prev) in enumerate(zip(sim, interactive.tree[seen:])):
                    premise_count.append(1 + sum(premise_count[idx] for idx in prev))
                    similarities.add((s.item() / premise_count[-1]**0.5, seen + i))
                    
                sim.detach()
                
                seen = len(interactive.problem.clauses)
            
            next_clause = similarities.pop(-1)[1]
            interactive.step(next_clause)
        
        with open('overfit/' + args.problem + '/overfit_proof.p', 'w') as f:
            f.write(interactive.proof)
        