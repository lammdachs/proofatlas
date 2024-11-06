VAMPIRE_PATH = '/home/apluska/.vampire/bin/vampire_z3_rel_static_casc2023_6749'
TPTP_PATH = '/home/apluska/TPTP-v8.2.0/'


train_problems = [
    'GRP/GRP002-1.p',
    'GRP/GRP003-1.p',
    'GRP/GRP004-1.p',
    'GRP/GRP005-1.p',
    'GRP/GRP006-1.p',
    'GRP/GRP008-1.p',
    'GRP/GRP009-1.p',
    'GRP/GRP010-4.p',
    'GRP/GRP011-4.p',
    'GRP/GRP013-1.p',
    'GRP/GRP017-1.p',
    'GRP/GRP018-1.p',
    'GRP/GRP019-1.p',
    'GRP/GRP020-1.p',
    'GRP/GRP021-1.p',
    'GRP/GRP023-1.p'
]

test_problems = [
    'GRP/GRP001-1.p',
    'GRP/GRP007-1.p',
    'GRP/GRP012-1.p',
    'GRP/GRP022-1.p'
]

import argparse
import os
from collections import defaultdict
import json
from subprocess import Popen, PIPE, STDOUT
import torch
from sortedcontainers import SortedList

from foreduce.vampire.vampire import VampireInteractive, VampireAutomatic
from foreduce.data.data import ProofTokens
from foreduce.transformer.tokenizer import TokenConfig
from foreduce.tptp.parser import read_file as read_tptp
from foreduce.vampire.parser import read_file as read_vampire
from foreduce.transformer.embedding import FormulaEmbedding


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accumulate_grad_batches", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=96)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=256)
    parser.add_argument("--clause_selection", type=str, default='10')
    parser.add_argument("--cont", type=bool, default=False)
    args = parser.parse_args()
    
    if not os.path.exists('generalization/config.json') or not os.path.exists('generalization/mapping.json'):
        os.makedirs('generalization/problems/GRP/', exist_ok=True)
        symbols = set()
        for problem in train_problems + test_problems:
            if not os.path.exists('generalization/problems/' + problem):
                p = read_tptp(TPTP_PATH + 'Problems/' + problem, include_path=TPTP_PATH, max_size=100_000)
                with open('generalization/problems/' + problem, 'w') as f:
                    f.write(p.to_tptp())
            else:
                p = read_tptp('generalization/problems/' + problem)        
            symbols.update(p.function_symbols() | p.predicate_symbols())
        
        arity_dict = defaultdict(list)
        for s in symbols:
            arity_dict[s.arity].append(s.name)
        arity_list = list(arity_dict.values())
        config = TokenConfig(num_functions=[len(s) for s in arity_list])
            
        with open('generalization/config.json', 'w') as f:
            json.dump(config.to_dict(), f)

        function_mapping = config.random_function_mapping(arity_list)
        mapping = config.reserved_token_mapping | config.random_function_mapping(arity_list)
        with open('generalization/mapping.json', 'w') as f:
            json.dump(mapping, f)
    else:
        with open('generalization/config.json') as f:
            config = TokenConfig.from_dict(json.load(f))
        with open('generalization/mapping.json') as f:
            mapping = json.load(f)

    if not os.path.exists('generalization/proofs/'):
        train_dataset = ProofTokens(config, seq_len=args.seq_len)
        test_dataset = ProofTokens(config, seq_len=args.seq_len)
        os.makedirs('generalization/proofs/GRP/', exist_ok=True)
        for problem in train_problems + test_problems:
            if not os.path.exists('generalization/proofs/' + problem):
                vampire = VampireAutomatic(
                    VAMPIRE_PATH,
                    'generalization/problems/' + problem,
                    selection=args.clause_selection,
                    activation_limit=64,
                )
                vampire.run() 
                with open('generalization/proofs/' + problem, 'w') as f:
                    f.write(vampire.proof)
            if problem in train_problems:
                train_dataset.add_proof(vampire.problem, vampire.tree, mapping)
            elif problem in test_problems:
                test_dataset.add_proof(vampire.problem, vampire.tree, mapping)
        train_dataset.to_file('generalization/train_dataset.pt')
        test_dataset.to_file('generalization/test_dataset.pt')
        
    if not os.path.exists('generalization/model.ckpt') or args.cont:
        with Popen(
            [
                'python', '-u', 'train_generalization.py',
                '--epochs', str(args.epochs),
                '--batch_size', str(args.batch_size), '--accumulate_grad_batches', str(args.accumulate_grad_batches),
                '--seq_len', str(args.seq_len),
                '--dim', str(args.dim), '--n_layers', str(args.n_layers), '--n_heads', str(args.n_heads),
            ],
            bufsize=1, universal_newlines=True, stdout=PIPE, stderr=STDOUT
        ) as proc:
            for line in proc.stdout:
                print(line, end='')
    
    embedding = FormulaEmbedding.load_from_checkpoint('generalization/model.ckpt')
    
    os.makedirs('generalization/model_proofs/GRP', exist_ok=True)
    
    goal = torch.tensor([mapping['<START>'], mapping['$false'], mapping['<END>']] + [mapping['<PAD>']] * (args.seq_len - 3), dtype=torch.long)
    goal_embedding = embedding(goal.unsqueeze(0))

    for problem in train_problems + test_problems:
        with VampireInteractive(VAMPIRE_PATH, 'generalization/problems/' + problem) as interactive:
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
            
            with open('generalization/model_proofs/' + problem, 'w') as f:
                f.write(interactive.proof)
            