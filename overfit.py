VAMPIRE_PATH = '/home/apluska/.vampire/bin/vampire_z3_rel_static_casc2023_6749'
TPTP_PATH = '/home/apluska/TPTP-v8.2.0/'
MAX_STEP = 100


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

import os
from collections import defaultdict
from tqdm.auto import tqdm
from lightning import Trainer
import wandb
from lightning.pytorch.loggers import WandbLogger
import torch
from sortedcontainers import SortedList
import time
import signal
import sys
import json

from foreduce.vampire.vampire import VampireInteractive
from foreduce.vampire.vampire import VampireAutomatic
from foreduce.data.data import ProofTokens
from foreduce.transformer.tokenizer import TokenConfig
from foreduce.tptp.parser import read_file as read_tptp
from foreduce.vampire.parser import read_file as read_vampire
from foreduce.transformer.embedding import FormulaEmbedding
from torch.utils.data import DataLoader


def signal_handler(sig, frame):
    signal.signal(sig, signal.SIG_IGN)
    if os.path.exists('.mapping.json'):
        os.remove('.mapping.json')
    if os.path.exists('.dataset.pt'):
        os.remove('.dataset.pt')
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    starttime = time.time()

    if os.path.exists('.mapping.json'):
        with open('.mapping.json') as f:
            mapping = json.load(f)
    else:
        symbols = set()

        for p in tqdm(problems):
            problem = read_tptp(TPTP_PATH + 'Problems/GRP/' + p, include_path=TPTP_PATH, max_size=10_000)
            symbols.update(problem.function_symbols() | problem.predicate_symbols())

        if not os.path.exists('problems/' + p):
            with open('problems/' + p, 'w') as f:
                f.write(problem.to_tptp())

        arity_dict = defaultdict(list)

        for s in symbols:
            arity_dict[s.arity].append(s.name)

        arity_list = list(arity_dict.values())
        config = TokenConfig(num_functions=[len(s) for s in arity_list])
        function_mapping = config.random_function_mapping(arity_list)
        mapping = config.reserved_token_mapping | config.random_function_mapping(arity_list)
        with open('.mapping.json', 'w') as f:
            json.dump(mapping, f)

    for p in tqdm(problems):
        if not os.path.exists('.dataset.pt'):
            dataset = ProofTokens(config, seq_len=24)
            if os.path.exists('proofs/vampire/' + p):
                proof, tree, _ = read_vampire('proofs/vampire/' + p)
                dataset.add_proof(proof, tree, mapping)
            else:
                vampire = VampireAutomatic(VAMPIRE_PATH, 'problems/' + p, selection='10')
                vampire.run() 
                dataset.add_proof(vampire.problem, vampire.tree, mapping)
                os.makedirs('proofs/vampire/', exist_ok=True)
                with open('proofs/vampire/' + p, 'w') as f:
                    f.write(vampire.proof)
            dataset.to_file('.dataset.pt')
        else:
            dataset = ProofTokens.from_file('.dataset.pt')
            
        embedding = FormulaEmbedding(dataset.config, seq_len=24, dim=1536, n_layers=24, n_heads=16)
        data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=16, persistent_workers=True)

        torch.set_float32_matmul_precision('medium')
        logger = WandbLogger(project='vampire_select', name=p, group='overfit {starttime:.0f}')
        trainer = Trainer(max_epochs=1, logger=logger, accumulate_grad_batches=128, log_every_n_steps=1, devices=4)
        trainer.fit(embedding, data_loader)
        wandb.finish()
        
        if os.path.exists('.dataset.pt'):
            os.remove('.dataset.pt')
        
        goal = torch.tensor([mapping['<START>'], mapping['$false'], mapping['<END>']] + [mapping['<PAD>']] * 21, dtype=torch.long)
        goal_embedding = embedding(goal.unsqueeze(0))

        with VampireInteractive(VAMPIRE_PATH, 'problems/' + p) as interactive:
            seen = 0
            similarities = SortedList()
            premise_count = []
            
            while not interactive.finished and interactive.step_count < MAX_STEP:
                new_clauses = interactive.problem.clauses[seen:]
                if new_clauses:
                    tokens = [clause.tokenize(dataset.config, mapping) for clause in new_clauses]
                    __x = torch.zeros((len(new_clauses), 24), dtype=torch.long)
                    for i, clause in enumerate(tokens):
                        for j, token in enumerate(clause[:24]):
                            __x[i, j] = clause[j]
                    with torch.no_grad():
                        sim = torch.nn.functional.cosine_similarity(embedding(__x), goal_embedding, dim=-1)
                    
                    for i, (s, p) in enumerate(zip(sim, interactive.tree[seen:])):
                        premise_count.append(1 + sum(premise_count[idx] for idx in p))
                        similarities.add((s.item() / premise_count[-1]**0.5, seen + i))
                        
                    sim.detach()
                    
                    seen = len(interactive.problem.clauses)
                
                next_clause = similarities.pop(-1)[1]
                interactive.step(next_clause)
            
            with open('proofs/embedding/' + p, 'w') as f:
                f.write(interactive.proof)

    if os.path.exists('.mapping.json'):
        os.remove('.mapping.json')
        