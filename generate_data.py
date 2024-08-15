from itertools import zip_longest
import os
import subprocess
import torch
from tqdm import tqdm

from foreduce.data.data import VampireProofs
from foreduce.transformer.tokenizer import TokenConfig
from foreduce.tptp.parser import read_file as read_tptp
from foreduce.vampire.parser import read_file as read_vampire

TPTP_PATH = '/home/apluska/TPTP-v8.2.0/'
VAMPIRE_PATH = '/home/apluska/.vampire/bin/vampire_z3_rel_static_casc2023_6749'
datapoints_per_proof = 64

total, success = 0, 0
num_functions = []
for dir, file in (pbar := tqdm([(dir, file) for dir in sorted(os.listdir(TPTP_PATH + 'Problems')) for file in sorted(os.listdir(TPTP_PATH + 'Problems/' + dir))])):
    current = file
    pbar.set_description(f'Selected {success}/{total} Problems, parsing {dir}/{file}')
    if not file.endswith('.p'):
        continue
    try:
        total += 1
        problem = read_tptp(TPTP_PATH + 'Problems/' + dir + '/' + file, include_path=TPTP_PATH, max_size=100_000)
        _variables = max(len(clause.variables()) for clause in problem.clauses)
        _symbols = []
        for s in problem.function_symbols() | problem.predicate_symbols():
            if s.arity > 8:
                break
            if len(_symbols) <= s.arity:
                _symbols += [0 for _ in range(s.arity + 1 - len(_symbols))]
            _symbols[s.arity] += 1
        else:
            if any(count > 16 for count in _symbols):
                continue
            num_functions = [max(a, b) for a, b in zip_longest(num_functions, _symbols, fillvalue=0)]
            success += 1
            os.makedirs('./problems/' + dir, exist_ok=True)
            with open('./problems/' + dir + '/' + file, 'w') as f:
                f.write(problem.to_tptp())
    except Exception as e:
        continue

total, success = 0, 0
for dir, file in (pbar := tqdm([(dir, file) for dir in sorted(os.listdir('./problems')) for file in sorted(os.listdir('./problems/' + dir))])):
    pbar.set_description(f'Succesfully proved {success}/{total} Problems, proving {dir}/{file}')
    args = [VAMPIRE_PATH, './problems/' + dir + '/' + file,  '--show_new', 'on', '-t', '1', '--avatar', 'off', '--proof', 'off']
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=5)
    except subprocess.TimeoutExpired:
        continue
    if result.returncode == 0:
        if 'Refutation found.' in result.stdout:
            success += 1    
            os.makedirs('./proofs/' + dir, exist_ok=True)
            with open('./proofs/' + dir + '/' + file, 'w') as f:
                f.write(result.stdout)
    total += 1

total, success

config = TokenConfig(num_functions=num_functions)
dataset = VampireProofs(config=config, proofs=success*datapoints_per_proof, max_steps=1024, max_tokens=128)

for dir, file in (pbar := tqdm([(dir, file) for dir in sorted(os.listdir('./proofs')) for file in sorted(os.listdir('./proofs/' + dir))])):
    pbar.set_description(f'Parsing proof {dir}/{file}')
    problem, tree = read_vampire('./proofs/' + dir + '/' + file)
    for i in range(datapoints_per_proof):
        pbar.set_description(f'Converting proof of {dir}/{file} to {i+1}/{datapoints_per_proof} datapoints')
        dataset.add_proof(problem, tree, goal='last')

dataset.to_file('./proofs_last_test.pt')
