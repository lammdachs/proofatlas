import argparse
import os
import subprocess
import torch
from tqdm import tqdm

from foreduce.fol.logic import Problem
from foreduce.transformer.tokenizer import ProblemTokenizer
from foreduce.transformer.model import Model
from foreduce.vampire.parser import read_string as read_vampire

VAMPIRE_PATH = '/home/apluska/.vampire/bin/vampire_z3_rel_static_casc2023_6749'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--select_from", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=60)
    
    if parser.model is not None:
        model = Model.load_from_checkpoint('./models/' + parser.model)
        
    for dir, file in (pbar := tqdm([(dir, file) for dir in sorted(os.listdir('./problems')) for file in sorted(os.listdir('./problems/' + dir))])):
        if os.path.exists('./proofs/' + dir + '/' + file):
            continue
        os.makedirs('./results/' + dir, exist_ok=True)
        if model is None:
            result = subprocess.run([VAMPIRE_PATH, './problems/' + dir + '/' + file,  '--show_new', 'on',\
                '-t', str(parser.timeout), '--avatar', 'off', '--proof', 'off'], capture_output=True, text=True, timeout=parser.timeout)
            if result.returncode == 0 and 'Refutation found.' in result.stdout:
                with open('./results/' + dir + '/' + file, 'a') as f:
                    lines = len(result.stdout.split('\n'))
                    f.write(f"Vampire: {len(result.stdout.split('\n')) - 13}" + '\n')
            else:
                with open('./results/' + dir + '/' + file, 'a') as f:
                    f.write(f"Vampire: X" + '\n')
        else:
            result = subprocess.run([VAMPIRE_PATH, './problems/' + dir + '/' + file,  '--show_new', 'on',\
                '-t', 1, '--avatar', 'off', '--proof', 'off'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and 'Refutation found.' in result.stdout:
                with open('./results/' + dir + '/' + file, 'a') as f:
                    lines = len(result.stdout.split('\n'))
                    f.write(f"{model}: {len(result.stdout.split('\n')) - 13}" + '\n')
            else:
                problem, _ = read_vampire('./problems/' + dir + '/' + file)
                mapping = problem.random_mapping(model.config)
                x, mapping = ProblemTokenizer(model.config, parser.select_from, parser.topk, 42)(problem, mapping)
                model.formula_embedding.update_config(model.config)
                logits = model(x.unsqueeze(0)).squeeze(0)
                _, indices = torch.topk(logits, parser.topk)
                with open('./tmp.p/', 'w') as f:
                    f.write(Problem(*[problem.clauses[i] for i in indices]).to_tptp())
                result = subprocess.run([VAMPIRE_PATH, './tmp.p',  '--show_new', 'on',\
                    '-t', str(parser.timeout), '--avatar', 'off', '--proof', 'off'], capture_output=True, text=True, timeout=parser.timeout)
                if result.returncode == 0 and 'Refutation found.' in result.stdout:
                    with open('./results/' + dir + '/' + file, 'a') as f:
                        lines = len(result.stdout.split('\n'))
                        f.write(f"{model}: {1024 + len(result.stdout.split('\n')) - 13}" + '\n')
                else:
                    with open('./results/' + dir + '/' + file, 'a') as f:
                        f.write(f"{model}: X" + '\n')