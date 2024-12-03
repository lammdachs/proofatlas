import argparse
from dotenv import load_dotenv
import os
import signal
import sys
from tqdm import tqdm

from foreduce.tptp.parser import read_file as read_tptp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_size", type=int, default=2**12)
    parser.add_argument("--max_nodes", type=int, default=2**10)
    parser.add_argument("--max_depth", type=int, default=4)
    args = parser.parse_args()
    
    def signal_handler(signal, frame):
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    
    load_dotenv()
    TPTP = os.getenv('TPTP')

    success = 0
    for dir, problem in (pbar := tqdm(
        [
            (d, p)
            for d in os.listdir(f'{TPTP}/Problems/')
            for p in os.listdir(f'{TPTP}/Problems/{d}')
            if p.endswith('.p')
        ],
        desc=f'Selecting Problems'
    )):
        try:
            p = read_tptp(
                f'{TPTP}/Problems/{dir}/{problem}',
                include_path=TPTP,
                max_size=args.max_size
            )
        except:
            continue
        if p.depth() > args.max_depth:
            continue
        graph, _, _ = p.to_graph()
        if graph.number_of_nodes() > args.max_nodes:
            continue
        os.makedirs(f'./problems/{dir}', exist_ok=True)
        with open(f'./problems/{dir}/{problem}', 'w') as f:
            f.write(p.to_tptp())
        success += 1
        pbar.set_postfix({
            'Success': success,
        })
